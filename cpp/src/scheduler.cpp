// The staged pipeline. See scheduler.hpp for what it is and why.

#include "scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>

namespace eirl {
namespace {

/// One frame's result once its buffers have gone back: small, copyable, and
/// carrying nothing that owns memory, so the emit queue costs nothing to move
/// through.
struct Done {
    std::uint64_t seq = 0;
    bool ok = true;
};

/// How many frames are inside the pipeline, counted at one lock.
///
/// The obvious implementation -- an atomic incremented here and decremented
/// there -- is what produced a peak of nine against eight buffers on the first
/// run of this scheduler. Two atomics are two events; a high-water mark taken
/// across them belongs to neither. Here `enter`, `leave` and the peak update
/// happen under the same mutex, so the recorded maximum is a value the counter
/// actually held.
///
/// What it counts is stated precisely, because the bound depends on it: a frame
/// is inside the pipeline from the moment the reader takes an input buffer for
/// it until the postprocess worker finishes with it. The emit queue and the
/// emitter are past that boundary and hold no resources.
class InflightGauge {
public:
    void enter() {
        std::lock_guard<std::mutex> lk(m_);
        ++now_;
        if (now_ > peak_) peak_ = now_;
    }
    void leave() {
        std::lock_guard<std::mutex> lk(m_);
        if (now_) --now_;
    }
    std::uint32_t peak() const {
        std::lock_guard<std::mutex> lk(m_);
        return peak_;
    }

private:
    mutable std::mutex m_;
    std::uint32_t now_ = 0, peak_ = 0;
};

struct Shared {
    InputPool in_pool;
    OutputPool out_pool;
    RingQueue<FrameContext> q_pre, q_infer, q_post;
    RingQueue<Done> q_emit;
    InflightGauge gauge;

    std::atomic<std::uint64_t> accepted{0}, completed{0}, emitted{0};
    std::atomic<std::uint64_t> mismatched{0}, duplicated{0};
    std::atomic<std::size_t> peak_q_pre{0}, peak_q_infer{0}, peak_q_post{0};
    std::atomic<int> pre_left, infer_left, post_left;

    Shared(const SchedulerConfig& c)
        : in_pool(c.buffers, c.bytes),
          out_pool(c.out_buffers ? c.out_buffers : c.buffers,
                   c.out_bytes ? c.out_bytes : c.bytes),
          q_pre(c.q_pre),
          q_infer(c.q_infer),
          q_post(c.q_post),
          q_emit(c.q_post + static_cast<std::size_t>(c.post_workers) + 1),
          pre_left(c.pre_workers),
          infer_left(c.infer_workers),
          post_left(c.post_workers) {}

    static void bump(std::atomic<std::size_t>& peak, std::size_t now) {
        std::size_t seen = peak.load(std::memory_order_relaxed);
        while (now > seen &&
               !peak.compare_exchange_weak(seen, now, std::memory_order_relaxed)) {
        }
    }
};

}  // namespace

SchedulerStats run_pipeline(const SchedulerConfig& cfg,
                            const Stages& stages,
                            const std::function<void()>& on_warm) {
    const std::uint64_t accept_limit =
        cfg.interrupt_at ? std::min(cfg.interrupt_at, cfg.frames) : cfg.frames;

    auto sh = std::make_unique<Shared>(cfg);

    // Sized from the bound that actually applies to it. The first version used
    // the ceiling -- min(resource, structural) -- which bounds frames HOLDING A
    // BUFFER. Completions reach the reorder after their buffers have gone back,
    // so the indices outstanding here are bounded by pipeline occupancy plus
    // whatever is queued for the emitter, and that is larger. Undersizing it did
    // not merely lose ordering: it aliased occupied slots and hung the drain.
    const std::size_t reorder_cap =
        static_cast<std::size_t>(structural_capacity(cfg))
        + cfg.q_post + static_cast<std::size_t>(cfg.post_workers) + 2;
    RingReorder<Done> reorder(reorder_cap);

    auto seen = std::make_unique<std::atomic<std::uint8_t>[]>(cfg.frames ? cfg.frames : 1);
    for (std::uint64_t i = 0; i < cfg.frames; ++i) seen[i].store(0);

    const auto t0 = std::chrono::steady_clock::now();

    std::thread reader([&] {
        for (std::uint64_t seq = 0; seq < accept_limit; ++seq) {
            InputLease lease = sh->in_pool.acquire();
            if (!lease) break;
            sh->gauge.enter();
            stages.read(seq, lease.data(), lease.size());

            FrameContext ctx;
            ctx.seq = seq;
            ctx.in = std::move(lease);
            if (!sh->q_pre.push(std::move(ctx))) {
                sh->gauge.leave();
                break;
            }
            Shared::bump(sh->peak_q_pre, sh->q_pre.size());
            sh->accepted.fetch_add(1, std::memory_order_relaxed);
            if (seq + 1 == cfg.warmup && on_warm) on_warm();
        }
        sh->q_pre.close();
    });

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(cfg.pre_workers + cfg.infer_workers +
                                             cfg.post_workers + 1));

    for (int i = 0; i < cfg.pre_workers; ++i) {
        threads.emplace_back([&] {
            for (;;) {
                auto got = sh->q_pre.pop();
                if (!got) break;
                FrameContext ctx = std::move(*got);
                stages.preprocess(ctx.in.data(), ctx.in.size());
                if (!sh->q_infer.push(std::move(ctx))) { sh->gauge.leave(); break; }
                Shared::bump(sh->peak_q_infer, sh->q_infer.size());
            }
            if (sh->pre_left.fetch_sub(1) == 1) sh->q_infer.close();
        });
    }

    for (int i = 0; i < cfg.infer_workers; ++i) {
        threads.emplace_back([&] {
            for (;;) {
                auto got = sh->q_infer.pop();
                if (!got) break;
                FrameContext ctx = std::move(*got);
                ctx.out = sh->out_pool.acquire();
                if (!ctx.out) { sh->gauge.leave(); break; }
                // The input lease stays held across this call. That is the rule
                // repro/cpp/01 justified: a reader handed a pointer looks at it
                // when the work completes, not when it was handed over.
                stages.infer(ctx.in.data(), ctx.out.data(), ctx.in.size());
                sh->completed.fetch_add(1, std::memory_order_relaxed);
                if (!sh->q_post.push(std::move(ctx))) { sh->gauge.leave(); break; }
                Shared::bump(sh->peak_q_post, sh->q_post.size());
            }
            if (sh->infer_left.fetch_sub(1) == 1) sh->q_post.close();
        });
    }

    for (int i = 0; i < cfg.post_workers; ++i) {
        threads.emplace_back([&] {
            for (;;) {
                auto got = sh->q_post.pop();
                if (!got) break;
                FrameContext ctx = std::move(*got);
                const bool ok = stages.postprocess(ctx.seq, ctx.out.data(), ctx.out.size());
                if (!ok) sh->mismatched.fetch_add(1, std::memory_order_relaxed);
                if (seen[ctx.seq].fetch_add(1, std::memory_order_relaxed) != 0) {
                    sh->duplicated.fetch_add(1, std::memory_order_relaxed);
                }
                const Done d{ctx.seq, ok};
                ctx.out.release();
                ctx.in.release();
                sh->q_emit.push(d);
                // Last, and inside the worker's own iteration: until this
                // returns the frame still occupies a post-worker place, which
                // is what keeps the gauge inside the structural bound.
                sh->gauge.leave();
            }
            if (sh->post_left.fetch_sub(1) == 1) sh->q_emit.close();
        });
    }

    // One emitter, because restoring an order needs a single place to do it.
    threads.emplace_back([&] {
        auto sink = [&](std::uint64_t, Done&&) {
            sh->emitted.fetch_add(1, std::memory_order_relaxed);
        };
        for (;;) {
            auto got = sh->q_emit.pop();
            if (!got) break;
            reorder.push(got->seq, *got, sink);
        }
        reorder.drain(sink);
    });

    reader.join();
    for (auto& t : threads) t.join();

    const auto t1 = std::chrono::steady_clock::now();

    sh->in_pool.close();
    sh->out_pool.close();

    SchedulerStats s;
    s.accepted = sh->accepted.load();
    s.completed = sh->completed.load();
    s.emitted = sh->emitted.load();
    s.mismatched = sh->mismatched.load();
    s.duplicated = sh->duplicated.load();
    s.out_of_order = reorder.out_of_order();
    s.max_displacement = reorder.max_displacement();
    s.reorder_overflows = reorder.overflows();
    s.reorder_max_held = reorder.max_held();

    s.peak_pipeline_inflight = sh->gauge.peak();
    s.peak_in_leases = sh->in_pool.peak_outstanding();
    s.peak_out_leases = sh->out_pool.peak_outstanding();
    s.peak_q_pre = sh->peak_q_pre.load();
    s.peak_q_infer = sh->peak_q_infer.load();
    s.peak_q_post = sh->peak_q_post.load();

    s.structural_capacity = structural_capacity(cfg);
    s.resource_capacity = resource_capacity(cfg);

    s.in_outstanding = sh->in_pool.outstanding();
    s.out_outstanding = sh->out_pool.outstanding();
    s.in_acquires = sh->in_pool.acquires();
    s.in_releases = sh->in_pool.releases();
    s.out_acquires = sh->out_pool.acquires();
    s.out_releases = sh->out_pool.releases();
    s.double_releases = sh->in_pool.double_releases() + sh->out_pool.double_releases();
    s.seconds = std::chrono::duration<double>(t1 - t0).count();
    return s;
}

}  // namespace eirl
