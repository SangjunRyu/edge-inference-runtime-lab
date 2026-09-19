// The staged pipeline. See scheduler.hpp for what it is and why.

#include "scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <memory>

namespace eirl {
namespace {

/// One frame's result once its buffers have gone back: small, copyable, and
/// carrying nothing that owns memory, so the emit queue costs nothing to move
/// through.
struct Done {
    std::uint64_t seq = 0;
    bool ok = true;
};

struct Shared {
    InputPool in_pool;
    OutputPool out_pool;
    RingQueue<FrameContext> q_pre, q_infer, q_post;
    RingQueue<Done> q_emit;

    std::atomic<std::uint64_t> accepted{0}, completed{0}, emitted{0};
    std::atomic<std::uint64_t> mismatched{0}, duplicated{0};
    std::atomic<std::uint32_t> inflight{0}, peak_inflight{0};
    std::atomic<std::size_t> peak_q_pre{0}, peak_q_infer{0}, peak_q_post{0};
    std::atomic<int> pre_left, infer_left, post_left;

    Shared(const SchedulerConfig& c)
        : in_pool(c.buffers, c.bytes),
          out_pool(c.buffers, c.bytes),
          q_pre(c.queue_capacity),
          q_infer(c.queue_capacity),
          q_post(c.queue_capacity),
          q_emit(c.queue_capacity + static_cast<std::size_t>(c.post_workers) + 1),
          pre_left(c.pre_workers),
          infer_left(c.infer_workers),
          post_left(c.post_workers) {}

    void bump_inflight() {
        const std::uint32_t now = inflight.fetch_add(1, std::memory_order_relaxed) + 1;
        std::uint32_t seen = peak_inflight.load(std::memory_order_relaxed);
        while (now > seen &&
               !peak_inflight.compare_exchange_weak(seen, now, std::memory_order_relaxed)) {
        }
    }

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

    // Sized from the ownership rule, not from a guess: an unemitted frame still
    // holds a lease, so no more indices can be outstanding than the ceiling.
    const std::uint32_t cap = ceiling(cfg);
    RingReorder<Done> reorder(cap + 1);

    auto seen = std::make_unique<std::atomic<std::uint8_t>[]>(cfg.frames ? cfg.frames : 1);
    for (std::uint64_t i = 0; i < cfg.frames; ++i) seen[i].store(0);

    const auto t0 = std::chrono::steady_clock::now();

    std::thread reader([&] {
        for (std::uint64_t seq = 0; seq < accept_limit; ++seq) {
            InputLease lease = sh->in_pool.acquire();
            if (!lease) break;
            sh->bump_inflight();
            stages.read(seq, lease.data(), lease.size());

            FrameContext ctx;
            ctx.seq = seq;
            ctx.in = std::move(lease);
            if (!sh->q_pre.push(std::move(ctx))) {
                sh->inflight.fetch_sub(1, std::memory_order_relaxed);
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
                if (!sh->q_infer.push(std::move(ctx))) break;
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
                if (!ctx.out) break;
                // The input lease stays held across this call. That is the rule
                // repro/cpp/01 justified: a reader handed a pointer looks at it
                // when the work completes, not when it was handed over.
                stages.infer(ctx.in.data(), ctx.out.data(), ctx.in.size());
                sh->completed.fetch_add(1, std::memory_order_relaxed);
                if (!sh->q_post.push(std::move(ctx))) break;
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
                sh->inflight.fetch_sub(1, std::memory_order_relaxed);
                if (!sh->q_emit.push(d)) break;
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
    // Taken from the pool, not from a counter bracketed around the frame. The
    // frame-bracketed version reported 9 with 8 buffers on the first run of
    // this harness, because a buffer can be taken again in the window between
    // its release and the counter's decrement. repro/cpp/06 is that mistake,
    // kept runnable.
    s.peak_inflight = sh->in_pool.peak_outstanding();
    s.peak_inflight_frame_bracketed = sh->peak_inflight.load();
    s.predicted_ceiling = cap;
    s.peak_q_pre = sh->peak_q_pre.load();
    s.peak_q_infer = sh->peak_q_infer.load();
    s.peak_q_post = sh->peak_q_post.load();
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
