// W2 in C++ -- the overlapped pipeline.
//
//     What changes when stages overlap, and how much of what Python measured
//     was Python?
//
// Structure follows the Python async runner, which follows DX-APP's:
//
//   read_worker -> [read_q] -> preprocess_worker --start_async--> (OV callback)
//                                                                      |
//              main <- [result_q] <- postprocess_worker <--------------+ [output_q]
//
// Submission and collection are separate, which is what makes this
// asynchronous rather than merely threaded: the preprocess worker calls
// `ov_infer_request_start_async` and moves straight to the next frame, and the
// completion callback -- running on an OpenVINO thread, not one of ours --
// hands the slot to a different worker. A single worker that submitted and
// then waited would block on collection and overlap nothing.
//
// In-flight depth is the number of infer requests, because one request runs
// one inference at a time. Each request owns its input and output buffers, and
// a slot is not reused until postprocess has read its output -- which is what
// keeps the input buffer alive for as long as OpenVINO is reading it.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "harness.hpp"
#include "ov_backend.hpp"
#include "pipeline.hpp"
#include "postprocess.hpp"
#include "reorder.hpp"
#include "safe_queue.hpp"
#include "sources.hpp"

using namespace eirl;
using nlohmann::json;

namespace {

const std::vector<std::string> kService = {"read", "preprocess", "inference",
                                           "postprocess"};
const std::vector<std::string> kWaits = {"q_read", "q_inflight", "q_output",
                                         "q_reorder"};

struct Args {
    std::string model, device = "CPU", source, tag = "cppw2", out_dir = "results";
    long long frames = 40000;
    double duration_s = 10.0, warmup_s = 2.0, cooldown = 0.0;
    int warmup = 0, top_k = 5, inflight = 4, queue_size = 4, repeat = 1;
    int reorder_cap = 0, stall_every = 0, post_workers = 1;
    double stall_ms = 40.0;
    std::string dump_topk;
    bool reorder = true, fused = false;
};

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s --model M.onnx [--device CPU|GPU|NPU] [--source clip.mp4]\n"
        "          [--inflight N] [--queue-size N] [--reorder on|off]\n"
        "          [--reorder-cap N] [--frames N] [--duration-s S]\n"
        "          [--warmup-s S] [--top-k K] [--repeat N] [--cooldown S]\n"
        "          [--post-workers N] [--stall-every N] [--stall-ms M]\n"
        "          [--dump-topk FILE] [--fused] [--tag T] [--out-dir D]\n", argv0);
    std::exit(2);
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (k == "--model") a.model = val();
        else if (k == "--device") a.device = val();
        else if (k == "--source") a.source = val();
        else if (k == "--inflight") a.inflight = std::stoi(val());
        else if (k == "--queue-size") a.queue_size = std::stoi(val());
        else if (k == "--reorder") a.reorder = (val() != "off");
        else if (k == "--reorder-cap") a.reorder_cap = std::stoi(val());
        else if (k == "--frames") a.frames = std::stoll(val());
        else if (k == "--duration-s") a.duration_s = std::stod(val());
        else if (k == "--warmup") a.warmup = std::stoi(val());
        else if (k == "--warmup-s") a.warmup_s = std::stod(val());
        else if (k == "--top-k") a.top_k = std::stoi(val());
        else if (k == "--repeat") a.repeat = std::stoi(val());
        else if (k == "--cooldown") a.cooldown = std::stod(val());
        else if (k == "--post-workers") a.post_workers = std::stoi(val());
        else if (k == "--stall-every") a.stall_every = std::stoi(val());
        else if (k == "--stall-ms") a.stall_ms = std::stod(val());
        else if (k == "--dump-topk") a.dump_topk = val();
        else if (k == "--fused") a.fused = true;
        else if (k == "--tag") a.tag = val();
        else if (k == "--out-dir") a.out_dir = val();
        else usage(argv[0]);
    }
    if (a.model.empty()) usage(argv[0]);
    return a;
}

std::string stem(const std::string& p) {
    const size_t s = p.find_last_of("/\\");
    const size_t b = s == std::string::npos ? 0 : s + 1;
    const size_t d = p.find_last_of('.');
    return p.substr(b, (d == std::string::npos || d < b ? p.size() : d) - b);
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

/// One in-flight slot: a request, its buffers, and the frame riding on it.
struct Slot {
    RequestHandle req;
    std::vector<float> input, output;
    TensorHandle input_tensor;

    uint64_t index = 0;
    double t_start = 0.0, read = 0.0, preprocess = 0.0;
    double t_enq_read = 0.0, q_read = 0.0, q_inflight = 0.0;
    double t_submit = 0.0, inference = 0.0;
    double t_enq_output = 0.0, q_output = 0.0;
    double postprocess = 0.0, t_done_post = 0.0;
    std::vector<std::pair<int, float>> topk;

    class AsyncPipeline* owner = nullptr;
    // The callback descriptor must outlive the request that holds it.
    // ov_infer_request_set_callback takes a pointer; passing the address of a
    // stack local left it dangling as soon as the constructor loop advanced,
    // which worked by accident with one slot and hung with two.
    ov_callback_t cb{};
};

class AsyncPipeline {
public:
    AsyncPipeline(const OvBackend& backend, const Args& args)
        : backend_(backend), args_(args),
          read_q_(static_cast<size_t>(args.queue_size)),
          free_q_(static_cast<size_t>(args.inflight)),
          // Unbounded on purpose. The completion callback runs on an
          // OpenVINO thread, and blocking there deadlocks the runtime: the
          // callback waits for queue space while the postprocess worker's
          // `ov_infer_request_wait` waits for the callback to return. A
          // callback must never block. Backpressure is not lost by this --
          // it lives in the slot pool, which is the real in-flight limit.
          output_q_(SIZE_MAX),
          result_q_(static_cast<size_t>(args.queue_size)) {
        const size_t in_n = static_cast<size_t>(backend.height()) * backend.width() * 3;
        // Reserved, never resized: the callback holds a Slot* and a
        // reallocation would leave OpenVINO writing through a dangling pointer.
        slots_.reserve(static_cast<size_t>(args.inflight));
        for (int i = 0; i < args.inflight; ++i) {
            slots_.emplace_back();
            Slot& s = slots_.back();
            s.req = backend.make_request();
            s.input.assign(in_n, 0.0f);
            s.output.assign(backend.output_elements(), 0.0f);
            s.input_tensor = backend.wrap_input(s.input.data());
            s.owner = this;
            check(ov_infer_request_set_input_tensor(s.req.get(), s.input_tensor.get()),
                  "ov_infer_request_set_input_tensor");
            s.cb.callback_func = &AsyncPipeline::on_complete;
            s.cb.args = &s;
            check(ov_infer_request_set_callback(s.req.get(), &s.cb),
                  "ov_infer_request_set_callback");
            free_q_.push(i);
        }
    }

    json run(Source& source, Recorder& rec, json& order_out, json& reorder_out,
             std::ofstream* dump);

private:
    /// Runs on an OpenVINO thread. Does the least possible and never blocks.
    static void OPENVINO_C_API_CALLBACK on_complete(void* args) {
        Slot* s = static_cast<Slot*>(args);
        const double now = now_s();
        s->inference = now - s->t_submit;
        s->t_enq_output = now;
        s->owner->output_q_.push(s);  // unbounded: cannot block here
    }

    void read_worker(Source& source);
    void preprocess_worker();
    void postprocess_worker();

    const OvBackend& backend_;
    const Args& args_;
    SafeQueue<Slot*> read_q_;   // carries a slot already holding a frame
    SafeQueue<int> free_q_;
    SafeQueue<Slot*> output_q_, result_q_;
    std::vector<Slot> slots_;
    std::vector<cv::Mat> frames_;  // one staging frame per slot
    std::atomic<long long> submitted_{0};
    std::atomic<int> posts_running_{0};
    std::string error_;
};

void AsyncPipeline::read_worker(Source& source) {
    const double deadline = args_.duration_s > 0.0 ? now_s() + args_.duration_s : -1.0;
    uint64_t idx = 0;
    try {
        while (true) {
            if (deadline > 0.0 && now_s() >= deadline) break;

            // A free slot is taken *before* decoding, so the frame is decoded
            // straight into the buffer it will be inferred from rather than
            // copied into it later. Waiting for one is the in-flight
            // backpressure -- the thing the depth knob controls -- so it is
            // timed and recorded as `q_inflight` rather than disappearing into
            // the reader.
            const double t_slot = now_s();
            auto got = free_q_.pop();
            if (!got) break;
            const double slot_wait = now_s() - t_slot;
            Slot& s = slots_[static_cast<size_t>(*got)];

            const double t0 = now_s();
            if (!source.next(frames_[static_cast<size_t>(*got)])) {
                free_q_.push(*got);
                break;
            }
            const double t1 = now_s();
            s.index = idx++;
            // The frame's clock starts when its slot was requested: the wait
            // for capacity is part of that frame's latency, not free time.
            s.t_start = t_slot;
            s.q_inflight = slot_wait;
            s.read = t1 - t0;
            // Stamped before the push, because the consumer may dequeue it
            // before this thread runs again.
            s.t_enq_read = now_s();
            if (!read_q_.push(&s)) break;
        }
    } catch (const std::exception& e) {
        error_ = std::string("read_worker: ") + e.what();
    }
    read_q_.close();
}

void AsyncPipeline::preprocess_worker() {
    try {
        while (true) {
            auto got = read_q_.pop();
            if (!got) break;
            Slot* s = *got;
            s->q_read = now_s() - s->t_enq_read;

            const size_t slot_i = static_cast<size_t>(s - slots_.data());
            const double t = now_s();
            if (args_.fused) {
                preprocess_fused(frames_[slot_i], backend_.height(),
                                 backend_.width(), s->input.data());
            } else {
                preprocess(frames_[slot_i], backend_.height(), backend_.width(),
                           s->input.data());
            }
            s->preprocess = now_s() - t;

            // q_inflight was already measured in the reader, where the slot
            // is actually reserved.
            s->t_submit = now_s();
            submitted_.fetch_add(1);
            check(ov_infer_request_start_async(s->req.get()),
                  "ov_infer_request_start_async");
        }
    } catch (const std::exception& e) {
        error_ = std::string("preprocess_worker: ") + e.what();
    }
    // Wait for every outstanding completion before closing: submission returns
    // immediately, so at this point several frames are still inside the
    // runtime. Closing now would let the marker overtake them and their
    // completions would arrive after the consumer had gone -- the Python side
    // lost 4 frames of 200 to exactly that, with the count almost right and
    // nothing looking wrong.
    for (auto& s : slots_) {
        ov_infer_request_wait(s.req.get());
    }
    output_q_.close();
}

void AsyncPipeline::postprocess_worker() {
    try {
        while (true) {
            auto got = output_q_.pop();
            if (!got) break;
            Slot* s = *got;
            s->q_output = now_s() - s->t_enq_output;

            const double t = now_s();
            // The completion callback runs before the request leaves its
            // running state, so reading the output straight from it returns
            // REQUEST_BUSY (-8). `wait` returns immediately here -- the work is
            // already done -- and transitions the request to idle. It is the
            // collect half of the submit/collect split made explicit by the
            // API rather than by our structure.
            check(ov_infer_request_wait(s->req.get()), "ov_infer_request_wait");
            ov_tensor_t* raw = nullptr;
            check(ov_infer_request_get_output_tensor(s->req.get(), &raw),
                  "ov_infer_request_get_output_tensor");
            TensorHandle out(raw);
            void* data = nullptr;
            check(ov_tensor_data(out.get(), &data), "ov_tensor_data");
            s->topk = postprocess(static_cast<const float*>(data),
                                  s->output.size(), args_.top_k);
            if (args_.stall_every > 0 &&
                s->index % static_cast<uint64_t>(args_.stall_every) == 0) {
                // Make one frame deliberately heavier so a lighter successor
                // can overtake it. On a uniform workload completions rarely
                // reorder and the correctness question never gets exercised:
                // zero violations then says only that nothing was tried.
                std::this_thread::sleep_for(
                    std::chrono::duration<double, std::milli>(args_.stall_ms));
            }
            s->postprocess = now_s() - t;
            s->t_done_post = now_s();
            if (!result_q_.push(s)) break;
        }
    } catch (const std::exception& e) {
        error_ = std::string("postprocess_worker: ") + e.what();
    }
    // Several of these may be running; the queue closes when the last leaves,
    // or the consumer would stop while another worker still has frames.
    if (posts_running_.fetch_sub(1) == 1) result_q_.close();
}

json AsyncPipeline::run(Source& source, Recorder& rec, json& order_out,
                        json& reorder_out, std::ofstream* dump) {
    frames_.assign(slots_.size(), cv::Mat());

    const size_t cap = args_.reorder_cap > 0
                           ? static_cast<size_t>(args_.reorder_cap)
                           : std::max<size_t>(16, 4 * (static_cast<size_t>(args_.inflight)
                                                       + static_cast<size_t>(args_.queue_size)));
    ReorderBuffer<Slot*> buf(cap);
    OrderChecker checker;

    std::thread t_read(&AsyncPipeline::read_worker, this, std::ref(source));
    std::thread t_pre(&AsyncPipeline::preprocess_worker, this);
    // More than one collector lets a light frame finish ahead of a heavy one,
    // which is the only way completions here can arrive out of order -- and
    // therefore the only way the reorder buffer is exercised rather than
    // merely present.
    const int n_post = std::max(1, args_.post_workers);
    posts_running_.store(n_post);
    std::vector<std::thread> t_posts;
    for (int i = 0; i < n_post; ++i) {
        t_posts.emplace_back(&AsyncPipeline::postprocess_worker, this);
    }

    auto emit = [&](Slot* s) {
        checker.observe(s->index);
        if (dump) {
            json line = {{"idx", s->index}, {"topk", json::array()}};
            for (const auto& kv : s->topk) {
                line["topk"].push_back(json::array({kv.first, kv.second}));
            }
            (*dump) << line.dump() << "\n";
        }
        const double now = now_s();
        rec.add_frame({{"read", s->read}, {"preprocess", s->preprocess},
                       {"inference", s->inference}, {"postprocess", s->postprocess},
                       {"q_read", s->q_read}, {"q_inflight", s->q_inflight},
                       {"q_output", s->q_output},
                       {"q_reorder", now - s->t_done_post}},
                      now - s->t_start);
        // The slot goes back only after its numbers are taken: returning it
        // earlier would let the reader overwrite the frame still being read.
        free_q_.push(static_cast<int>(s - slots_.data()));
    };

    while (true) {
        auto got = result_q_.pop();
        if (!got) break;
        if (args_.reorder) {
            for (auto& kv : buf.push((*got)->index, *got)) emit(kv.second);
        } else {
            emit(*got);
        }
    }
    if (args_.reorder) {
        for (auto& kv : buf.drain()) emit(kv.second);
    }

    free_q_.close();
    t_read.join();
    t_pre.join();
    for (auto& t : t_posts) t.join();

    order_out = checker.stats();
    reorder_out = args_.reorder ? buf.stats() : json{{"enabled", false}};
    return {{"submitted", submitted_.load()},
            {"errors", error_.empty() ? json::array() : json::array({error_})}};
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    try {
        OvBackend backend(a.model, a.device);
        std::printf("model input %dx%d  device=%s  inflight=%d queue=%d "
                    "post-workers=%d reorder=%s\n",
                    backend.height(), backend.width(), a.device.c_str(),
                    a.inflight, a.queue_size, a.post_workers,
                    a.reorder ? "on" : "off");

        json runs = json::array(), sources = json::array();
        const int repeats = std::max(1, a.repeat);
        for (int i = 0; i < repeats; ++i) {
            auto source = make_source(a.source, a.frames);
            Recorder rec(kService, kWaits, a.warmup, a.warmup_s, /*pipelined=*/true);
            AsyncPipeline pipe(backend, a);
            json order, reorder;
            std::unique_ptr<std::ofstream> dump;
            if (!a.dump_topk.empty()) {
                dump = std::make_unique<std::ofstream>(a.dump_topk);
                if (!*dump) {
                    std::fprintf(stderr, "cannot write %s\n", a.dump_topk.c_str());
                    return 1;
                }
            }
            const json info = pipe.run(*source, rec, order, reorder, dump.get());

            char title[256];
            std::snprintf(title, sizeof(title),
                          "%s | openvino/%s | ASYNC inflight=%d | run %d/%d",
                          stem(a.model).c_str(), a.device.c_str(), a.inflight,
                          i + 1, repeats);
            rec.print_summary(title);
            std::printf(" order         emitted %llu  violations %llu  %s\n",
                        static_cast<unsigned long long>(order["emitted"].get<uint64_t>()),
                        static_cast<unsigned long long>(order["order_violations"].get<uint64_t>()),
                        order["in_submission_order"].get<bool>() ? "IN SUBMISSION ORDER"
                                                                 : "SHUFFLED");
            if (a.reorder) {
                std::printf(" reorder       max depth %llu  out-of-order arrivals %llu"
                            "  max displacement %llu  forced flushes %llu\n",
                            static_cast<unsigned long long>(reorder["max_depth"].get<size_t>()),
                            static_cast<unsigned long long>(reorder["out_of_order_arrivals"].get<size_t>()),
                            static_cast<unsigned long long>(reorder["max_displacement"].get<uint64_t>()),
                            static_cast<unsigned long long>(reorder["forced_flushes"].get<size_t>()));
            }
            if (!info["errors"].empty()) {
                std::printf(" !! errors: %s\n", info["errors"].dump().c_str());
            }

            json res = rec.result();
            res["order"] = order;
            res["reorder"] = reorder;
            res["submitted"] = info["submitted"];
            res["errors"] = info["errors"];
            runs.push_back(std::move(res));
            sources.push_back(source->describe());

            if (i + 1 < repeats && a.cooldown > 0.0) {
                std::this_thread::sleep_for(std::chrono::duration<double>(a.cooldown));
            }
        }

        json meta = {
            {"tag", a.tag}, {"mode", "async"}, {"impl", "cpp"},
            {"args", {{"model", a.model}, {"backend", "openvino"},
                      {"ov_device", a.device}, {"source", a.source},
                      {"frames", a.frames}, {"duration_s", a.duration_s},
                      {"warmup", a.warmup}, {"warmup_s", a.warmup_s},
                      {"top_k", a.top_k}, {"inflight", a.inflight},
                      {"queue_size", a.queue_size}, {"reorder", a.reorder ? "on" : "off"},
                      {"repeat", a.repeat}, {"cooldown", a.cooldown},
                      {"fused", a.fused}, {"post_workers", a.post_workers},
                      {"stall_every", a.stall_every},
                      {"stall_ms", a.stall_ms},
                      {"tag", a.tag}, {"out_dir", a.out_dir},
                      {"intra_threads", nullptr}}},
            {"backend", backend.describe()},
            {"sources", sources},
            {"model_input_hw", json::array({backend.height(), backend.width()})},
            {"idle_gate", {{"checked", false}, {"reason", "not implemented in cpp"}}},
            {"env", {{"impl", "cpp"}}},
        };

        const std::string src_slug = a.source.empty() ? "synthetic" : stem(a.source);
        const std::string path = a.out_dir + "/" + a.tag + "__cppasync" +
                                 std::to_string(a.inflight) + "__openvino-" +
                                 lower(a.device) + "__" + stem(a.model) + "__" +
                                 src_slug + ".json";
        std::ofstream f(path);
        if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return 1; }
        f << json{{"meta", meta}, {"runs", runs}}.dump(2) << "\n";
        std::printf("\nsaved -> %s\n", path.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
