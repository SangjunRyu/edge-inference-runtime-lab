// W5-3 -- what the host pays to hand a tensor to an NPU.
//
// Three modes, and they answer different kinds of question on purpose:
//
//   --verify   correctness, by inverse. decode(encode(x)) == x needs no NPU and
//              no reference output, because the round trip is its own oracle.
//   --counts   allocations and bytes moved per frame. Counted and derived, so
//              machine-independent -- the rows W5 exists to produce.
//   --bench    naive against blocked transpose, paired, order alternated, with
//              a control. The only mode whose output depends on the machine
//              holding still, and it refuses to report a ratio when it did not.
//
// The geometry is not invented. `tools/dxnn_inspect.py --nfh-args model.dxnn`
// prints the flags for the model that was actually compiled; for densenet121
// that is 224 rows of 672 elements at align_unit 64, which is how 150,528
// logical bytes become the 157,696 the container declares.
//
// Nothing here is copied from DEEPX. See nfh.hpp.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "alloc_counter.hpp"
#include "nfh.hpp"

namespace {

using namespace eirl;

struct Geom {
    std::size_t rows = 224;
    std::size_t row_elems = 672;  // 224 x 3 for densenet121's PRE_IM2COL input
    std::size_t unit = 64;
    std::size_t elem_bytes = 1;
};

std::vector<std::uint8_t> make_input(const Geom& g) {
    std::vector<std::uint8_t> v(g.rows * g.row_elems * g.elem_bytes);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<std::uint8_t>(i * 131u + (i >> 8) * 17u + 1u);
    }
    return v;
}

struct Check {
    std::string what;
    bool ok;
    std::string detail;
};

int verify(const Geom& g) {
    std::vector<Check> checks;

    // The element/byte distinction, restated where it can be checked.
    checks.push_back({"alignment counts elements, not bytes",
                      nfh::aligned_row_elems(1000, 64) * 4 == 4096
                          && nfh::aligned_row_elems(4000, 64) * 1 == 4032,
                      "align(1000 elems)*4 = 4096, align(4000 bytes) = 4032"});

    // The densenet121 input geometry, closed exactly.
    const std::size_t enc = nfh::encoded_bytes(224, 672, 64, 1);
    checks.push_back({"densenet121 input geometry reproduces the container",
                      enc == 157696,
                      "224 x align(672,64) x 1 = " + std::to_string(enc)
                          + " against 157,696 declared, dense 150,528"});

    for (const Geom& shape : std::vector<Geom>{
             g,
             {224, 672, 64, 1},
             {1, 1000, 64, 4},
             {17, 5, 64, 2},     // deliberately awkward
             {64, 64, 64, 1},    // no padding at all
             {3, 1, 8, 4},
         }) {
        const auto src = make_input(shape);
        std::vector<std::uint8_t> encoded(
            nfh::encoded_bytes(shape.rows, shape.row_elems, shape.unit, shape.elem_bytes));
        std::vector<std::uint8_t> back(src.size());

        nfh::encode(src.data(), encoded.data(), shape.rows, shape.row_elems, shape.unit,
                    shape.elem_bytes);
        nfh::decode(encoded.data(), back.data(), shape.rows, shape.row_elems, shape.unit,
                    shape.elem_bytes);

        const std::string tag = std::to_string(shape.rows) + "x" + std::to_string(shape.row_elems)
                                + " unit " + std::to_string(shape.unit) + " elem "
                                + std::to_string(shape.elem_bytes);
        checks.push_back({"decode(encode(x)) == x  [" + tag + "]", back == src,
                          std::to_string(src.size()) + " -> " + std::to_string(encoded.size())
                              + " -> " + std::to_string(back.size()) + " bytes"});

        // Padding is zeroed, so the encoding is a function of the input alone.
        bool zeroed = true;
        const std::size_t padded = nfh::aligned_row_elems(shape.row_elems, shape.unit);
        for (std::size_t r = 0; r < shape.rows && zeroed; ++r) {
            for (std::size_t k = shape.row_elems * shape.elem_bytes;
                 k < padded * shape.elem_bytes; ++k) {
                if (encoded[r * padded * shape.elem_bytes + k] != 0) { zeroed = false; break; }
            }
        }
        checks.push_back({"padding is zeroed  [" + tag + "]", zeroed,
                          "two encodes of one input are byte-identical"});
    }

    // Transposes: the blocked one must be indistinguishable from the naive one.
    for (auto dims : std::vector<std::pair<std::size_t, std::size_t>>{
             {224, 672}, {64, 64}, {97, 13}, {1, 1000}, {1000, 1}}) {
        const std::size_t rows = dims.first, cols = dims.second;
        std::vector<std::uint8_t> src(rows * cols);
        for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::uint8_t>(i * 7u + 3u);
        std::vector<std::uint8_t> a(src.size()), b(src.size()), rt(src.size());

        nfh::transpose_naive(src.data(), a.data(), rows, cols, 1);
        nfh::transpose_blocked(src.data(), b.data(), rows, cols, 1);
        nfh::transpose_naive(a.data(), rt.data(), cols, rows, 1);

        const std::string tag = std::to_string(rows) + "x" + std::to_string(cols);
        checks.push_back({"blocked transpose == naive transpose  [" + tag + "]", a == b,
                          "bitwise identical over " + std::to_string(src.size()) + " bytes"});
        checks.push_back({"transpose(transpose(x)) == x  [" + tag + "]", rt == src, ""});

        // One pass must equal the two passes it replaces.
        std::vector<std::uint8_t> fused(nfh::encoded_bytes(cols, rows, 64, 1));
        std::vector<std::uint8_t> two(fused.size());
        nfh::encode_transposed(src.data(), fused.data(), rows, cols, 64, 1);
        nfh::encode(a.data(), two.data(), cols, rows, 64, 1);
        checks.push_back({"fused encode+transpose == encode(transpose(x))  [" + tag + "]",
                          fused == two,
                          std::to_string(fused.size()) + " bytes"});
    }

    std::size_t failed = 0;
    for (const auto& c : checks) if (!c.ok) ++failed;
    for (const auto& c : checks) {
        if (!c.ok) std::cout << "  FAIL  " << c.what << "\n            " << c.detail << "\n";
    }
    std::cout << "verify         " << (checks.size() - failed) << "/" << checks.size()
              << " checks pass\n";
    return failed == 0 ? 0 : 1;
}

// --- counts ---------------------------------------------------------------

struct PathResult {
    std::string name;
    std::uint64_t allocations;
    std::uint64_t bytes_moved;   // derived: what the algorithm must touch
    std::uint64_t passes;
    std::string note;
};

int counts(const Geom& g, std::uint64_t frames) {
    const std::size_t dense = g.rows * g.row_elems * g.elem_bytes;
    const std::size_t encoded_sz = nfh::encoded_bytes(g.rows, g.row_elems, g.unit, g.elem_bytes);
    const std::size_t transposed_sz =
        nfh::encoded_bytes(g.row_elems, g.rows, g.unit, g.elem_bytes);

    const auto src = make_input(g);
    std::vector<PathResult> out;

    // 1. separate, allocating per frame -- the shape a first version takes.
    {
        eirl::alloc::reset();
        eirl::alloc::arm();
        for (std::uint64_t f = 0; f < frames; ++f) {
            std::vector<std::uint8_t> tmp(encoded_sz);
            std::vector<std::uint8_t> dst(transposed_sz);
            nfh::encode(src.data(), tmp.data(), g.rows, g.row_elems, g.unit, g.elem_bytes);
            nfh::transpose_naive(tmp.data(), dst.data(), g.rows,
                                 nfh::aligned_row_elems(g.row_elems, g.unit), g.elem_bytes);
        }
        eirl::alloc::disarm();
        const auto c = eirl::alloc::read();
        out.push_back({"separate, per-frame buffers", c.allocations / frames,
                       (dense + encoded_sz) + (encoded_sz + transposed_sz), 2,
                       "two passes, two buffers, allocated every frame"});
    }

    // 2. the same work with the buffers hoisted out of the loop.
    {
        std::vector<std::uint8_t> tmp(encoded_sz), dst(transposed_sz);
        eirl::alloc::reset();
        eirl::alloc::arm();
        for (std::uint64_t f = 0; f < frames; ++f) {
            nfh::encode(src.data(), tmp.data(), g.rows, g.row_elems, g.unit, g.elem_bytes);
            nfh::transpose_naive(tmp.data(), dst.data(), g.rows,
                                 nfh::aligned_row_elems(g.row_elems, g.unit), g.elem_bytes);
        }
        eirl::alloc::disarm();
        const auto c = eirl::alloc::read();
        out.push_back({"separate, preallocated", c.allocations / frames,
                       (dense + encoded_sz) + (encoded_sz + transposed_sz), 2,
                       "identical arithmetic, zero allocations"});
    }

    // 3. one pass instead of two.
    {
        std::vector<std::uint8_t> dst(transposed_sz);
        eirl::alloc::reset();
        eirl::alloc::arm();
        for (std::uint64_t f = 0; f < frames; ++f) {
            nfh::encode_transposed(src.data(), dst.data(), g.rows, g.row_elems, g.unit,
                                   g.elem_bytes);
        }
        eirl::alloc::disarm();
        const auto c = eirl::alloc::read();
        out.push_back({"fused encode+transpose", c.allocations / frames,
                       dense + transposed_sz, 1,
                       "the intermediate never exists"});
    }

    std::cout << "geometry       " << g.rows << " rows x " << g.row_elems << " elems, unit "
              << g.unit << ", " << g.elem_bytes << " B/elem\n";
    std::cout << "               dense " << dense << " B, encoded " << encoded_sz
              << " B, transposed+padded " << transposed_sz << " B\n";
    std::cout << "               " << frames << " frames per path\n\n";
    std::cout << "  path                          allocs/frame   passes   bytes touched/frame\n";
    for (const auto& p : out) {
        printf("  %-28s %12llu %8llu %21llu\n", p.name.c_str(),
               static_cast<unsigned long long>(p.allocations),
               static_cast<unsigned long long>(p.passes),
               static_cast<unsigned long long>(p.bytes_moved));
    }
    std::cout << "\n";
    for (const auto& p : out) std::cout << "  " << p.name << ": " << p.note << "\n";

    const double saved = static_cast<double>(out[1].bytes_moved - out[2].bytes_moved);
    std::cout << "\nfusing removes " << static_cast<std::uint64_t>(saved)
              << " B of host traffic per frame, which is " << out[2].bytes_moved
              << " instead of " << out[1].bytes_moved << ".\n"
                 "Counted and derived, not measured: whether it shows up in a stage time is a\n"
                 "separate question, and W4 already found one case where the answer was no.\n";

    const bool ok = out[0].allocations > 0 && out[1].allocations == 0 && out[2].allocations == 0;
    std::cout << "\nverdict        " << (ok ? "as expected" : "UNEXPECTED allocation counts") << "\n";
    return ok ? 0 : 1;
}

// --- bench ----------------------------------------------------------------

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

double cv(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    if (mean == 0.0) return 0.0;
    double s = 0.0;
    for (double x : v) s += (x - mean) * (x - mean);
    return std::sqrt(s / static_cast<double>(v.size() - 1)) / mean;
}

int bench(const Geom& g, int pairs, int reps) {
    const std::size_t rows = g.rows;
    const std::size_t cols = nfh::aligned_row_elems(g.row_elems, g.unit);
    std::vector<std::uint8_t> src(rows * cols * g.elem_bytes);
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::uint8_t>(i * 29u + 5u);
    std::vector<std::uint8_t> dst(src.size());
    std::vector<std::uint8_t> ctrl_dst(src.size());

    auto time_it = [&](auto&& fn) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) fn();
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count() / reps;
    };

    auto naive = [&] { nfh::transpose_naive(src.data(), dst.data(), rows, cols, g.elem_bytes); };
    auto blocked = [&] { nfh::transpose_blocked(src.data(), dst.data(), rows, cols, g.elem_bytes); };
    // The control is a straight copy of the same number of bytes: it is not
    // affected by either change, so its spread across pairs measures the
    // machine rather than the code.
    auto control = [&] { std::memcpy(ctrl_dst.data(), src.data(), src.size()); };

    // Warm the caches and the branch predictors before anything is recorded.
    for (int i = 0; i < 3; ++i) { naive(); blocked(); control(); }

    std::vector<double> ratios, controls;
    for (int p = 0; p < pairs; ++p) {
        double a, b;
        if (p % 2 == 0) { a = time_it(naive); b = time_it(blocked); }
        else            { b = time_it(blocked); a = time_it(naive); }
        controls.push_back(time_it(control));
        if (a > 0.0) ratios.push_back(b / a);
    }

    const double ctrl_cv = cv(controls);
    const double ratio_cv = cv(ratios);
    const double r = median(ratios);

    std::cout << "bench          transpose " << rows << " x " << cols << " x " << g.elem_bytes
              << " B  (" << src.size() << " B), " << pairs << " pairs of " << reps
              << " repetitions, order alternated within pairs\n";
    printf("control        memcpy of the same bytes, CV %.1f%% across pairs\n", ctrl_cv * 100);
    printf("ratio          blocked/naive median %.3f, CV %.1f%%\n", r, ratio_cv * 100);

    const double kControlBar = 0.15;
    if (ctrl_cv > kControlBar) {
        printf("\nverdict        UNMEASURABLE on this machine right now.\n"
               "               The control moved %.1f%% between pairs, so a ratio from this\n"
               "               window says more about the laptop than about the code. Recorded\n"
               "               rather than discarded, which is the treatment W4 settled on.\n",
               ctrl_cv * 100);
        return 0;
    }
    printf("\nverdict        control held (CV %.1f%% under the %.0f%% bar).\n",
           ctrl_cv * 100, kControlBar * 100);
    if (r < 1.0) printf("               blocked is faster by %.2fx.\n", 1.0 / r);
    else printf("               blocked is NOT faster here (%.3fx).\n", r);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Geom g;
    std::string mode = "verify";
    std::uint64_t frames = 200;
    int pairs = 8, reps = 20;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            auto val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
                return argv[++i];
            };
            if (k == "--rows") g.rows = std::stoull(val());
            else if (k == "--row-elems") g.row_elems = std::stoull(val());
            else if (k == "--align") g.unit = std::stoull(val());
            else if (k == "--elem-bytes") g.elem_bytes = std::stoull(val());
            else if (k == "--frames") frames = std::stoull(val());
            else if (k == "--pairs") pairs = std::stoi(val());
            else if (k == "--reps") reps = std::stoi(val());
            else if (k == "--verify") mode = "verify";
            else if (k == "--counts") mode = "counts";
            else if (k == "--bench") mode = "bench";
            else if (k == "-h" || k == "--help") {
                std::cout <<
                    "w5_format -- host-side NPU format conversion (no accelerator)\n\n"
                    "  --verify            round trips and equivalences (default)\n"
                    "  --counts            allocations and bytes moved per frame\n"
                    "  --bench             naive vs blocked transpose, paired, with a control\n\n"
                    "  --rows N --row-elems M --align U --elem-bytes B\n"
                    "                      geometry; tools/dxnn_inspect.py --nfh-args prints it\n"
                    "  --frames N          frames for --counts (default 200)\n"
                    "  --pairs N --reps N  for --bench (default 8 pairs of 20)\n";
                return 0;
            } else throw std::runtime_error("unknown argument " + k);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }

    if (mode == "verify") return verify(g);
    if (mode == "counts") return counts(g, frames);
    return bench(g, pairs, reps);
}
