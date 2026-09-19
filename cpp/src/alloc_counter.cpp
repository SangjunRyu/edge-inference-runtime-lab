// Global operator new/delete, counted.
//
// Replacing these is the only way to see an allocation a library made on your
// behalf, which is the point: a pool that claims to allocate nothing must be
// checked by something it does not control.
//
// Every overload the standard lets a compiler pick has to be replaced, not just
// the two obvious ones. Replacing `operator new(size_t)` while leaving
// `operator new(size_t, align_val_t)` alone would leave over-aligned types
// allocating through a path the counter cannot see, and the resulting zero
// would be an artefact of the instrumentation rather than a fact about the
// program.

#include "alloc_counter.hpp"

#include <atomic>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace eirl::alloc {
namespace {

std::atomic<std::uint64_t> g_allocs{0};
std::atomic<std::uint64_t> g_frees{0};
std::atomic<std::uint64_t> g_bytes{0};
std::atomic<bool> g_armed{false};

inline void note_alloc(std::size_t n) noexcept {
    if (g_armed.load(std::memory_order_relaxed)) {
        g_allocs.fetch_add(1, std::memory_order_relaxed);
        g_bytes.fetch_add(n, std::memory_order_relaxed);
    }
}

inline void note_free() noexcept {
    if (g_armed.load(std::memory_order_relaxed)) {
        g_frees.fetch_add(1, std::memory_order_relaxed);
    }
}

void* raw(std::size_t n) {
    // A zero-byte request must still return a distinct pointer.
    return std::malloc(n ? n : 1);
}

void* raw_aligned(std::size_t n, std::size_t align) {
#if defined(_WIN32)
    return _aligned_malloc(n ? n : 1, align);
#else
    std::size_t rounded = ((n ? n : 1) + align - 1) / align * align;
    return std::aligned_alloc(align, rounded);
#endif
}

void free_aligned(void* p) noexcept {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

}  // namespace

void arm() { g_armed.store(true, std::memory_order_relaxed); }
void disarm() { g_armed.store(false, std::memory_order_relaxed); }

void reset() {
    g_allocs.store(0, std::memory_order_relaxed);
    g_frees.store(0, std::memory_order_relaxed);
    g_bytes.store(0, std::memory_order_relaxed);
}

Counts read() {
    return Counts{
        g_allocs.load(std::memory_order_relaxed),
        g_frees.load(std::memory_order_relaxed),
        g_bytes.load(std::memory_order_relaxed),
    };
}

}  // namespace eirl::alloc

// --- plain ---------------------------------------------------------------

void* operator new(std::size_t n) {
    void* p = eirl::alloc::raw(n);
    if (!p) throw std::bad_alloc();
    eirl::alloc::note_alloc(n);
    return p;
}

void* operator new[](std::size_t n) { return ::operator new(n); }

void operator delete(void* p) noexcept {
    if (p) {
        eirl::alloc::note_free();
        std::free(p);
    }
}

void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }

// --- nothrow -------------------------------------------------------------

void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    void* p = eirl::alloc::raw(n);
    if (p) eirl::alloc::note_alloc(n);
    return p;
}

void* operator new[](std::size_t n, const std::nothrow_t& t) noexcept {
    return ::operator new(n, t);
}

void operator delete(void* p, const std::nothrow_t&) noexcept { ::operator delete(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { ::operator delete(p); }

// --- over-aligned --------------------------------------------------------

void* operator new(std::size_t n, std::align_val_t a) {
    void* p = eirl::alloc::raw_aligned(n, static_cast<std::size_t>(a));
    if (!p) throw std::bad_alloc();
    eirl::alloc::note_alloc(n);
    return p;
}

void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }

void operator delete(void* p, std::align_val_t) noexcept {
    if (p) {
        eirl::alloc::note_free();
        eirl::alloc::free_aligned(p);
    }
}

void operator delete[](void* p, std::align_val_t a) noexcept { ::operator delete(p, a); }

void operator delete(void* p, std::size_t, std::align_val_t a) noexcept {
    ::operator delete(p, a);
}

void operator delete[](void* p, std::size_t, std::align_val_t a) noexcept {
    ::operator delete(p, a);
}

void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    void* p = eirl::alloc::raw_aligned(n, static_cast<std::size_t>(a));
    if (p) eirl::alloc::note_alloc(n);
    return p;
}

void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t& t) noexcept {
    return ::operator new(n, a, t);
}

void operator delete(void* p, std::align_val_t a, const std::nothrow_t&) noexcept {
    ::operator delete(p, a);
}

void operator delete[](void* p, std::align_val_t a, const std::nothrow_t&) noexcept {
    ::operator delete(p, a);
}
