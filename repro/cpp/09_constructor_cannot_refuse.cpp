// A device that does not exist opens successfully, and the guard that was meant
// to catch it could not have.
//
// SYMPTOM   Nothing, at the point of failure. An adapter is constructed for a
//           device path that is not there, the object exists, and the first
//           thing to go wrong is a submission several calls later reporting an
//           invalid handle. The error names the submission, not the open.
//
// WHERE     dx_rt, lib/driver_adapter/windows_driver_adapter.cpp. The
//           constructor calls CreateFile and does not look at the result:
//
//               _fd = CreateFile(fileName, ..., OPEN_EXISTING, ...);
//           #if 0
//               if (_fd == INVALID_HANDLE_VALUE) {
//                   cout << "Error: Can't open " << _file << endl;
//                   return;
//               }
//           #endif
//
// CAUSE     Two things, and the second is the one worth the program.
//
//           The check is compiled out. That is the obvious half.
//
//           The other half is that **turning it back on would not fix it.** The
//           guarded code prints and returns, and returning from a constructor
//           still constructs the object. There is no value to inspect and no
//           way to refuse. A constructor cannot fail, so it is the wrong place
//           to put an operation that can, and no amount of checking inside one
//           changes that. `--mode probe` runs the guard as written and shows
//           the object arriving anyway.
//
// FIX       Move the fallible part out of the constructor. A factory that
//           returns an empty optional can say no, and a caller that gets
//           nothing cannot proceed by accident. The type stops carrying a state
//           that should never have existed.
//
// WHY IT MATTERS BEYOND ITSELF
//           This is the same shape as the other silent failures this project
//           collects: OpenVINO's error 127, ONNX Runtime falling back to the
//           CPU without saying so, a missing MinGW runtime exiting 127. In all
//           of them a dependency that is absent looks exactly like a program
//           that decided not to use it, and the cost is paid later by something
//           that is not at fault.
//
//           The general rule: a constructor that cannot refuse should not be
//           given a job that can be refused. An object whose invariants are not
//           established by the time it exists will be asked to honour them
//           anyway.
//
// NEEDS NO ACCELERATOR, and no DEEPX driver -- the point is what happens when
// the device is NOT there. Windows-specific because the adapter is; the same
// structure with fopen behaves identically and is noted below.

#include <cstdio>
#include <optional>
#include <string>
#include <utility>

#include "repro.hpp"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

/// A device path that is not present on any machine. If this ever exists, the
/// program says so rather than quietly testing something else.
const char* const kAbsentDevice = "\\\\.\\eirl_device_that_does_not_exist";

#if defined(_WIN32)
using Handle = HANDLE;
const Handle kInvalid = INVALID_HANDLE_VALUE;

Handle open_device(const char* path) {
    return CreateFileA(path, GENERIC_READ | GENERIC_WRITE,
                       FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                       OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
}

void close_device(Handle h) {
    if (h != kInvalid) CloseHandle(h);
}

/// Stand-in for a submission. On an invalid handle it fails, and the error it
/// reports is about the submission.
bool submit(Handle h, unsigned long* err) {
    DWORD returned = 0;
    char buf[8] = {0};
    const BOOL ok = DeviceIoControl(h, 0x222000u, buf, sizeof buf, buf, sizeof buf,
                                    &returned, nullptr);
    if (!ok) *err = GetLastError();
    return ok != 0;
}
#else
// Same structure, portable: a stream that was never opened.
using Handle = std::FILE*;
Handle const kInvalid = nullptr;
Handle open_device(const char* path) { return std::fopen(path, "rb"); }
void close_device(Handle h) { if (h) std::fclose(h); }
bool submit(Handle h, unsigned long* err) {
    if (!h) { *err = 9; return false; }
    char c;
    return std::fread(&c, 1, 1, h) == 1;
}
#endif

/// The adapter as written: the open happens in the constructor and its result
/// is not examined. `guard` switches on the code that sits inside `#if 0`.
class AdapterAsWritten {
public:
    explicit AdapterAsWritten(const char* path, bool guard) : path_(path) {
        fd_ = open_device(path);
        if (guard) {
            if (fd_ == kInvalid) {
                std::printf("  [guard] Error: Can't open %s\n", path);
                return;   // returns from a constructor; the object still exists
            }
        }
        constructed_ = true;
    }
    ~AdapterAsWritten() { close_device(fd_); }

    Handle fd() const { return fd_; }
    bool reached_end_of_body() const { return constructed_; }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    Handle fd_ = kInvalid;
    bool constructed_ = false;
};

/// The correction: the fallible step is outside the constructor, so there is a
/// place for the answer "no" to be returned to.
class Adapter {
public:
    Adapter(const Adapter&) = delete;
    Adapter& operator=(const Adapter&) = delete;
    Adapter(Adapter&& o) noexcept : fd_(o.fd_) { o.fd_ = kInvalid; }
    ~Adapter() { close_device(fd_); }

    static std::optional<Adapter> open(const char* path) {
        Handle h = open_device(path);
        if (h == kInvalid) return std::nullopt;
        return Adapter(h);
    }

    Handle fd() const { return fd_; }

private:
    explicit Adapter(Handle h) : fd_(h) {}
    Handle fd_ = kInvalid;
};

}  // namespace

int main(int argc, char** argv) {
    repro::Opts o = repro::parse(
        argc, argv, "09_constructor_cannot_refuse -- an absent device opens fine");

    std::printf("device path            %s\n", kAbsentDevice);

    bool reproduced = false;
    char detail[320];

    if (o.mode == "fixed") {
        auto maybe = Adapter::open(kAbsentDevice);
        std::printf("factory returned       %s\n", maybe ? "an adapter" : "nothing");
        unsigned long err = 0;
        int submits = 0;
        if (maybe) {
            submits = 1;
            submit(maybe->fd(), &err);
        }
        std::printf("submissions attempted  %d\n", submits);
        // The defect is "a usable-looking adapter for a device that is not
        // there". With the factory there is no such object and nothing is sent.
        reproduced = static_cast<bool>(maybe);
        std::snprintf(detail, sizeof detail,
                      "the factory refused, so no adapter existed and %d submissions "
                      "were attempted", submits);
    } else {
        const bool guard = (o.mode == "probe");
        AdapterAsWritten a(kAbsentDevice, guard);
        const bool have_object = true;  // it is on the stack; it exists
        const bool valid = a.fd() != kInvalid;
        unsigned long err = 0;
        const bool sent = submit(a.fd(), &err);

        std::printf("guard compiled in      %s\n", guard ? "yes (--mode probe)" : "no");
        std::printf("object constructed     %s\n", have_object ? "yes" : "no");
        std::printf("constructor body ran   %s\n",
                    a.reached_end_of_body() ? "to the end" : "returned early");
        std::printf("handle valid           %s\n", valid ? "yes" : "NO");
        std::printf("submission succeeded   %s  (error %lu)\n", sent ? "yes" : "no", err);

        reproduced = have_object && !valid;
        if (guard) {
            std::snprintf(detail, sizeof detail,
                          "the guard printed and returned, and the object exists anyway; "
                          "the first real symptom is submission error %lu", err);
        } else {
            std::snprintf(detail, sizeof detail,
                          "an adapter was constructed over a device that is not there; "
                          "the first symptom is submission error %lu, several calls later",
                          err);
        }
    }

    if (o.mode == "probe") {
        // probe reports what it saw and does not claim a mode's prediction.
        std::printf("\nprobe    %s\n", detail);
        std::printf("RESULT   the guard does not prevent the object from existing.\n"
                    "         A constructor has no way to refuse, so the check inside\n"
                    "         one cannot be the fix.\n");
        return reproduced ? 0 : 1;
    }
    return repro::verdict(o, reproduced, detail);
}
