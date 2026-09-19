// This file exists to fail to build.
//
// "Input and output buffers are not interchangeable" is a claim about the type
// system, so the way to check it is to ask the compiler. Configure with
// -DEIRL_NEGATIVE_COMPILE=ON and build target `negative_direction`: the build
// must fail, and it must fail on the line below rather than anywhere else.
//
// A run-time check could not make this claim. It could only report a mistake
// after it had already been made.

#include "ownership.hpp"

using namespace eirl;

static void consumes_an_output(OutputLease&) {}

int main() {
    InputPool pool(1, 64);
    InputLease lease = pool.acquire();

    // error: cannot bind 'BufferLease<Direction::Input>' to
    //        'BufferLease<Direction::Output>&'
    consumes_an_output(lease);

    return 0;
}
