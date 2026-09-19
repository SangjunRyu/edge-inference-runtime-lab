cmake_minimum_required(VERSION 3.20)
project(edge_inference_runtime_lab CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --- OpenVINO, through its C API -----------------------------------------
#
# The C API, not the C++ one, and deliberately: MinGW and MSVC share no C++
# ABI, so the MSVC-built openvino.lib cannot be linked from GCC, while C has a
# stable ABI and openvino_c.lib can. That is what lets this build with no
# Microsoft toolchain present, and it keeps the same source building under MSVC
# later.
#
# The runtime comes from the project's own venv, so the C++ and Python sides
# are guaranteed to be the same OpenVINO build rather than two that happen to
# have similar version numbers.
set(OV_ROOT "${CMAKE_SOURCE_DIR}/../.venv/Lib/site-packages/openvino"
    CACHE PATH "OpenVINO runtime root (the pip package ships the C SDK)")
if(NOT EXISTS "${OV_ROOT}/include/openvino/c/openvino.h")
  message(FATAL_ERROR
    "OpenVINO C headers not found under OV_ROOT=${OV_ROOT}\n"
    "Point -DOV_ROOT at the directory holding include/ and libs/, e.g. the\n"
    "openvino package inside the venv used for the Python results.")
endif()

add_library(openvino_c SHARED IMPORTED)
set_target_properties(openvino_c PROPERTIES
  IMPORTED_LOCATION "${OV_ROOT}/libs/openvino_c.dll"
  IMPORTED_IMPLIB   "${OV_ROOT}/libs/openvino_c.lib"
  INTERFACE_INCLUDE_DIRECTORIES "${OV_ROOT}/include")

# --- nlohmann/json --------------------------------------------------------
# Vendored as a single header so the build has one less thing to install, and
# so the same tree builds under MSVC on a machine with no pacman.
add_library(eirl_json INTERFACE)
target_include_directories(eirl_json INTERFACE third_party)

# --- The part that needs no OpenCV ---------------------------------------
# Kept separate on purpose: it lets the phase 0 probe exercise the backend,
# the harness and the top-k path before OpenCV is installed, which is what
# makes "fail fast on the toolchain" real rather than aspirational.
add_library(eirl_base STATIC
  src/harness.cpp
  src/ov_backend.cpp
  src/postprocess.cpp)
target_include_directories(eirl_base PUBLIC include)
target_link_libraries(eirl_base PUBLIC openvino_c eirl_json)

if(MSVC)
  target_compile_options(eirl_base PRIVATE /W4)
else()
  target_compile_options(eirl_base PRIVATE -Wall -Wextra)
endif()

add_executable(probe src/probe.cpp)
target_link_libraries(probe PRIVATE eirl_base)

# --- W5-1: ownership, with nothing else linked ---------------------------
#
# Deliberately does not link eirl_base, and therefore not OpenVINO and not
# OpenCV. That link line *is* the fence around the zero-allocation claim: with
# nothing else in the binary, every allocation the counter sees is one this
# repository asked for. It is also why this target builds and runs on a machine
# with no accelerator and no OpenCV.
find_package(Threads REQUIRED)

add_executable(w5_ownership
  src/w5_ownership.cpp
  src/alloc_counter.cpp)
target_include_directories(w5_ownership PRIVATE include)
target_link_libraries(w5_ownership PRIVATE Threads::Threads)
if(MSVC)
  target_compile_options(w5_ownership PRIVATE /W4)
else()
  target_compile_options(w5_ownership PRIVATE -Wall -Wextra)
endif()

# A target that must not compile. See src/negative_direction.cpp.
option(EIRL_NEGATIVE_COMPILE "Add the target that proves leases are not interchangeable" OFF)
if(EIRL_NEGATIVE_COMPILE)
  add_executable(negative_direction src/negative_direction.cpp)
  target_include_directories(negative_direction PRIVATE include)
endif()

# --- The part that needs OpenCV ------------------------------------------
find_package(OpenCV QUIET COMPONENTS core imgproc videoio)
if(OpenCV_FOUND)
  add_library(eirl_core STATIC src/pipeline.cpp src/sources.cpp)
  target_link_libraries(eirl_core PUBLIC eirl_base ${OpenCV_LIBS})
  target_include_directories(eirl_core PUBLIC ${OpenCV_INCLUDE_DIRS})
  if(MSVC)
    target_compile_options(eirl_core PRIVATE /W4)
  else()
    target_compile_options(eirl_core PRIVATE -Wall -Wextra)
  endif()

  add_executable(run_sync_cpp src/run_sync.cpp)
  target_link_libraries(run_sync_cpp PRIVATE eirl_core)

  add_executable(run_async_cpp src/run_async.cpp)
  target_link_libraries(run_async_cpp PRIVATE eirl_core)
  find_package(Threads REQUIRED)
  target_link_libraries(run_async_cpp PRIVATE Threads::Threads)
else()
  message(WARNING
    "OpenCV not found -- building 'probe' only.\n"
    "  MSYS2: pacman -S mingw-w64-ucrt-x86_64-opencv")
endif()

# The OpenVINO DLLs are not on PATH; put them beside the executables so a build
# is runnable without the caller arranging an environment. Copying only the
# import target's DLL is not enough -- openvino_c.dll loads openvino.dll and
# the device plugins next to it -- so the whole libs directory goes.
foreach(tgt probe run_sync_cpp run_async_cpp)
  if(TARGET ${tgt})
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              "${OV_ROOT}/libs" "$<TARGET_FILE_DIR:${tgt}>"
      COMMENT "copying the OpenVINO runtime next to ${tgt}")
  endif()
endforeach()

# --- reproductions --------------------------------------------------------
# Runnable versions of the defects found while building this, each with a
# broken and a fixed mode. See repro/README.md. Built here so they share the
# OpenVINO target and cannot drift away from the code they document.
if(EXISTS "${CMAKE_SOURCE_DIR}/../repro/cpp/CMakeLists.txt")
  add_subdirectory("${CMAKE_SOURCE_DIR}/../repro/cpp" repro)
endif()
