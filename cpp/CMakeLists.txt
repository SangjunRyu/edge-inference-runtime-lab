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
option(EIRL_BUILD_OPENVINO "Build the original OpenVINO targets" ON)
option(EIRL_BUILD_ORT "Build ONNX Runtime CPU/CUDA targets" OFF)
if(EIRL_BUILD_OPENVINO)
  if(NOT EXISTS "${OV_ROOT}/include/openvino/c/openvino.h")
    message(FATAL_ERROR "OpenVINO headers missing. Set OV_ROOT, or configure -DEIRL_BUILD_OPENVINO=OFF -DEIRL_BUILD_ORT=ON")
  endif()
  add_library(openvino_c SHARED IMPORTED)
  set_target_properties(openvino_c PROPERTIES
    IMPORTED_LOCATION "${OV_ROOT}/libs/openvino_c.dll"
    IMPORTED_IMPLIB "${OV_ROOT}/libs/openvino_c.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${OV_ROOT}/include")
endif()

# --- nlohmann/json --------------------------------------------------------
# Vendored as a single header so the build has one less thing to install, and
# so the same tree builds under MSVC on a machine with no pacman.
add_library(eirl_json INTERFACE)
target_include_directories(eirl_json INTERFACE third_party)

# --- The part that needs no OpenCV ---------------------------------------
# Kept separate on purpose: it lets the phase 0 probe exercise the backend,
# the harness and the top-k path before OpenCV is installed, which is what
# makes "fail fast on the toolchain" real rather than aspirational.
if(EIRL_BUILD_OPENVINO)
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
endif()

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

# W5-2: the same rules, generalised into stages, workers and bounded queues.
add_executable(w5_scheduler
  src/w5_scheduler.cpp
  src/scheduler.cpp
  src/alloc_counter.cpp)
target_include_directories(w5_scheduler PRIVATE include)
target_link_libraries(w5_scheduler PRIVATE Threads::Threads)

# W5-3: the host-side format conversion, driven by a real .dxnn's metadata.
add_executable(w5_format
  src/w5_format.cpp
  src/nfh.cpp
  src/alloc_counter.cpp)
target_include_directories(w5_format PRIVATE include)
target_link_libraries(w5_format PRIVATE Threads::Threads)

# W5-3 part two: the compiled model as the single source of truth for buffers.
# Needs the vendored nlohmann header, which is header-only and pulls in nothing.
add_executable(w5_contract
  src/w5_contract.cpp
  src/contract.cpp
  src/scheduler.cpp)
target_include_directories(w5_contract PRIVATE include)
target_link_libraries(w5_contract PRIVATE eirl_json Threads::Threads)

# W5-4: admission policies against a source that keeps its own clock.
add_executable(w5_backpressure src/w5_backpressure.cpp)
target_include_directories(w5_backpressure PRIVATE include)
target_link_libraries(w5_backpressure PRIVATE Threads::Threads)

foreach(tgt w5_ownership w5_scheduler w5_format w5_contract w5_backpressure)
  if(MSVC)
    target_compile_options(${tgt} PRIVATE /W4)
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra)
  endif()
endforeach()

# A target that must not compile. See src/negative_direction.cpp.
option(EIRL_NEGATIVE_COMPILE "Add the target that proves leases are not interchangeable" OFF)
if(EIRL_NEGATIVE_COMPILE)
  add_executable(negative_direction src/negative_direction.cpp)
  target_include_directories(negative_direction PRIVATE include)
endif()

# --- The part that needs OpenCV ------------------------------------------
find_package(OpenCV QUIET COMPONENTS core imgproc videoio)
if(OpenCV_FOUND AND EIRL_BUILD_OPENVINO)
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
elseif(NOT OpenCV_FOUND AND EIRL_BUILD_OPENVINO)
  message(WARNING
    "OpenCV not found -- building 'probe' only.\n"
    "  MSYS2: pacman -S mingw-w64-ucrt-x86_64-opencv")
endif()

# ONNX Runtime uses the same source/harness/preprocessing as the OV targets.
if(EIRL_BUILD_ORT)
  if(NOT WIN32)
    message(FATAL_ERROR "ORT SDK/runtime deployment currently targets Windows x64")
  endif()
  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "ORT video runners require the OpenCV C++ SDK; set OpenCV_DIR")
  endif()
  set(ORT_ROOT "${CMAKE_SOURCE_DIR}/build/deps/onnxruntime-win-x64-1.26.0" CACHE PATH "ONNX Runtime C/C++ SDK")
  set(ORT_RUNTIME_DIR "${CMAKE_SOURCE_DIR}/../.venv-cuda/Lib/site-packages/onnxruntime/capi" CACHE PATH "Matching CPU or CUDA ORT DLL directory")
  set(NVIDIA_ROOT "${CMAKE_SOURCE_DIR}/../.venv-cuda/Lib/site-packages/nvidia" CACHE PATH "Optional pip CUDA/cuDNN packages")
  if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h" OR NOT EXISTS "${ORT_RUNTIME_DIR}/onnxruntime.dll")
    message(FATAL_ERROR "Set ORT_ROOT to the SDK and ORT_RUNTIME_DIR to matching runtime DLLs")
  endif()
  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_RUNTIME_DIR}/onnxruntime.dll"
    IMPORTED_IMPLIB "${ORT_ROOT}/lib/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include")
  add_library(eirl_ort STATIC src/ort_backend.cpp src/harness.cpp src/postprocess.cpp src/pipeline.cpp src/sources.cpp)
  target_include_directories(eirl_ort PUBLIC include ${OpenCV_INCLUDE_DIRS})
  target_compile_definitions(eirl_ort PUBLIC EIRL_USE_ORT PRIVATE EIRL_NVIDIA_ROOT="${NVIDIA_ROOT}")
  target_link_libraries(eirl_ort PUBLIC onnxruntime eirl_json Threads::Threads ${OpenCV_LIBS})
  if(MSVC)
    target_compile_options(eirl_ort PRIVATE /W4 /utf-8)
  else()
    target_compile_options(eirl_ort PRIVATE -Wall -Wextra)
  endif()
  add_executable(run_sync_ort src/run_sync.cpp)
  add_executable(run_async_ort src/run_async.cpp)
  target_link_libraries(run_sync_ort PRIVATE eirl_ort)
  target_link_libraries(run_async_ort PRIVATE eirl_ort)
  file(GLOB ORT_DLLS "${ORT_RUNTIME_DIR}/*.dll")
  foreach(tgt run_sync_ort run_async_ort)
    foreach(dll IN LISTS ORT_DLLS)
      add_custom_command(TARGET ${tgt} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:${tgt}>")
    endforeach()
    # Copy imported OpenCV DLLs plus its dynamically loaded FFmpeg plugin.
    foreach(cvtarget IN LISTS OpenCV_LIBS)
      if(TARGET ${cvtarget})
        add_custom_command(TARGET ${tgt} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${cvtarget}>" "$<TARGET_FILE_DIR:${tgt}>")
      endif()
    endforeach()
    get_filename_component(OPENCV_LIB_PARENT "${OpenCV_DIR}" DIRECTORY)
    file(GLOB_RECURSE OPENCV_FFMPEG "${OpenCV_DIR}/bin/opencv_videoio_ffmpeg*.dll" "${OPENCV_LIB_PARENT}/bin/opencv_videoio_ffmpeg*.dll")
    foreach(dll IN LISTS OPENCV_FFMPEG)
      add_custom_command(TARGET ${tgt} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:${tgt}>")
    endforeach()
  endforeach()
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
