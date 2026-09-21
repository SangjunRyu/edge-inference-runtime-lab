# The reproductions.
#
# Added from cpp/CMakeLists.txt so they share the OpenVINO import target and
# the include paths, and so `cmake --build build` builds them alongside the
# runners. They deliberately do NOT depend on OpenCV: a defect that can only be
# reproduced once a video decoder is installed is a defect nobody will
# reproduce.
#
# 04 and 05 do not link OpenVINO at all. Those are the two most likely to come
# back, and they run on any machine.

add_library(eirl_repro INTERFACE)
target_include_directories(eirl_repro INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include)

find_package(Threads REQUIRED)

# --- need OpenVINO and a model -------------------------------------------
foreach(name 01_dangling_callback 02_callback_blocks_queue 03_request_busy)
  add_executable(repro_${name} ${name}.cpp)
  target_link_libraries(repro_${name} PRIVATE eirl_base eirl_repro Threads::Threads)
  if(MSVC)
    target_compile_options(repro_${name} PRIVATE /W4)
  else()
    target_compile_options(repro_${name} PRIVATE -Wall -Wextra)
  endif()
endforeach()

# --- need nothing ---------------------------------------------------------
foreach(name 04_sentinel_overtakes 05_reorder_off 06_inflight_overcount
             08_reorder_window_overflow 09_constructor_cannot_refuse)
  add_executable(repro_${name} ${name}.cpp)
  target_link_libraries(repro_${name} PRIVATE eirl_json eirl_repro Threads::Threads)
  if(MSVC)
    target_compile_options(repro_${name} PRIVATE /W4)
  else()
    target_compile_options(repro_${name} PRIVATE -Wall -Wextra)
  endif()
endforeach()

# The OpenVINO-linked ones need the runtime beside them, for the same reason
# the runners do: the DLLs are not on PATH and the Windows loader names only
# the first library it cannot find.
foreach(name 01_dangling_callback 02_callback_blocks_queue 03_request_busy)
  add_custom_command(TARGET repro_${name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${OV_ROOT}/libs" "$<TARGET_FILE_DIR:repro_${name}>"
    COMMENT "copying the OpenVINO runtime next to repro_${name}")
endforeach()

# --- needs the scheduler, but still no accelerator -------------------------
# Links cpp/src/scheduler.cpp rather than restating the pipeline, so the
# reproduction cannot drift away from the code whose bound it is about.
add_executable(repro_07_bound_wrong_metric
  07_bound_wrong_metric.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/src/scheduler.cpp)
target_link_libraries(repro_07_bound_wrong_metric PRIVATE eirl_repro Threads::Threads)
if(MSVC)
  target_compile_options(repro_07_bound_wrong_metric PRIVATE /W4)
else()
  target_compile_options(repro_07_bound_wrong_metric PRIVATE -Wall -Wextra)
endif()
