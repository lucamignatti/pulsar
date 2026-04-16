cmake_minimum_required(VERSION 3.25)

project(
  pulsar
  VERSION 0.1.0
  DESCRIPTION "Modular Rocket League bot training stack in C++ and Python"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

function(pulsar_force_target_cxx20 target_name)
  if(NOT TARGET "${target_name}")
    return()
  endif()

  get_target_property(_pulsar_target_sources "${target_name}" SOURCES)
  if(NOT _pulsar_target_sources)
    return()
  endif()

  foreach(_pulsar_source IN LISTS _pulsar_target_sources)
    get_source_file_property(_pulsar_source_options "${_pulsar_source}" COMPILE_OPTIONS)
    if(_pulsar_source_options STREQUAL "NOTFOUND")
      unset(_pulsar_source_options)
    endif()

    list(APPEND _pulsar_source_options "$<$<COMPILE_LANGUAGE:CXX>:-std=c++20>")
    set_source_files_properties(
      "${_pulsar_source}"
      TARGET_DIRECTORY "${target_name}"
      PROPERTIES COMPILE_OPTIONS "${_pulsar_source_options}"
    )
  endforeach()
endfunction()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release MinSizeRel RelWithDebInfo)

option(PULSAR_ENABLE_TORCH "Enable libtorch-backed model and trainer targets" ON)
option(PULSAR_ENABLE_PYTHON "Enable Python bindings and visualization helpers" ON)
option(PULSAR_ENABLE_TESTS "Build unit and smoke tests" ON)
option(PULSAR_ENABLE_BENCHMARKS "Build benchmark targets" ON)
option(PULSAR_ENABLE_ROCKETSIM "Enable RocketSim-backed simulation" ON)
option(PULSAR_DISABLE_ROCM "Disable ROCm integration and ROCm-only tests even when ROCm is available." OFF)
option(PULSAR_FETCH_JSON "Fetch nlohmann_json when not provided by the system" ON)
option(PULSAR_FETCH_PYBIND11 "Fetch pybind11 when not provided by the system" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CTest)
include(cmake/Dependencies.cmake)

if(NOT Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter QUIET)
endif()

if(PULSAR_ENABLE_ROCKETSIM AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/RocketSim/CMakeLists.txt")
  set(_PULSAR_SAVED_CMAKE_WARN_DEPRECATED "${CMAKE_WARN_DEPRECATED}")
  set(CMAKE_WARN_DEPRECATED FALSE)
  add_subdirectory(external/RocketSim EXCLUDE_FROM_ALL)
  set(CMAKE_WARN_DEPRECATED "${_PULSAR_SAVED_CMAKE_WARN_DEPRECATED}")
  if(TARGET RocketSim)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_compile_options(RocketSim PRIVATE -Wno-return-type)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_compile_options(RocketSim PRIVATE -Wno-return-type)
    endif()
  endif()
endif()

add_library(
  pulsar_core
  cpp/src/config/config.cpp
  cpp/src/rl/interfaces.cpp
  cpp/src/rl/action_table.cpp
  cpp/src/env/mutators.cpp
  cpp/src/env/obs_builder.cpp
  cpp/src/env/done.cpp
  cpp/src/env/rocketsim_engine.cpp
  cpp/src/checkpoint/checkpoint.cpp
  cpp/src/logging/wandb_logger.cpp
)

target_include_directories(
  pulsar_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

target_link_libraries(
  pulsar_core
  PUBLIC
    nlohmann_json::nlohmann_json
)

if(TARGET RocketSim)
  target_link_libraries(pulsar_core PUBLIC RocketSim)
  target_include_directories(pulsar_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/RocketSim/src)
  target_compile_definitions(pulsar_core PUBLIC PULSAR_HAS_ROCKETSIM=1)
endif()

target_compile_features(pulsar_core PUBLIC cxx_std_20)

if(PULSAR_ENABLE_TORCH)
  pulsar_find_torch()

  if(Torch_FOUND)
    add_library(
      pulsar_torch
      cpp/src/model/actor_critic.cpp
      cpp/src/model/normalizer.cpp
      cpp/src/training/offline_dataset.cpp
      cpp/src/training/offline_pretrainer.cpp
      cpp/src/training/online_ngp_dataset_writer.cpp
      cpp/src/training/online_ngp_replay_buffer.cpp
      cpp/src/training/rollout_storage.cpp
      cpp/src/training/batched_rocketsim_collector.cpp
      cpp/src/training/ppo_math.cpp
      cpp/src/training/ppo_trainer.cpp
      cpp/src/training/self_play_manager.cpp
    )

    target_include_directories(pulsar_torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    target_link_libraries(pulsar_torch PUBLIC pulsar_core torch)
    target_compile_definitions(pulsar_torch PUBLIC PULSAR_HAS_TORCH=1)
    pulsar_force_target_cxx20(pulsar_torch)

    add_executable(pulsar_train cpp/src/apps/train_main.cpp)
    target_link_libraries(pulsar_train PRIVATE pulsar_torch)
    pulsar_force_target_cxx20(pulsar_train)

    add_executable(pulsar_offline_train cpp/src/apps/offline_train_main.cpp)
    target_link_libraries(pulsar_offline_train PRIVATE pulsar_torch)
    pulsar_force_target_cxx20(pulsar_offline_train)
  else()
    message(WARNING "Torch not found; skipping pulsar_torch and pulsar_train targets.")
  endif()
endif()

if(PULSAR_ENABLE_PYTHON)
  pulsar_find_python_bindings()

  if(Torch_FOUND AND pybind11_FOUND AND Python3_FOUND)
    pybind11_add_module(pulsar_native cpp/src/python/bindings.cpp)
    target_include_directories(pulsar_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    target_link_libraries(pulsar_native PRIVATE pulsar_torch pulsar_core torch)
    target_compile_definitions(pulsar_native PRIVATE PULSAR_HAS_TORCH=1)
    pulsar_force_target_cxx20(pulsar_native)
    set_target_properties(
      pulsar_native
      PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
    )
  else()
    message(STATUS "Python bindings disabled: Torch, pybind11, or Python3 Development.Module was unavailable.")
  endif()
endif()

if(PULSAR_ENABLE_TESTS)
  add_executable(
    pulsar_core_tests
    cpp/tests/test_main.cpp
  )
  target_include_directories(pulsar_core_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
  target_link_libraries(pulsar_core_tests PRIVATE pulsar_core)
  add_test(NAME pulsar_core_tests COMMAND pulsar_core_tests)

  add_executable(
    pulsar_env_tests
    cpp/tests/test_env.cpp
  )
  target_include_directories(pulsar_env_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
  target_link_libraries(pulsar_env_tests PRIVATE pulsar_core)
  add_test(NAME pulsar_env_tests COMMAND pulsar_env_tests)

  if(TARGET pulsar_torch)
    add_executable(
      pulsar_torch_tests
      cpp/tests/test_torch.cpp
    )
    target_include_directories(pulsar_torch_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    target_link_libraries(pulsar_torch_tests PRIVATE pulsar_torch)
    pulsar_force_target_cxx20(pulsar_torch_tests)
    add_test(NAME pulsar_torch_tests COMMAND pulsar_torch_tests)

    add_executable(
      pulsar_offline_tests
      cpp/tests/test_offline.cpp
    )
    target_include_directories(pulsar_offline_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    target_link_libraries(pulsar_offline_tests PRIVATE pulsar_torch)
    pulsar_force_target_cxx20(pulsar_offline_tests)
    add_test(NAME pulsar_offline_tests COMMAND pulsar_offline_tests)

    add_executable(
      pulsar_dataset_tests
      cpp/tests/test_dataset.cpp
    )
    target_include_directories(pulsar_dataset_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    target_link_libraries(pulsar_dataset_tests PRIVATE pulsar_torch)
    pulsar_force_target_cxx20(pulsar_dataset_tests)
    add_test(NAME pulsar_dataset_tests COMMAND pulsar_dataset_tests)

    add_executable(
      pulsar_ppo_math_tests
      cpp/tests/test_ppo_math.cpp
    )
    target_include_directories(pulsar_ppo_math_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    target_link_libraries(pulsar_ppo_math_tests PRIVATE pulsar_torch)
    pulsar_force_target_cxx20(pulsar_ppo_math_tests)
    add_test(NAME pulsar_ppo_math_tests COMMAND pulsar_ppo_math_tests)
    set_tests_properties(pulsar_ppo_math_tests PROPERTIES LABELS "reference")

    add_executable(
      pulsar_collector_tests
      cpp/tests/test_collector.cpp
    )
    target_include_directories(pulsar_collector_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    target_link_libraries(pulsar_collector_tests PRIVATE pulsar_torch)
    pulsar_force_target_cxx20(pulsar_collector_tests)
    add_test(NAME pulsar_collector_tests COMMAND pulsar_collector_tests)
    set_tests_properties(pulsar_collector_tests PROPERTIES LABELS "reference")

    add_executable(
      pulsar_self_play_tests
      cpp/tests/test_self_play.cpp
    )
    target_include_directories(pulsar_self_play_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    target_link_libraries(pulsar_self_play_tests PRIVATE pulsar_torch)
    pulsar_force_target_cxx20(pulsar_self_play_tests)
    add_test(NAME pulsar_self_play_tests COMMAND pulsar_self_play_tests)

    if(TARGET pulsar_train AND TARGET pulsar_offline_train AND TARGET pulsar_native AND Python3_EXECUTABLE)
      add_test(
        NAME pulsar_e2e_smoke
        COMMAND
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/e2e_smoke.py
          ${CMAKE_CURRENT_SOURCE_DIR}
          $<TARGET_FILE:pulsar_train>
          $<TARGET_FILE:pulsar_offline_train>
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/2v2_ppo.json
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/2v2_offline.json
      )
    endif()

    if(TARGET pulsar_train AND TARGET pulsar_offline_train AND Python3_EXECUTABLE)
      add_test(
        NAME pulsar_train_binary_smoke
        COMMAND
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/train_binary_smoke.py
          ${CMAKE_CURRENT_SOURCE_DIR}
          $<TARGET_FILE:pulsar_train>
          $<TARGET_FILE:pulsar_offline_train>
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/2v2_ppo.json
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/2v2_offline.json
      )
      set_tests_properties(pulsar_train_binary_smoke PROPERTIES LABELS "smoke")
    endif()

    if(TARGET pulsar_offline_train AND Python3_EXECUTABLE)
      add_test(
        NAME pulsar_offline_binary_smoke
        COMMAND
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/offline_binary_smoke.py
          ${CMAKE_CURRENT_SOURCE_DIR}
          $<TARGET_FILE:pulsar_offline_train>
      )
      set_tests_properties(pulsar_offline_binary_smoke PROPERTIES LABELS "smoke")
    endif()
  endif()
endif()

if(PULSAR_ENABLE_BENCHMARKS AND TARGET pulsar_torch)
  add_executable(
    pulsar_bench
    cpp/benchmarks/benchmark_main.cpp
  )
  target_include_directories(pulsar_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
  target_link_libraries(pulsar_bench PRIVATE pulsar_torch)
  pulsar_force_target_cxx20(pulsar_bench)

  if(PULSAR_ENABLE_TESTS AND Python3_EXECUTABLE)
    add_test(
      NAME pulsar_bench_smoke
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/bench_smoke.py
        $<TARGET_FILE:pulsar_bench>
    )
    set_tests_properties(pulsar_bench_smoke PROPERTIES LABELS "benchmark")
  endif()
endif()

if(PULSAR_ENABLE_TESTS AND Python3_EXECUTABLE)
  add_test(
    NAME pulsar_preprocess_smoke
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/preprocess_smoke.py
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(pulsar_preprocess_smoke PROPERTIES LABELS "smoke;dataset")
endif()

if(PULSAR_ENABLE_TESTS AND PULSAR_ROCM_FOUND AND NOT PULSAR_DISABLE_ROCM AND TARGET pulsar_train AND TARGET pulsar_offline_train AND Python3_EXECUTABLE)
  add_test(
    NAME pulsar_rocm_smoke
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/rocm_smoke.py
      ${CMAKE_CURRENT_SOURCE_DIR}
      $<TARGET_FILE:pulsar_train>
      $<TARGET_FILE:pulsar_offline_train>
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/2v2_ppo.json
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/2v2_offline.json
  )
  set_tests_properties(pulsar_rocm_smoke PROPERTIES LABELS "rocm;smoke" SKIP_RETURN_CODE 77)
endif()
