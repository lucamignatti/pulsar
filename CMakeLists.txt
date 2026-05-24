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

function(pulsar_link_torch_whole_core target_name)
  target_link_libraries(
    "${target_name}"
    PRIVATE
      pulsar_torch
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,pulsar_core>"
  )
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
option(PULSAR_ENABLE_CUDA_SMOKE "Enable CUDA integration smoke test when Python and Torch are available." ON)
option(
  PULSAR_ENABLE_VRPO_MATH_TESTS
  "Enable standalone VRPO math unit tests (pure computation, no IO). Should always be ON for CI."
  ON
)
option(
  PULSAR_ENABLE_CXX_TORCH_IO_TESTS
  "Enable standalone C++ libtorch serialization/checkpoint unit tests. Disabled by default because CUDA Linux libtorch builds can segfault or hang during standalone test process teardown; the production binary smokes cover these paths."
  OFF
)
option(PULSAR_FETCH_JSON "Fetch nlohmann_json when not provided by the system" ON)
option(PULSAR_FETCH_PYBIND11 "Fetch pybind11 when not provided by the system" OFF)
option(PULSAR_ENABLE_TRACING "Enable Perfetto-compatible trace instrumentation" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CTest)
include(CheckLanguage)
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

add_library(pulsar_core_usage INTERFACE)
target_include_directories(pulsar_core_usage INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
target_link_libraries(pulsar_core_usage INTERFACE nlohmann_json::nlohmann_json)
target_compile_features(pulsar_core_usage INTERFACE cxx_std_20)

if(TARGET RocketSim)
  target_link_libraries(pulsar_core_usage INTERFACE RocketSim)
  target_include_directories(pulsar_core_usage INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/RocketSim/src)
  target_compile_definitions(pulsar_core_usage INTERFACE PULSAR_HAS_ROCKETSIM=1)
endif()

if(PULSAR_ENABLE_TRACING)
  target_compile_definitions(pulsar_core_usage INTERFACE PULSAR_ENABLE_TRACING=1)
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
  cpp/src/tracing/tracing.cpp
)

target_link_libraries(pulsar_core PUBLIC pulsar_core_usage)

if(PULSAR_ENABLE_TORCH)
  pulsar_find_torch()

  if(Torch_FOUND)
    add_library(
      pulsar_torch
      cpp/src/model/vrpo_actor.cpp
      cpp/src/model/mamba2_ops.cpp
      cpp/src/model/layernorm_ops.cpp
      cpp/src/model/normalizer.cpp
      cpp/src/training/rollout_storage.cpp
      cpp/src/training/vrpo_math.cpp
      cpp/src/training/reward_engine.cpp
      cpp/src/training/batched_rocketsim_collector.cpp
      cpp/src/training/vrpo_trainer.cpp
      cpp/src/training/optimizers.cpp
      cpp/src/training/gradient_surgery.cpp
      cpp/src/training/predictor_trainer.cpp
      cpp/src/training/gcrl_trainer.cpp
      cpp/src/training/self_play_manager.cpp
    )

    target_include_directories(pulsar_torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    target_link_libraries(pulsar_torch PRIVATE pulsar_core_usage PUBLIC torch)
    target_compile_definitions(pulsar_torch PUBLIC PULSAR_HAS_TORCH=1)

    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND OR Torch_CUDA_FOUND)
      target_compile_definitions(pulsar_torch PUBLIC PULSAR_HAS_CUDA=1 PULSAR_ENABLE_AMP=1)
    endif()
    if(CUDAToolkit_FOUND)
      check_language(CUDA)
      if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        target_sources(
          pulsar_torch
          PRIVATE
            cpp/src/model/mamba2_ops_cuda.cu
            cpp/src/model/layernorm_ops_cuda.cu
            cpp/src/model/lora_ops_cuda.cu
            cpp/src/training/ppo_math_ops_cuda.cu
        )
        target_compile_definitions(
          pulsar_torch
          PRIVATE
            PULSAR_HAS_MAMBA2_CUDA_KERNELS=1
            PULSAR_HAS_LAYERNORM_CUDA_KERNELS=1
            PULSAR_HAS_LORA_CUDA_KERNELS=1
            PULSAR_HAS_PPO_MATH_CUDA_KERNELS=1
        )
        set_source_files_properties(
          cpp/src/model/mamba2_ops_cuda.cu
          cpp/src/model/layernorm_ops_cuda.cu
          cpp/src/model/lora_ops_cuda.cu
          cpp/src/training/ppo_math_ops_cuda.cu
          PROPERTIES LANGUAGE CUDA
        )
        message(STATUS "Custom CUDA kernels enabled: ${CMAKE_CUDA_COMPILER}")
      else()
        message(STATUS "Custom CUDA kernels disabled: CUDA language compiler unavailable.")
      endif()
    else()
      message(STATUS "Custom CUDA kernels disabled: CUDAToolkit was not found.")
    endif()
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
      enable_language(HIP)
      set(CMAKE_HIP_STANDARD 17)
      set(CMAKE_HIP_STANDARD_REQUIRED ON)
      target_sources(
        pulsar_torch
        PRIVATE
          cpp/src/model/mamba2_ops_hip.hip
          cpp/src/model/layernorm_ops_hip.hip
          cpp/src/model/lora_ops_hip.hip
          cpp/src/training/ppo_math_ops_hip.hip
      )
      target_compile_definitions(
        pulsar_torch
        PRIVATE
          PULSAR_HAS_MAMBA2_HIP_KERNELS=1
          PULSAR_HAS_LAYERNORM_HIP_KERNELS=1
          PULSAR_HAS_LORA_HIP_KERNELS=1
          PULSAR_HAS_PPO_MATH_HIP_KERNELS=1
      )
      set_source_files_properties(
        cpp/src/model/mamba2_ops_hip.hip
        cpp/src/model/layernorm_ops_hip.hip
        cpp/src/model/lora_ops_hip.hip
        cpp/src/training/ppo_math_ops_hip.hip
        PROPERTIES LANGUAGE HIP
      )
      message(STATUS "Custom HIP kernels enabled: ${CMAKE_HIP_COMPILER}")
    else()
      message(STATUS "Custom HIP kernels disabled: HIP language compiler unavailable.")
    endif()

    add_executable(pulsar_vrpo_train cpp/src/apps/train_main.cpp)
    pulsar_link_torch_whole_core(pulsar_vrpo_train)
  else()
    message(WARNING "Torch not found; skipping pulsar_torch and VRPO training targets.")
  endif()
endif()

if(PULSAR_ENABLE_PYTHON)
  pulsar_find_python_bindings()

  if(Torch_FOUND AND pybind11_FOUND AND Python3_FOUND)
    pybind11_add_module(pulsar_native cpp/src/python/bindings.cpp)
    target_include_directories(pulsar_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    target_link_libraries(pulsar_native PRIVATE pulsar_torch pulsar_core torch)
    target_compile_definitions(pulsar_native PRIVATE PULSAR_HAS_TORCH=1)
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
    pulsar_link_torch_whole_core(pulsar_torch_tests)
    add_test(NAME pulsar_torch_tests COMMAND pulsar_torch_tests)

    if(PULSAR_ENABLE_VRPO_MATH_TESTS)
      add_executable(
        pulsar_vrpo_math_tests
        cpp/tests/test_vrpo_math.cpp
      )
      target_include_directories(pulsar_vrpo_math_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
      pulsar_link_torch_whole_core(pulsar_vrpo_math_tests)
      add_test(NAME pulsar_vrpo_math_tests COMMAND pulsar_vrpo_math_tests)
      set_tests_properties(pulsar_vrpo_math_tests PROPERTIES LABELS "reference")
    endif()

    add_executable(
      pulsar_reward_engine_tests
      cpp/tests/test_reward_engine.cpp
    )
    target_include_directories(pulsar_reward_engine_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    pulsar_link_torch_whole_core(pulsar_reward_engine_tests)
    add_test(NAME pulsar_reward_engine_tests COMMAND pulsar_reward_engine_tests)

    add_executable(
      pulsar_config_tests
      cpp/tests/test_config.cpp
    )
    target_include_directories(pulsar_config_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    pulsar_link_torch_whole_core(pulsar_config_tests)
    add_test(NAME pulsar_config_tests COMMAND pulsar_config_tests)

    add_executable(
      pulsar_rollout_storage_tests
      cpp/tests/test_rollout_storage.cpp
    )
    target_include_directories(pulsar_rollout_storage_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    pulsar_link_torch_whole_core(pulsar_rollout_storage_tests)
    add_test(NAME pulsar_rollout_storage_tests COMMAND pulsar_rollout_storage_tests)

    add_executable(
      pulsar_self_play_tests
      cpp/tests/test_self_play.cpp
    )
    target_include_directories(pulsar_self_play_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    pulsar_link_torch_whole_core(pulsar_self_play_tests)
    add_test(NAME pulsar_self_play_tests COMMAND pulsar_self_play_tests)
    set_tests_properties(
      pulsar_self_play_tests
      PROPERTIES
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=;OMP_NUM_THREADS=1;MKL_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
        TIMEOUT 30
    )

  endif()
endif()

if(PULSAR_ENABLE_BENCHMARKS AND TARGET pulsar_torch)
  add_executable(
    pulsar_bench
    cpp/benchmarks/benchmark_main.cpp
  )
  target_include_directories(pulsar_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
  pulsar_link_torch_whole_core(pulsar_bench)

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

if(PULSAR_ENABLE_TESTS AND PULSAR_ENABLE_CUDA_SMOKE AND TARGET pulsar_vrpo_train AND Python3_EXECUTABLE)
  add_test(
    NAME pulsar_cuda_smoke
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cuda_smoke.py
      ${CMAKE_CURRENT_SOURCE_DIR}
      $<TARGET_FILE:pulsar_vrpo_train>
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/2v2_vrpo.json
  )
  set_tests_properties(pulsar_cuda_smoke PROPERTIES LABELS "cuda;smoke" SKIP_RETURN_CODE 77)
ENDIF()

if(PULSAR_ENABLE_TESTS AND TARGET pulsar_vrpo_train AND Python3_EXECUTABLE)
  add_test(
    NAME pulsar_cpu_smoke
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cpu_smoke.py
      ${CMAKE_CURRENT_SOURCE_DIR}
      $<TARGET_FILE:pulsar_vrpo_train>
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/2v2_vrpo.json
  )
  set_tests_properties(pulsar_cpu_smoke PROPERTIES LABELS "cpu;smoke")
endif()
