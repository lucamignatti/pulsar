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
  PULSAR_ENABLE_PPO_MATH_TESTS
  "Enable standalone PPO math unit tests (pure computation, no IO). Should always be ON for CI."
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
option(PULSAR_ENABLE_TRAINING "Build the Pulsar training targets (see PLAN.md)" ON)

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

# Legacy pulsar_torch (retained for: normalizer, rollout_storage, ppo_math,
# reward_engine, batched_rocketsim_collector, optimizers, gradient_ops,
# reachability_grid — all still used by the remaining tests).
# Mamba-2, LoRA, RSSM, SAC, world-model, trainer, self_play_manager,
# replay_buffer, subgoal_planner, gcrl_trainer have been deleted (Phase 7).
if(PULSAR_ENABLE_TORCH)
  pulsar_find_torch()

  if(Torch_FOUND)
    add_library(
      pulsar_torch
      cpp/src/model/normalizer.cpp
      cpp/src/training/rollout_storage.cpp
      cpp/src/training/ppo_math.cpp
      cpp/src/training/reward_engine.cpp
      cpp/src/training/batched_rocketsim_collector.cpp
      cpp/src/training/optimizers.cpp
      cpp/src/training/gradient_ops.cpp
      cpp/src/training/reachability_grid.cpp
    )

    target_include_directories(pulsar_torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    target_link_libraries(pulsar_torch PRIVATE pulsar_core_usage PUBLIC torch)
    target_compile_definitions(pulsar_torch PUBLIC PULSAR_HAS_TORCH=1)

    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND OR Torch_CUDA_FOUND)
      target_compile_definitions(pulsar_torch PUBLIC PULSAR_HAS_CUDA=1 PULSAR_ENABLE_AMP=1)
    endif()
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
      enable_language(HIP)
      set(CMAKE_HIP_STANDARD 17)
      set(CMAKE_HIP_STANDARD_REQUIRED ON)
      message(STATUS "HIP enabled: ${CMAKE_HIP_COMPILER}")
    else()
      message(STATUS "HIP not found.")
    endif()
  else()
    message(WARNING "Torch not found; skipping pulsar_torch.")
  endif()
endif()

if(PULSAR_ENABLE_TRAINING)
  # Ground-up Pulsar training system. New code lives under cpp/pulsar/, cleanly separated from the
  # legacy cpp/src/ tree. Phase 0 scaffold: parity loader + placeholders. Later phases
  # add the env/model/collector/update/curriculum sources here.
  # Header interface (parity loader + pulsar_env.hpp — no Torch, no RocketSim needed)
  add_library(pulsar_headers INTERFACE)
  target_include_directories(pulsar_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/pulsar/include)
  target_compile_features(pulsar_headers INTERFACE cxx_std_20)

  # pulsar_env.hpp needs the pulsar core types (types.hpp, interfaces.hpp)
  # These are header-only in pulsar_core_usage which carries no .cpp deps.
  target_link_libraries(pulsar_headers INTERFACE pulsar_core_usage)

  add_executable(pulsar_train cpp/pulsar/pulsar_train.cpp)
  target_link_libraries(pulsar_train PRIVATE pulsar_headers)

  # Phase 0: harness self-test (shape checks only)
  add_executable(pulsar_parity_check cpp/pulsar/parity_check.cpp)
  target_link_libraries(pulsar_parity_check PRIVATE pulsar_headers)
  if(BUILD_TESTING AND PULSAR_ENABLE_TESTS)
    add_test(
      NAME pulsar_parity_check
      COMMAND pulsar_parity_check ${CMAKE_CURRENT_SOURCE_DIR}/parity/ref
    )
  endif()

  # Phase 1: obs/action/done parity vs Python reference
  add_executable(pulsar_phase1_parity cpp/pulsar/test_phase1_parity.cpp)
  target_link_libraries(pulsar_phase1_parity PRIVATE pulsar_headers)
  if(BUILD_TESTING AND PULSAR_ENABLE_TESTS)
    add_test(
      NAME pulsar_phase1_parity
      COMMAND pulsar_phase1_parity ${CMAKE_CURRENT_SOURCE_DIR}/parity/ref
    )
  endif()

  # Phase 2+3: Torch-dependent models and collector
  if(PULSAR_ENABLE_TORCH AND Torch_FOUND)
    # Core Pulsar training library: models + collector (collector needs pulsar_core for the engine)
    add_library(pulsar_training STATIC
      cpp/pulsar/pulsar_models.cpp
      cpp/pulsar/pulsar_collector.cpp
      cpp/pulsar/pulsar_update.cpp
    )
    target_link_libraries(pulsar_training PUBLIC pulsar_headers torch)
    target_compile_definitions(pulsar_training PUBLIC PULSAR_HAS_TORCH=1)

    # When RocketSim is available, the collector needs pulsar_core (engine + parallel exec)
    if(TARGET pulsar_core)
      target_link_libraries(pulsar_training PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,pulsar_core>")
      target_compile_definitions(pulsar_training PUBLIC PULSAR_HAS_ROCKETSIM=1)
    endif()

    add_executable(pulsar_phase2_parity cpp/pulsar/test_phase2_parity.cpp)
    target_link_libraries(pulsar_phase2_parity PRIVATE pulsar_training)
    if(BUILD_TESTING AND PULSAR_ENABLE_TESTS)
      add_test(
        NAME pulsar_phase2_parity
        COMMAND pulsar_phase2_parity ${CMAKE_CURRENT_SOURCE_DIR}/parity/ref
      )
    endif()

    add_executable(pulsar_phase5_parity cpp/pulsar/test_phase5_parity.cpp)
    target_link_libraries(pulsar_phase5_parity PRIVATE pulsar_headers)
    if(BUILD_TESTING AND PULSAR_ENABLE_TESTS)
      add_test(
        NAME pulsar_phase5_parity
        COMMAND pulsar_phase5_parity ${CMAKE_CURRENT_SOURCE_DIR}/parity/ref
      )
    endif()

    add_executable(pulsar_phase4_parity cpp/pulsar/test_phase4_parity.cpp)
    target_link_libraries(pulsar_phase4_parity PRIVATE pulsar_training)
    if(BUILD_TESTING AND PULSAR_ENABLE_TESTS)
      add_test(
        NAME pulsar_phase4_parity
        COMMAND pulsar_phase4_parity ${CMAKE_CURRENT_SOURCE_DIR}/parity/ref
      )
    endif()

    add_executable(pulsar_phase3_collector cpp/pulsar/test_phase3_collector.cpp)
    target_link_libraries(pulsar_phase3_collector PRIVATE pulsar_training)
    if(BUILD_TESTING AND PULSAR_ENABLE_TESTS)
      add_test(
        NAME pulsar_phase3_collector
        COMMAND pulsar_phase3_collector ${CMAKE_CURRENT_SOURCE_DIR}/external/RocketSim/collision_meshes
      )
    endif()

    # Wire pulsar_train against the full Pulsar training library
    target_link_libraries(pulsar_train PRIVATE pulsar_training)
    target_compile_definitions(pulsar_train PRIVATE PULSAR_HAS_TORCH=1)
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
    # pulsar_torch_tests and pulsar_ppo_math_tests exercise normalizer, ppo_math,
    # rollout_storage, reward_engine — all still present.
    # test_torch.cpp and test_self_play.cpp reference the deleted actor/trainer/sac;
    # they are removed from the build.
    if(PULSAR_ENABLE_PPO_MATH_TESTS)
      add_executable(
        pulsar_ppo_math_tests
        cpp/tests/test_ppo_math.cpp
      )
      target_include_directories(pulsar_ppo_math_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
      target_link_libraries(pulsar_ppo_math_tests PRIVATE pulsar_torch "$<LINK_LIBRARY:WHOLE_ARCHIVE,pulsar_core>")
      add_test(NAME pulsar_ppo_math_tests COMMAND pulsar_ppo_math_tests)
      set_tests_properties(pulsar_ppo_math_tests PROPERTIES LABELS "reference")
    endif()

    add_executable(
      pulsar_reward_engine_tests
      cpp/tests/test_reward_engine.cpp
    )
    target_include_directories(pulsar_reward_engine_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    target_link_libraries(pulsar_reward_engine_tests PRIVATE pulsar_torch "$<LINK_LIBRARY:WHOLE_ARCHIVE,pulsar_core>")
    add_test(NAME pulsar_reward_engine_tests COMMAND pulsar_reward_engine_tests)

    add_executable(
      pulsar_rollout_storage_tests
      cpp/tests/test_rollout_storage.cpp
    )
    target_include_directories(pulsar_rollout_storage_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
    target_link_libraries(pulsar_rollout_storage_tests PRIVATE pulsar_torch "$<LINK_LIBRARY:WHOLE_ARCHIVE,pulsar_core>")
    add_test(NAME pulsar_rollout_storage_tests COMMAND pulsar_rollout_storage_tests)
  endif()
endif()

if(PULSAR_ENABLE_BENCHMARKS AND TARGET pulsar_torch)
  add_executable(
    pulsar_bench
    cpp/benchmarks/benchmark_main.cpp
  )
  target_include_directories(pulsar_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
  target_link_libraries(pulsar_bench PRIVATE pulsar_torch "$<LINK_LIBRARY:WHOLE_ARCHIVE,pulsar_core>")

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

if(PULSAR_ENABLE_TESTS AND PULSAR_ENABLE_CUDA_SMOKE AND TARGET pulsar_train AND Python3_EXECUTABLE)
  add_test(
    NAME pulsar_cuda_smoke
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cuda_smoke.py
      ${CMAKE_CURRENT_SOURCE_DIR}
      $<TARGET_FILE:pulsar_train>
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/2v2.json
  )
  set_tests_properties(pulsar_cuda_smoke PROPERTIES LABELS "cuda;smoke" SKIP_RETURN_CODE 77)
ENDIF()

if(PULSAR_ENABLE_TESTS AND TARGET pulsar_train AND Python3_EXECUTABLE)
  add_test(
    NAME pulsar_cpu_smoke
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cpu_smoke.py
      ${CMAKE_CURRENT_SOURCE_DIR}
      $<TARGET_FILE:pulsar_train>
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/2v2.json
  )
  set_tests_properties(pulsar_cpu_smoke PROPERTIES LABELS "cpu;smoke")
endif()
