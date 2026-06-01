cmake_minimum_required(VERSION 3.25)

project(
  pulsar
  VERSION 0.1.0
  DESCRIPTION "Pulsar — Rocket League bot trained from sparse rewards"
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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CTest)
include(cmake/Dependencies.cmake)
pulsar_find_torch()

if(NOT Torch_FOUND)
  message(FATAL_ERROR "LibTorch not found. Set CMAKE_PREFIX_PATH to the Torch cmake dir.")
endif()

# ── RocketSim ────────────────────────────────────────────────────────────────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/RocketSim/CMakeLists.txt")
  set(_PULSAR_SAVED_CMAKE_WARN_DEPRECATED "${CMAKE_WARN_DEPRECATED}")
  set(CMAKE_WARN_DEPRECATED FALSE)
  add_subdirectory(external/RocketSim EXCLUDE_FROM_ALL)
  set(CMAKE_WARN_DEPRECATED "${_PULSAR_SAVED_CMAKE_WARN_DEPRECATED}")
  if(TARGET RocketSim)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(RocketSim PRIVATE -Wno-return-type)
    endif()
  endif()
endif()

# ── pulsar_core — engine + interfaces (no Torch) ──────────────────────────────
add_library(pulsar_core_usage INTERFACE)
target_include_directories(pulsar_core_usage INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
target_compile_features(pulsar_core_usage INTERFACE cxx_std_20)

if(TARGET RocketSim)
  target_link_libraries(pulsar_core_usage INTERFACE RocketSim)
  target_include_directories(pulsar_core_usage INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/RocketSim/src)
  target_compile_definitions(pulsar_core_usage INTERFACE PULSAR_HAS_ROCKETSIM=1)
endif()

add_library(pulsar_core
  cpp/src/rl/interfaces.cpp
  cpp/src/env/rocketsim_engine.cpp
)
target_link_libraries(pulsar_core PUBLIC pulsar_core_usage)

# ── pulsar_headers — header interface for the training system ─────────────────
add_library(pulsar_headers INTERFACE)
target_include_directories(pulsar_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/pulsar/include)
target_compile_features(pulsar_headers INTERFACE cxx_std_20)
target_link_libraries(pulsar_headers INTERFACE pulsar_core_usage)

# ── pulsar_training — models + collector + update loop ────────────────────────
add_library(pulsar_training STATIC
  cpp/pulsar/pulsar_models.cpp
  cpp/pulsar/pulsar_collector.cpp
  cpp/pulsar/pulsar_update.cpp
)
target_link_libraries(pulsar_training PUBLIC pulsar_headers torch)
target_compile_definitions(pulsar_training PUBLIC PULSAR_HAS_TORCH=1)

if(TARGET pulsar_core)
  target_link_libraries(pulsar_training PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,pulsar_core>")
  target_compile_definitions(pulsar_training PUBLIC PULSAR_HAS_ROCKETSIM=1)
endif()

# ── Training entry point ──────────────────────────────────────────────────────
add_executable(pulsar_train cpp/pulsar/pulsar_train.cpp)
target_link_libraries(pulsar_train PRIVATE pulsar_training)
target_compile_definitions(pulsar_train PRIVATE PULSAR_HAS_TORCH=1)

# ── Python bindings (pybind11) ────────────────────────────────────────────────
option(PULSAR_ENABLE_PYTHON "Build pybind11 Python bindings (pulsar_py)" OFF)
if(PULSAR_ENABLE_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND)
    pybind11_add_module(pulsar_py MODULE cpp/pulsar/pulsar_bind.cpp)
    target_link_libraries(pulsar_py PRIVATE pulsar_headers)
    message(STATUS "pulsar_py: pybind11 ${pybind11_VERSION}")
  else()
    message(WARNING "PULSAR_ENABLE_PYTHON=ON but pybind11 not found — skipping")
  endif()
endif()

# ── Parity gates ──────────────────────────────────────────────────────────────
add_executable(pulsar_parity_check cpp/pulsar/parity_check.cpp)
target_link_libraries(pulsar_parity_check PRIVATE pulsar_headers)

add_executable(pulsar_phase1_parity cpp/pulsar/test_phase1_parity.cpp)
target_link_libraries(pulsar_phase1_parity PRIVATE pulsar_headers)

add_executable(pulsar_phase2_parity cpp/pulsar/test_phase2_parity.cpp)
target_link_libraries(pulsar_phase2_parity PRIVATE pulsar_training)

add_executable(pulsar_phase3_collector cpp/pulsar/test_phase3_collector.cpp)
target_link_libraries(pulsar_phase3_collector PRIVATE pulsar_training)

add_executable(pulsar_phase4_parity cpp/pulsar/test_phase4_parity.cpp)
target_link_libraries(pulsar_phase4_parity PRIVATE pulsar_training)

add_executable(pulsar_phase5_parity cpp/pulsar/test_phase5_parity.cpp)
target_link_libraries(pulsar_phase5_parity PRIVATE pulsar_headers)

if(BUILD_TESTING)
  add_test(NAME pulsar_parity_check   COMMAND pulsar_parity_check   ${CMAKE_CURRENT_SOURCE_DIR}/parity/ref)
  add_test(NAME pulsar_phase1_parity  COMMAND pulsar_phase1_parity  ${CMAKE_CURRENT_SOURCE_DIR}/parity/ref)
  add_test(NAME pulsar_phase2_parity  COMMAND pulsar_phase2_parity  ${CMAKE_CURRENT_SOURCE_DIR}/parity/ref)
  add_test(NAME pulsar_phase3_collector COMMAND pulsar_phase3_collector ${CMAKE_CURRENT_SOURCE_DIR}/external/RocketSim/collision_meshes)
  add_test(NAME pulsar_phase4_parity  COMMAND pulsar_phase4_parity  ${CMAKE_CURRENT_SOURCE_DIR}/parity/ref)
  add_test(NAME pulsar_phase5_parity  COMMAND pulsar_phase5_parity  ${CMAKE_CURRENT_SOURCE_DIR}/parity/ref)
endif()
