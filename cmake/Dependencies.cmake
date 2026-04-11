include(FetchContent)

function(pulsar_find_torch)
  find_package(Torch QUIET)

  if(Torch_FOUND)
    message(STATUS "Found Torch: ${TORCH_INSTALL_PREFIX}")
  else()
    message(STATUS "Torch not found. Set CMAKE_PREFIX_PATH to your libtorch install.")
  endif()

  set(Torch_FOUND ${Torch_FOUND} PARENT_SCOPE)
  set(TORCH_INSTALL_PREFIX ${TORCH_INSTALL_PREFIX} PARENT_SCOPE)
endfunction()

function(pulsar_find_python_bindings)
  find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
  find_package(pybind11 QUIET)

  if(NOT pybind11_FOUND AND PULSAR_FETCH_PYBIND11)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
  endif()

  set(Python3_FOUND ${Python3_FOUND} PARENT_SCOPE)
  set(Python3_EXECUTABLE ${Python3_EXECUTABLE} PARENT_SCOPE)
  set(pybind11_FOUND ${pybind11_FOUND} PARENT_SCOPE)
endfunction()

find_package(nlohmann_json QUIET)

if(NOT nlohmann_json_FOUND AND PULSAR_FETCH_JSON)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
  message(FATAL_ERROR "nlohmann_json is required. Install it or enable PULSAR_FETCH_JSON.")
endif()
