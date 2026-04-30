include(FetchContent)

function(_pulsar_suppress_external_warnings_begin)
  set(_PULSAR_SAVED_CMAKE_WARN_DEPRECATED "${CMAKE_WARN_DEPRECATED}" PARENT_SCOPE)
  set(_PULSAR_SAVED_CMAKE_SUPPRESS_DEVELOPER_WARNINGS "${CMAKE_SUPPRESS_DEVELOPER_WARNINGS}" PARENT_SCOPE)
  set(CMAKE_WARN_DEPRECATED FALSE PARENT_SCOPE)
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 PARENT_SCOPE)
endfunction()

function(_pulsar_suppress_external_warnings_end)
  set(CMAKE_WARN_DEPRECATED "${_PULSAR_SAVED_CMAKE_WARN_DEPRECATED}" PARENT_SCOPE)
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS "${_PULSAR_SAVED_CMAKE_SUPPRESS_DEVELOPER_WARNINGS}" PARENT_SCOPE)
endfunction()

function(_pulsar_detect_torch_install_prefix out_var)
  set(_torch_prefix "")

  if(DEFINED Torch_DIR AND EXISTS "${Torch_DIR}")
    get_filename_component(_torch_prefix "${Torch_DIR}/../../../" ABSOLUTE)
  else()
    foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
      if(EXISTS "${_prefix}/Torch/TorchConfig.cmake")
        get_filename_component(_torch_prefix "${_prefix}/../.." ABSOLUTE)
        break()
      endif()
      if(EXISTS "${_prefix}/share/cmake/Torch/TorchConfig.cmake")
        get_filename_component(_torch_prefix "${_prefix}" ABSOLUTE)
        break()
      endif()
    endforeach()
  endif()

  set(${out_var} "${_torch_prefix}" PARENT_SCOPE)
endfunction()

function(_pulsar_prepare_torch_optional_libs)
  _pulsar_detect_torch_install_prefix(_torch_prefix)
  if(NOT _torch_prefix)
    return()
  endif()

  if(NOT DEFINED kineto_LIBRARY OR NOT kineto_LIBRARY)
    foreach(_fallback IN ITEMS
      "${_torch_prefix}/lib/libtorch.so"
      "${_torch_prefix}/lib/libtorch_cpu.so"
      "${_torch_prefix}/lib/libc10.so"
      "${_torch_prefix}/lib/libtorch.dylib"
      "${_torch_prefix}/lib/libtorch_cpu.dylib"
      "${_torch_prefix}/lib/libc10.dylib"
    )
      if(EXISTS "${_fallback}")
        set(kineto_LIBRARY "${_fallback}" CACHE FILEPATH "Fallback path for optional kineto library." FORCE)
        break()
      endif()
    endforeach()
  endif()
endfunction()

function(_pulsar_strip_std_flags_from_list out_var)
  set(_cleaned "")
  foreach(_item IN LISTS ARGN)
    if(_item MATCHES "^-std=")
      continue()
    endif()
    list(APPEND _cleaned "${_item}")
  endforeach()
  set(${out_var} "${_cleaned}" PARENT_SCOPE)
endfunction()

function(_pulsar_sanitize_torch_language_standard)
  if(DEFINED TORCH_CXX_FLAGS AND NOT TORCH_CXX_FLAGS STREQUAL "")
    separate_arguments(_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    _pulsar_strip_std_flags_from_list(_torch_cxx_flags_clean ${_torch_cxx_flags})
    string(JOIN " " _torch_cxx_flags_joined ${_torch_cxx_flags_clean})
    set(TORCH_CXX_FLAGS "${_torch_cxx_flags_joined}" CACHE STRING "Torch C++ flags" FORCE)
    set(TORCH_CXX_FLAGS "${_torch_cxx_flags_joined}" PARENT_SCOPE)
  endif()

  foreach(_torch_target IN ITEMS
    headeronly
    c10
    c10_cuda
    torch
    torch_cpu
    torch_cpu_library
    torch_library
  )
    if(TARGET "${_torch_target}")
      get_target_property(_torch_compile_options "${_torch_target}" INTERFACE_COMPILE_OPTIONS)
      if(_torch_compile_options)
        _pulsar_strip_std_flags_from_list(_torch_compile_options_clean ${_torch_compile_options})
        set_target_properties(
          "${_torch_target}"
          PROPERTIES INTERFACE_COMPILE_OPTIONS "${_torch_compile_options_clean}"
        )
      endif()

      get_target_property(_torch_compile_features "${_torch_target}" INTERFACE_COMPILE_FEATURES)
      if(_torch_compile_features)
        set_target_properties(
          "${_torch_target}"
          PROPERTIES INTERFACE_COMPILE_FEATURES ""
        )
      endif()

      get_target_property(_torch_cxx_standard "${_torch_target}" CXX_STANDARD)
      if(_torch_cxx_standard)
        set_target_properties(
          "${_torch_target}"
          PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
        )
      endif()
    endif()
  endforeach()
endfunction()

function(pulsar_find_torch)
  _pulsar_prepare_torch_optional_libs()
  _pulsar_suppress_external_warnings_begin()
  find_package(Torch QUIET)
  _pulsar_suppress_external_warnings_end()

  if(Torch_FOUND)
    _pulsar_sanitize_torch_language_standard()
  endif()

  if(Torch_FOUND)
    message(STATUS "Found Torch: ${TORCH_INSTALL_PREFIX}")
  else()
    message(STATUS "Torch not found. Set CMAKE_PREFIX_PATH to your libtorch install.")
  endif()

  set(Torch_FOUND ${Torch_FOUND} PARENT_SCOPE)
  set(TORCH_INSTALL_PREFIX ${TORCH_INSTALL_PREFIX} PARENT_SCOPE)
endfunction()

function(pulsar_find_python_bindings)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)

  if(Python3_FOUND)
    if((NOT DEFINED pybind11_DIR OR NOT EXISTS "${pybind11_DIR}") AND Python3_EXECUTABLE)
      execute_process(
        COMMAND
          "${Python3_EXECUTABLE}"
          -c
          "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _pulsar_pybind11_dir
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pulsar_pybind11_result
      )
      if(_pulsar_pybind11_result EQUAL 0 AND EXISTS "${_pulsar_pybind11_dir}")
        set(pybind11_DIR "${_pulsar_pybind11_dir}" CACHE PATH "pybind11 CMake package directory." FORCE)
      endif()
    endif()

    if(DEFINED pybind11_DIR AND EXISTS "${pybind11_DIR}")
      _pulsar_suppress_external_warnings_begin()
      find_package(pybind11 QUIET CONFIG PATHS "${pybind11_DIR}" NO_DEFAULT_PATH)
      _pulsar_suppress_external_warnings_end()
    else()
      message(STATUS "pybind11 was not available from the selected Python interpreter; skipping pybind11 discovery.")
      set(pybind11_FOUND FALSE)
    endif()
  else()
    message(STATUS "Python3 Development.Module not found; skipping pybind11 discovery.")
    set(pybind11_FOUND FALSE)
  endif()

  if(NOT pybind11_FOUND AND PULSAR_FETCH_PYBIND11)
    message(STATUS "PULSAR_FETCH_PYBIND11 is enabled, but automatic pybind11 fetching is disabled. Install pybind11 in the selected Python environment or set pybind11_DIR explicitly.")
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
