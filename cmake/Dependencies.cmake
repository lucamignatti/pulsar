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

function(_pulsar_append_prefix_if_exists prefix)
  if(prefix AND EXISTS "${prefix}")
    list(APPEND CMAKE_PREFIX_PATH "${prefix}")
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
  endif()
endfunction()

function(_pulsar_rocm_prefixes out_var)
  set(_rocm_prefixes "")

  if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    list(APPEND _rocm_prefixes "$ENV{ROCM_PATH}")
  endif()
  if(DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
    list(APPEND _rocm_prefixes "$ENV{HIP_PATH}")
  endif()

  list(APPEND _rocm_prefixes
    /opt/rocm
    /usr
    /usr/local
  )

  set(${out_var} "${_rocm_prefixes}" PARENT_SCOPE)
endfunction()

function(_pulsar_add_rocm_prefixes)
  _pulsar_rocm_prefixes(_rocm_prefixes)

  foreach(_prefix IN LISTS _rocm_prefixes)
    _pulsar_append_prefix_if_exists("${_prefix}")
    _pulsar_append_prefix_if_exists("${_prefix}/hip")
    _pulsar_append_prefix_if_exists("${_prefix}/lib/cmake")
    _pulsar_append_prefix_if_exists("${_prefix}/lib64/cmake")
    _pulsar_append_prefix_if_exists("${_prefix}/lib/cmake/hip")
    _pulsar_append_prefix_if_exists("${_prefix}/lib64/cmake/hip")
    _pulsar_append_prefix_if_exists("${_prefix}/hip/lib/cmake")
    _pulsar_append_prefix_if_exists("${_prefix}/hip/lib64/cmake")
    _pulsar_append_prefix_if_exists("${_prefix}/lib/cmake/hiprtc")
    _pulsar_append_prefix_if_exists("${_prefix}/lib64/cmake/hiprtc")
    _pulsar_append_prefix_if_exists("${_prefix}/hip/lib/cmake/hiprtc")
    _pulsar_append_prefix_if_exists("${_prefix}/hip/lib64/cmake/hiprtc")
  endforeach()
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

function(_pulsar_alias_imported_target alias)
  if(TARGET "${alias}")
    return()
  endif()

  foreach(candidate IN LISTS ARGN)
    if(TARGET "${candidate}")
      add_library("${alias}" INTERFACE IMPORTED)
      target_link_libraries("${alias}" INTERFACE "${candidate}")
      return()
    endif()
  endforeach()
endfunction()

function(_pulsar_find_rocm_package package_name)
  _pulsar_add_rocm_prefixes()
  _pulsar_suppress_external_warnings_begin()
  find_package(${package_name} QUIET CONFIG)
  _pulsar_suppress_external_warnings_end()
  if(${package_name}_FOUND)
    message(STATUS "Found ${package_name}")
  endif()
  set(${package_name}_FOUND "${${package_name}_FOUND}" PARENT_SCOPE)
endfunction()

function(pulsar_find_rocm_hip)
  _pulsar_add_rocm_prefixes()

  _pulsar_suppress_external_warnings_begin()
  find_package(hip QUIET CONFIG)
  _pulsar_suppress_external_warnings_end()

  if(hip_FOUND)
    message(STATUS "Found HIP: ${hip_DIR}")
    if((NOT DEFINED ENV{ROCM_PATH} OR "$ENV{ROCM_PATH}" STREQUAL "") AND hip_DIR)
      get_filename_component(_hip_cmake_dir "${hip_DIR}" DIRECTORY)
      get_filename_component(_hip_lib_dir "${_hip_cmake_dir}" DIRECTORY)
      get_filename_component(_hip_prefix "${_hip_lib_dir}" DIRECTORY)
      if(EXISTS "${_hip_prefix}")
        set(ENV{ROCM_PATH} "${_hip_prefix}")
      endif()
    endif()
  endif()

  set(hip_FOUND ${hip_FOUND} PARENT_SCOPE)
  set(hip_DIR ${hip_DIR} PARENT_SCOPE)
endfunction()

function(pulsar_find_rocm_hiprtc)
  _pulsar_find_rocm_package(hiprtc)

  if(TARGET hiprtc AND NOT TARGET hiprtc::hiprtc)
    add_library(hiprtc::hiprtc INTERFACE IMPORTED)
    target_link_libraries(hiprtc::hiprtc INTERFACE hiprtc)
  endif()

  if(hiprtc_FOUND OR TARGET hiprtc::hiprtc)
    message(STATUS "Found HIPRTC")
  endif()

  set(hiprtc_FOUND ${hiprtc_FOUND} PARENT_SCOPE)
endfunction()

function(pulsar_find_rocm_libraries)
  _pulsar_find_rocm_package(hipblas)
  _pulsar_find_rocm_package(hipblaslt)
  _pulsar_find_rocm_package(hipfft)
  _pulsar_find_rocm_package(hiprand)
  _pulsar_find_rocm_package(hipsparse)
  _pulsar_find_rocm_package(hipsparselt)
  _pulsar_find_rocm_package(hipsolver)
  _pulsar_find_rocm_package(hiptensor)
  _pulsar_find_rocm_package(rocblas)
  _pulsar_find_rocm_package(rocfft)
  _pulsar_find_rocm_package(rocrand)
  _pulsar_find_rocm_package(rocsolver)
  _pulsar_find_rocm_package(MIOpen)
  _pulsar_find_rocm_package(rccl)

  _pulsar_alias_imported_target("roc::hipblas" "hipblas::hipblas" "hipblas")
  _pulsar_alias_imported_target("roc::hipblaslt" "hipblaslt::hipblaslt" "hipblaslt")
  _pulsar_alias_imported_target("roc::hipfft" "hipfft::hipfft" "hipfft")
  _pulsar_alias_imported_target("hip::hiprand" "hiprand::hiprand" "hiprand")
  _pulsar_alias_imported_target("roc::hiprand" "hiprand::hiprand" "hiprand")
  _pulsar_alias_imported_target("roc::hipsparse" "hipsparse::hipsparse" "hipsparse")
  _pulsar_alias_imported_target("roc::hipsparselt" "hipsparselt::hipsparselt" "hipsparselt")
  _pulsar_alias_imported_target("roc::hipsolver" "hipsolver::hipsolver" "hipsolver")
  _pulsar_alias_imported_target("roc::hiptensor" "hiptensor::hiptensor" "hiptensor")
  _pulsar_alias_imported_target("roc::rocblas" "rocblas::rocblas" "rocblas")
  _pulsar_alias_imported_target("roc::rocfft" "rocfft::rocfft" "rocfft")
  _pulsar_alias_imported_target("roc::rocrand" "rocrand::rocrand" "rocrand")
  _pulsar_alias_imported_target("roc::rocsolver" "rocsolver::rocsolver" "rocsolver")
  _pulsar_alias_imported_target("MIOpen" "MIOpen::MIOpen")
  _pulsar_alias_imported_target("rccl::rccl" "rccl")
endfunction()

function(pulsar_find_torch)
  pulsar_find_rocm_hip()
  pulsar_find_rocm_hiprtc()
  pulsar_find_rocm_libraries()
  _pulsar_prepare_torch_optional_libs()
  _pulsar_suppress_external_warnings_begin()
  find_package(Torch QUIET)
  _pulsar_suppress_external_warnings_end()

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
