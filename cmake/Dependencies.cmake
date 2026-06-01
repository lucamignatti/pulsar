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
    c10_hip
    c10_cuda
    torch
    torch_cpu
    torch_hip
    torch_hip_library
    torch_cpu_library
    torch_library
  )
    if(TARGET "${_torch_target}")
      get_target_property(_torch_link_libraries "${_torch_target}" INTERFACE_LINK_LIBRARIES)
      if(_torch_link_libraries)
        set(_torch_link_libraries_clean "")
        foreach(_torch_link_item IN LISTS _torch_link_libraries)
          if(_torch_link_item MATCHES "(^|[^A-Za-z0-9_])(hip|hiprtc|roc)::")
            continue()
          endif()
          list(APPEND _torch_link_libraries_clean "${_torch_link_item}")
        endforeach()
        set_target_properties(
          "${_torch_target}"
          PROPERTIES INTERFACE_LINK_LIBRARIES "${_torch_link_libraries_clean}"
        )
      endif()

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

