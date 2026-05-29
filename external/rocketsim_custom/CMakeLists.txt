cmake_minimum_required (VERSION 3.8)

project("RocketSim")

# Include custom simulator directories
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

# Add all headers and code files from custom simulator
file(GLOB_RECURSE FILES_ROCKETSIM "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

# Include custom Bullet source files
file(GLOB_RECURSE FILES_BULLET "${CMAKE_CURRENT_SOURCE_DIR}/libsrc/bullet3-3.24/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/libsrc/bullet3-3.24/*.h")

add_library(RocketSim ${FILES_ROCKETSIM} ${FILES_BULLET})

set_target_properties(RocketSim PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RocketSim PROPERTIES CXX_STANDARD 20)
