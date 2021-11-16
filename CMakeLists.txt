# Set minimum version of CMake.
cmake_minimum_required(VERSION 3.20)

# Set project name and version
project(CLIc VERSION 0.4.0)

# Set environment variables
include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)

# find required package
find_package(OpenCL REQUIRED)
message(STATUS "Found OpenCL v${OpenCL_VERSION_STRING}: ${OpenCL_INCLUDE_DIRS}")
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python v${Python_VERSION}: ${Python_EXECUTABLE}")
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")


# build clic
add_subdirectory(${LIBRARY_DIR})

# build wrapper
add_subdirectory("python")



