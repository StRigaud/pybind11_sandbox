# Set minimum version of CMake.
cmake_minimum_required(VERSION 3.20)

project(pyclesperanto VERSION 0.4.0)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)

Python_add_library(data MODULE src/pydata.cpp)
target_link_libraries(data PRIVATE pybind11::headers OpenCL::OpenCL CLIc::CLIc)
set_target_properties(data PROPERTIES
                      INTERPROCEDURAL_OPTIMIZATION ON
                      CXX_VISIBILITY_PRESET internal
                      VISIBLITY_INLINES_HIDDEN ON)
add_dependencies(data CLIc)

Python_add_library(gpu MODULE src/pygpu.cpp)
target_link_libraries(gpu PRIVATE pybind11::headers OpenCL::OpenCL CLIc::CLIc)
set_target_properties(gpu PROPERTIES
                      INTERPROCEDURAL_OPTIMIZATION ON
                      CXX_VISIBILITY_PRESET internal
                      VISIBLITY_INLINES_HIDDEN ON)
add_dependencies(gpu CLIc)

Python_add_library(tier1 MODULE src/pytier1.cpp)
target_link_libraries(tier1 PRIVATE pybind11::headers OpenCL::OpenCL CLIc::CLIc)
set_target_properties(tier1 PROPERTIES
                      INTERPROCEDURAL_OPTIMIZATION ON
                      CXX_VISIBILITY_PRESET internal
                      VISIBLITY_INLINES_HIDDEN ON)
add_dependencies(tier1 CLIc)