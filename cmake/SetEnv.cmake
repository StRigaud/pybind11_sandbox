
## Project environement variables

set(CMAKE_CXX_STANDARD 11) # Use C++11
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Require (at least) it
set(CMAKE_CXX_EXTENSIONS OFF) # Don't use e.g. GNU extension (like -std=gnu++11) for portability

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set PROJECT_NAME_UPPERCASE and PROJECT_NAME_LOWERCASE variables
string(TOUPPER ${PROJECT_NAME} PROJECT_NAME_UPPERCASE)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWERCASE)

# Library name (by default is the project name)
if(NOT LIBRARY_NAME)
  set(LIBRARY_NAME ${PROJECT_NAME})
endif()

# Library folder name (by default is the project name in lowercase), Example: #include <foo/foo.h>
if(NOT LIBRARY_DIR)
  set(LIBRARY_DIR ${PROJECT_NAME_LOWERCASE})
endif()

# Set additional folder path
if(NOT THIRDPARTY_DIR)
  set(THIRDPARTY_DIR "${PROJECT_SOURCE_DIR}/thirdparty" CACHE PATH "Third party libraries")
endif()

if(NOT UTILITIES_DIR)
  set(UTILITIES_DIR "${PROJECT_SOURCE_DIR}/utilities" CACHE PATH "Utilities folder")
endif()


## Configuration and Build options

# Set library type optiONs (default: STATIC)
option(BUILD_SHARED_LIBS "Build ${LIBRARY_NAME} as a shared library." OFF)
message(STATUS "BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")

# Set Doc compilation (default: ON) (NOT IMPLEMENTED)
option(BUILD_DOCUMENTATION "Build ${LIBRARY_NAME} Documentation." ON)
message(STATUS "BUILD_DOCUMENTATION: ${BUILD_DOCUMENTATION} (NOT IMPLEMENTED)")

# Set OpenCL Standard version number (default: 120 - v1.2)
set(OPENCL_VERSION 120)
message(STATUS "OPENCL_VERSION: ${OPENCL_VERSION}")
mark_as_advanced(OPENCL_VERSION)

# Set Preamble file 
set(CLIC_PREAMBLE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${LIBRARY_DIR}/preamble.cl" CACHE FILEPATH "CLIJ preamble file")
mark_as_advanced(CLIC_PREAMBLE_FILE)

# Set clij folder file 
set(CLIC_KERNELS_DIR "${THIRDPARTY_DIR}/clij-opencl-kernels/kernels" CACHE PATH "CLIJ kernels directory")
mark_as_advanced(CLIC_KERNELS_DIR)

# Manage build type options (default: RELEASE)
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(isMultiConfig)
  set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo" CACHE STRING "" FORCE)
  message(STATUS "CMAKE_CONFIGURATION_TYPES: ${CMAKE_CONFIGURATION_TYPES}")
  message(STATUS "CMAKE_GENERATOR: Multi-config")
else()
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  endif()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
  message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
  message(STATUS "CMAKE_GENERATOR: Single-config")
endif()
message(STATUS "CMAKE_GENERATOR: ${CMAKE_GENERATOR}")

# Configurations tag to avoid compilation colliding
set(CMAKE_DEBUG_POSTFIX "_d")
set(CMAKE_RELEASE_POSTFIX "")

# Coverage flags and includes
if(BUILD_CODE_COVERAGE)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
  include(CodeCoverage) 
  append_coverage_compiler_flags()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
endif()

# List subdirectory macro
macro(subdirlist result curdir)
  file(GLOB children ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${child})
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

# Introduce variables:
#   - CMAKE_INSTALL_LIBDIR
#   - CMAKE_INSTALL_BINDIR
#   - CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)

# CMake target export name
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")

# Always full RPATH (for shared libraries)
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
if(BUILD_SHARED_LIBS)
  # use, i.e. don't skip the full RPATH for the build tree
  set(CMAKE_SKIP_BUILD_RPATH FALSE)

  # when building, don't use the install RPATH already
  # (but later on when installing)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

  # the RPATH to be used when installing, but only if it's not a system directory
  list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  if("${isSystemDir}" STREQUAL "-1")
      set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  endif()
endif()


# CMake Registry
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeRegistry.cmake)
