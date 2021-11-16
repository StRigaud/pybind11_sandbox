#ifndef clic_h
#define clic_h

#define CLIC_MAJOR_VERSION (0)
#define CLIC_MINOR_VERSION (4)
#define CLIC_PATCH_VERSION (0)
#define CLIC_VERSION "0.4.0"

#ifndef CL_HPP_ENABLE_EXCEPTIONS
#   define CL_HPP_ENABLE_EXCEPTIONS
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
#   define __CL_ENABLE_EXCEPTIONS
#endif

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#   define CL_HPP_TARGET_OPENCL_VERSION 120
#endif

#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#   define CL_HPP_MINIMUM_OPENCL_VERSION 120
#endif

#ifndef CL_TARGET_OPENCL_VERSION
#  define CL_TARGET_OPENCL_VERSION 120
#endif

#if __has_include("OpenCL/opencl.hpp")
#   include <OpenCL/opencl.hpp>
#elif __has_include("CL/opencl.hpp")
#   include <CL/opencl.hpp>
#else
#   include <opencl.hpp>
#endif

#define DEBUG(x) do { std::cerr << x; } while (0)

#endif  // clic_h
