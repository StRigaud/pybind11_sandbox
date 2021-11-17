#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "cleGPU.hpp"
#include "cleBuffer.hpp"
#include "cleImage.hpp"

using namespace cle;


// todo: trampoline class for interface C++ python
// Should we do that for all wrapped class? (aka overlayer of wrapper)

class PyGPU : public GPU {
public:
    /* Inherit the constructors */
    using GPU::GPU;
    /* numpy compatibility */
    Buffer CreateBuffer(pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>& dimensions) 
        { 
            pybind11::buffer_info arr = dimensions.request();
            if (arr.ndim > 1)
            {
                throw std::runtime_error("Expecting 1d shape array");
            }
            if (arr.size > 3)
            {
                throw std::runtime_error("Number of dimensions must be three or less");
            }
            float* ptr = static_cast<float*>(arr.ptr);
            std::array<size_t,3> shape = {1, 1, 1};
            for (auto i = 0;  i < arr.size; ++i)
            {
                if(ptr[i] > 0)
                    shape[i] = static_cast<size_t>(ptr[i]);
            }
            return GPU::CreateBuffer<float>(shape); 
        };

    Buffer PushBuffer(pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>& ndarray) 
        { 
            pybind11::buffer_info arr = ndarray.request();
            if (arr.ndim > 3)
            {
                throw std::runtime_error("Number of dimensions must be three or less");
            }
            std::array<size_t,3> shape = {1, 1, 1};
            for (auto i = 0;  i < arr.ndim; ++i)
            {
                if(arr.shape[i] > 0)
                    shape[i] = static_cast<size_t>(arr.shape[i]);
            }
            float* arr_ptr = static_cast<float*>(arr.ptr);
            std::vector<float> values(arr_ptr, arr_ptr + arr.size);
            return GPU::PushBuffer<float>(values, shape); 
        };

    pybind11::array_t<float> PullBuffer(Buffer& buffer) 
        { 
            auto output = GPU::PullBuffer<float>(buffer);
            auto result = pybind11::array_t<float>(output.size());
            float* ptr = static_cast<float*>(result.request().ptr);
            for (auto i = 0;  i < output.size(); ++i)
            {
                ptr[i] = output[i];
            }
            result.resize({buffer.Shape()[0], buffer.Shape()[1], buffer.Shape()[2]});
            return result.squeeze();
        }    


    Image CreateImage(pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>& dimensions) 
        { 
            pybind11::buffer_info arr = dimensions.request();
            if (arr.ndim > 1)
            {
                throw std::runtime_error("Expecting 1d shape array");
            }
            if (arr.size > 3)
            {
                throw std::runtime_error("Number of dimensions must be three or less");
            }
            float* ptr = static_cast<float*>(arr.ptr);
            std::array<size_t,3> shape = {1, 1, 1};
            for (auto i = 0;  i < arr.size; ++i)
            {
                if(ptr[i] > 0)
                    shape[i] = static_cast<size_t>(ptr[i]);
            }
            return GPU::CreateImage<float>(shape); 
        };

    Image PushImage(pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>& ndarray) 
        { 
            pybind11::buffer_info arr = ndarray.request();
            if (arr.ndim > 3)
            {
                throw std::runtime_error("Number of dimensions must be three or less");
            }
            std::array<size_t,3> shape = {1, 1, 1};
            for (auto i = 0;  i < arr.ndim; ++i)
            {
                if(arr.shape[i] > 0)
                    shape[i] = static_cast<size_t>(arr.shape[i]);
            }
            float* arr_ptr = static_cast<float*>(arr.ptr);
            std::vector<float> values(arr_ptr, arr_ptr + arr.size);
            return GPU::PushImage<float>(values, shape); 
        };

    pybind11::array_t<float> PullImage(Image& image) 
        { 
            auto output = GPU::PullImage<float>(image);
            auto result = pybind11::array_t<float>(output.size());
            float* ptr = static_cast<float*>(result.request().ptr);
            for (auto i = 0;  i < output.size(); ++i)
            {
                ptr[i] = output[i];
            }
            result.resize({image.Shape()[0], image.Shape()[1], image.Shape()[2]});
            return result.squeeze();
        }   
};




PYBIND11_MODULE(gpu, m) {  // define a module. module name = file name = cmake target name 

    // class object definition
    pybind11::class_<PyGPU> object(m, "gpu");
        // constructor
        object.def(pybind11::init<>(), "GPU default constructor");
        object.def(pybind11::init<const char*, const char*>(), "GPU constructor", 
            pybind11::arg("t_device_name"), pybind11::arg("t_device_type") = "all");
        // generic methods
        object.def("select_device", &PyGPU::SelectDevice, "select GPU device", 
            pybind11::arg("t_device_name"), pybind11::arg("t_device_type") = "all");
        object.def("info", &PyGPU::Info, "return gpu informations");
        object.def("name", &PyGPU::Name, "return gpu name");
        object.def("score", &PyGPU::Score, "return gpu score");
        object.def("wait_for_kernel_to_finish", &GPU::WaitForKernelToFinish, "Force GPU to wait until kernel finished");
        // buffer methods
        object.def("create_buffer", &PyGPU::CreateBuffer, "create a buffer object",
            pybind11::arg("dimensions"));
        object.def("push_buffer", &PyGPU::PushBuffer, "create and write a buffer object",
            pybind11::arg("ndarray"));
        object.def("pull_buffer", &PyGPU::PullBuffer, "read a buffer object",
            pybind11::arg("buffer"));
        // image methods
        object.def("create_image", &PyGPU::CreateImage, "create an image object",
            pybind11::arg("dimensions"));
        object.def("push_image", &PyGPU::PushImage, "create and write an image object",
            pybind11::arg("ndarray"));
        object.def("pull_image", &PyGPU::PullImage, "read an image object",
            pybind11::arg("image"));  
        // help(gpu) cmd
        object.doc() = R"pbdoc(
            gpu class wrapper
            -----------------------
            shape()
            ndim()
            size()

            create_image()
            push_image()
            pull_image()

            create_buffer()
            push_buffer()
            pull_buffer()

        )pbdoc";
}