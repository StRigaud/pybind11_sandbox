#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "cleObject.hpp"
#include "cleBuffer.hpp"
#include "cleImage.hpp"
#include "cleGPU.hpp"
#include "cleAddImageAndScalarKernel.hpp"

using namespace cle;

#include "pygpu.hpp"  // todo: find a cleaner way to call the class PyGPU


template<class T>
void AddImageAndScalar(T& input, T& output, float scalar, PyGPU& device)
{
    AddImageAndScalarKernel kernel(std::make_shared<GPU>(device));
    kernel.SetInput(input);
    kernel.SetOutput(output);
    kernel.SetScalar(scalar);
    kernel.Execute();
}


PYBIND11_MODULE(tier1, m) {
    
    m.def("add_image_and_scalar", &AddImageAndScalar<Buffer>, "Add buffer and scalar",
        pybind11::arg("input"), pybind11::arg("output"), pybind11::arg("scalar"), pybind11::arg("device"));
    m.def("add_image_and_scalar", &AddImageAndScalar<Image>, "Add image and scalar",
        pybind11::arg("input"), pybind11::arg("output"), pybind11::arg("scalar"), pybind11::arg("device"));

    m.doc() = R"pbdoc(
        tier1 wrapper
        -----------------------
        add_image_and_scalar()
    )pbdoc";
}
