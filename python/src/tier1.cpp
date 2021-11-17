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

#include "gpu.cpp"  // todo: find a cleaner way to call the class PyGPU

void AddImageAndScalar(Buffer& input, Buffer& output, float scalar, PyGPU& gpu)
{
    // auto raw_gpu = dynamic_cast<GPU&>(gpu);
    AddImageAndScalarKernel kernel(std::make_shared<GPU>(gpu));
    kernel.SetInput(input);
    kernel.SetOutput(output);
    kernel.SetScalar(scalar);
    kernel.Execute();
}



PYBIND11_MODULE(tier1, m) {
    
    m.def("add_image_and_scalar", &AddImageAndScalar, "Add image and scalar");

    m.doc() = R"pbdoc(
        tier1 wrapper
        -----------------------
        add_image_and_scalar()
    )pbdoc";
}
