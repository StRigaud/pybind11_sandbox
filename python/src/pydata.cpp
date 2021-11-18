#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "cleObject.hpp"
#include "cleBuffer.hpp"
#include "cleImage.hpp"

using namespace cle;

PYBIND11_MODULE(data, m) {

    pybind11::class_<Object> object(m, "object");
    object.def(pybind11::init<>());
    object.def("ndim", &Object::nDim, "return object dimensionality");
    object.def("size", &Object::Size, "return object size (elements wise)");
    object.def("shape", &Object::Shape, "return object shape (x,y,z)");
    object.def("dtype", &Object::GetDataType, "return object data type (float, double, etc.)");
    object.doc() = R"pbdoc(
        object class wrapper
        -----------------------
        shape()
        ndim()
        size()
    )pbdoc";

    pybind11::class_<Buffer> buffer(m, "buffer", object);
    buffer.def(pybind11::init<>());
    buffer.def("info", &Buffer::Info, "return buffer information");
    buffer.doc() = R"pbdoc(
        buffer class wrapper
        -----------------------
        info()
    )pbdoc";

    pybind11::class_<Image> image(m, "image", object);
    image.def(pybind11::init<>());
    image.def("info", &Image::Info, "return image information");
    image.doc() = R"pbdoc(
        image class wrapper
        -----------------------
        info()
    )pbdoc";
}