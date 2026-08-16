#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mytorch.h"

namespace py = pybind11;

PYBIND11_MODULE(mytorch, m) {

py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
    .def(py::init<std::vector<float>, std::vector<int>, bool>(),
         py::arg("values"), py::arg("shape"), py::arg("requires_grad") = false)
    .def_readwrite("data", &Tensor::data)
    .def_readwrite("grad", &Tensor::grad)
    .def("backward", &Tensor::backward)
    .def_readonly("shape",&Tensor::shape);


py::class_<Linear>(m, "Linear")
    .def(py::init<int,int>())
    .def("forward",&Linear::forward)
    .def_readwrite("W",&Linear::W)
    .def_readwrite("b",&Linear::b);

py::class_<SGD>(m, "SGD")
    .def(py::init<std::vector<std::shared_ptr<Tensor>>, float>())
    .def("step",&SGD::step)
    .def("zero_grad",&SGD::zero_grad);

py::class_<Conv2D>(m,"Conv2D")
    .def(py::init<int,int,int,int,int,int>())      
    .def("forward",&Conv2D::forward)
    .def_readwrite("W",&Conv2D::W)
    .def_readwrite("b",&Conv2D::b);
    
py::class_<MaxPool2D>(m,"MaxPool2D")
    .def(py::init<int>())
    .def("forward",&MaxPool2D::forward);

py::class_<Flatten>(m,"Flatten")
    .def(py::init<>())
    .def("forward",&Flatten::forward);



m.def("add", &add);
m.def("mul", &mul);
m.def("sum", &sum);
m.def("matmul", &matmul);
m.def("relu", &relu);
m.def("cross_entropy",&cross_entropy);
m.def("pad2d",&pad2d);
//m.def("load_image",&load_image);



}
