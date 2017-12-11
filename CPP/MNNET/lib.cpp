#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "headers.h"

PYBIND11_MODULE(nnet, m) {

    m.doc() = "nnet"; // optional module docstring
    //m.def("distance_matrix", &distance_matrix, "Calculate distance matrix with Jaccard distance");

	// Wrap class
	//pybind11::class_<layer, std::shared_ptr<layer>> clsLayer(m, "layer");

	pybind11::class_<layer>(m, "layer")
        .def(pybind11::init<int,std::string>())
        .def("get_nodes", &layer::get_nodes)
        .def("get_activation", &layer::get_activation);

	pybind11::class_<network>(m, "network")
		.def(pybind11::init<double,double>())
		.def("add_layer", &network::add_layer)
		.def("get_structure", &network::get_structure)
		.def("get_weights", [](network &nn, int const i, std::vector<std::vector<double>> &w){
            nn.get_weights(i,w); return w;
		})
		.def("get_dweights", [](network &nn, int const i, std::vector<std::vector<double>> &w){
            nn.get_dweights(i,w); return w;
		})
		.def("get_output", [](network &nn, int const i, std::vector<double> &v){
            nn.get_output(i,v); return v;
		})
		.def("train", &network::train)
		.def("predict", [](network &nn, std::vector<double> const &x, std::vector<double> &y){
            nn.predict(x,y); return y;
		});
}
