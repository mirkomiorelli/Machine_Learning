
#include <iostream>
#include <cmath>

#include "nnet.h"

network::network(){

}

void network::initialize(vsz ni, vsz no){
    _nodes.push_back(ni); // input nodes
    _labels.push_back("i");
    _nodes.push_back(no);	// output nodes
    _labels.push_back("o");
	return;
}

// Add a hidden layer to the network, update labels, update transfer matrices
void network::addLayer(vsz nh){
	// Update structure
    _nodes.insert(_nodes.end()-1,nh);
    std::string s = "h"+std::to_string(_nodes.size()-3);
    _labels.insert(_labels.end()-1,s);
    // Reset transfer matrices
    for (size_t s = 1;  s != _labels.size(); s++){
		// Label of transfer matrix
        std::string tlab = _labels[s-1] + "->" + _labels[s];
        // Add transfer matrix
        _W[tlab].initialize(_nodes[s-1] + 1,_nodes[s]);
        _W[tlab].fill_random_uniform();
        //_W[tlab].fill(1.0d);
        _dW[tlab].initialize(_nodes[s-1] + 1,_nodes[s]);
        _dW[tlab].fill(0.0d);
        tlab = _labels[s-1] + "<-" + _labels[s];
        _delta[tlab].initialize(_nodes[s]);
    }
    // Initialize input and output vectors (no bias)
    for (std::size_t s = 0; s != _labels.size(); s++){
        _inputs[_labels[s]].initialize(_nodes[s]);
		_outputs[_labels[s]].initialize(_nodes[s]);
    }

	return;
}

void network::train(std::vector<double> const &x, std::vector<double> const &y){

	// Forward propagation
	_forwardProp(x);
	// Back propagation
	_backProp(y);
	// Update weights


	return;
}

void network::_backProp(std::vector<double> const &y){

	std::size_t ydim = y.size();

	// Setup deltas
    for (std::size_t s = _labels.size()-1; s != 0; s--){
		std::string tlab = _labels[s-1] + "<-" + _labels[s];
		//std::cout << tlab <<std::endl;
		if (s == _labels.size()-1){
            for (std::size_t i = 0; i != ydim; i++){
				double ynnet = _outputs["o"].get(i);
				_delta[tlab].set(i,2.0d * (ynnet - y[i]) * ynnet * (1.0d - ynnet) );
			}
		} else {
			for (std::size_t i = 0; i != _nodes[s]; i++){
                double val = 0.0d;
                for (std::size_t j = 0; j != _delta[_labels[s] + "<-" + _labels[s+1]].size(); j++)
					val += _W[_labels[s] + "->" + _labels[s+1]].get(i,j)
						* _delta[_labels[s] + "<-" + _labels[s+1]].get(j);
				double oi = _outputs[_labels[s]].get(i);
				val = val * oi * (1.0d - oi);
				_delta[_labels[s-1] + "<-" + _labels[s]].set(i,val);
			}
		}
	}

	// Get delta weights by forward propagating with deltas
	for (std::size_t s = 0; s != _labels.size()-1; s++){
		std::string tlab_back = _labels[s] + "<-" + _labels[s+1];
		std::string tlab_forw = _labels[s] + "->" + _labels[s+1];
		if (s == 0){
			for (std::size_t j = 0; j != _delta[tlab_back].size(); j++){
				for (std::size_t i = 0; i != _inputs[_labels[s]].size(); i++){
                    double val = _delta[tlab_back].get(j) * _inputs[_labels[s]].get(i);
                    _dW[tlab_forw].set(i,j,val);

				}
				// Add bias unit
				std::size_t i = _inputs[_labels[s]].size();
				_dW[tlab_forw].set(i,j,_delta[tlab_back].get(j));
			}
		} else {
			for (std::size_t j = 0; j != _delta[tlab_back].size(); j++){
				for (std::size_t i = 0; i != _outputs[_labels[s]].size(); i++){
					double val = _delta[tlab_back].get(j) * _outputs[_labels[s]].get(i);
					_dW[tlab_forw].set(i,j,val);
				}
				// Add bias unit
				std::size_t i = _inputs[_labels[s]].size();
				_dW[tlab_forw].set(i,j,_delta[tlab_back].get(j));
			}
        }
	}

	return;
}

void network::_updateW(){

	return;
}


// Numerical gradient routine to check if backpropagation is right
void network::numgradcheck(){

	// Generate synthetic data
	std::vector<double> x;
	x.push_back(0.5);
	x.push_back(0.8);
	std::vector<double> y;
	y.push_back(0.9);
	y.push_back(0.1);

	double eps = 1e-5;

	for (std::size_t s = 0; s != _labels.size()-1; s++){
        std::string tlab = _labels[s] + "->" + _labels[s+1];
        for (std::size_t i = 0; i != _dW[tlab].rows(); i++){
			for (std::size_t j = 0; j != _dW[tlab].cols(); j++){
				// Get from backprop
				_forwardProp(x);
				_backProp(y);
				double dW_bp = _dW[tlab].get(i,j);
				// Get upper numerical
				double temp = _W[tlab].get(i,j);
				_W[tlab].set(i,j,temp+eps);
				_forwardProp(x);
				double errup = 0.0d;
				for (std::size_t k = 0; k != _outputs["o"].size(); k++)
					errup += (_outputs["o"].get(k) - y[k]) * (_outputs["o"].get(k) - y[k]);
				// Get lower numerical
				temp = _W[tlab].get(i,j);
				_W[tlab].set(i,j,temp-2.0d*eps);
				_forwardProp(x);
				double errlo = 0.0d;
				for (std::size_t k = 0; k != _outputs["o"].size(); k++)
					errlo += (_outputs["o"].get(k) - y[k]) * (_outputs["o"].get(k) - y[k]);
				// Calculate derivative
				double dW_num = (errup - errlo) / (2.0d * eps);
				// Calculate mismatch
				double error = (dW_num - dW_bp);
				std::cout << error << " " << tlab << " " << dW_num << " " << dW_bp << " " << i << " " << j << std::endl;
				//double tot = tot + delta
				// Refix the original weight
				_W[tlab].set(i,j,temp + eps);
			}
        }
	}

	return;
}

// Perform dot product of vin supplemented by bias unit and W
void network::_collapse(cvector<double> &vin, cmatrix<double> &W,
	cvector<double> &vout){

    for (std::size_t c = 0; c != vout.size(); c++){
		double temp = 0.0d;
		for (std::size_t r = 0; r != vin.size(); r++){
			temp += vin.get(r) * W.get(r,c);
			//std::cout << vin.get(r) << " ** " << W.get(r,c) << std::endl;
		}
		// Add bias
		temp += W.get(W.rows()-1,c);

        vout.set(c,temp);
    }

	return;
}

void network::_activate(cvector<double> &in, cvector<double> &out){

	std::size_t nin = in.size();
	std::size_t nout = out.size();
	if (nin != nout){
		std::cout << "Activation function, mismatch in vector dimensions!" << std::endl;
		return;
	}

    for (std::size_t i = 0; i != nout; i++){
        out.set(i,1.0d / (1.0d - std::exp(-in.get(i))));
    }

	return;
}

void network::_forwardProp(std::vector<double> const &x){

	// Build input (no bias, the bias is appendend during collapsing)
	_inputs["i"].copy(x);
	_outputs["i"].copy(x);
	for (std::size_t idx = 1; idx != _labels.size(); idx++){
		std::string tlab = _labels[idx-1] + "->" + _labels[idx];
		_collapse(_outputs[_labels[idx-1]],_W[tlab],_inputs[_labels[idx]]);
		//_inputs[_labels[idx]].print();
		_activate(_inputs[_labels[idx]],_outputs[_labels[idx]]);
		//_outputs[_labels[idx]].print();
	}

	return;
}

void network::printLabels(){
    for (vsz i = 0; i != _labels.size(); i++){
		std::cout << _labels[i] << std::endl;
    }
}
