#include "headers.h"

using namespace std;

network::network(double lrate, double momentum){
	_lrate = lrate;
	_momentum = momentum;
    srand(0);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    _distribution = distribution;
}

network::~network(){}

void network::add_layer(layer const &l){
	_layers.push_back(l);
    _nlayers = _layers.size();
    // Initialize weigth matrices
    if (_nlayers > 1){
		_init_weights(_layers[_nlayers-2].get_nodes(), _layers[_nlayers-1].get_nodes());
    }
    // Initialize state outputs
    vector<double> r;
    r.resize(_layers[_nlayers-1].get_nodes());
    _o.push_back(r);
	return;
}

// Initialize connections between the layer at give layer_id and
// the previous one. Store the initialize random weight into
// the vector _W (each element of this vector is a matrix of weights)
// Dimension: _W[l] -> n(l+1) x n(l)
void network::_init_weights(int const nodes_0, int const nodes_1){

	// Loop over rows of weigth matrix (add 1 for bias unit)
	vector<vector<double>> M;
	vector<vector<double>> M2;
	for (int j = 0; j != nodes_0+1; j++){
		vector<double> r;
		vector<double> c;
		// Loop over columns of weight matrix (add  1 for bias unit)
        for (int i = 0; i != nodes_1; i++){
			// Initialize weights between -0.5 and 0.5
			double random_num = (_distribution(_generator) - 0.5d);
			r.push_back(random_num);
			c.push_back(0.0d);
        }
        // Append row to matrix
        M.push_back(r);
        M2.push_back(c);
	}
	// Append matrix of weights to weights array of the network
	_W.push_back(M);
	_dW.push_back(M2);
	_dWold.push_back(M2);

	return;
}

// Train the network, input is a matrix of the samples (no bias added)
void network::train(vector<vector<double>> const &input,
	vector<vector<double>> const &target, int const epochs){

	_epochs = epochs;

	for (int e = 0; e != _epochs; ++e){

		for (uint i = 0; i != input.size(); ++i){
			// Assign input
			_o[0] = input[i];
			// Forward propagate
			_forward_prop();
			//cout << "OUT_1" << endl; print_vector(_o[1]); cout << endl;
			//cout << "OUT_2" << endl; print_vector(_o[2]); cout << endl;
			// Assign target
			_y = target[i];
			// Back propagation
			_backward_prop();
			// Update weigths
			_update_weights();
		}

	}

	return;
}

// Product between weights and input
void network::_collapse(int const idlayer){

    // Get sizes for given operation
    int nrows = _W[idlayer-1].size();
    int ncols = _W[idlayer-1][0].size();
    int ninp = nrows - 1;
	for (int j = 0; j != ncols; j++){
		_o[idlayer][j] = 0.0d;
		for (int i = 0; i != ninp; i++){
			_o[idlayer][j] += _W[idlayer-1][i][j] * _o[idlayer-1][i];
		}
		// Bias
		_o[idlayer][j] += _W[idlayer-1][ninp][j];
    }

	return;
}

// Activate the product between weights and input
void network::_activate(int const idlayer){

	int nodes = _layers[idlayer].get_nodes();
	string activation = _layers[idlayer].get_activation();
	for (int i = 0; i != nodes; i++){
        if (activation == "sigmoid"){
			_o[idlayer][i] = 1.0d / (1.0d + exp(-_o[idlayer][i]));
		} else if (activation == "relu") {
            _o[idlayer][i] = max(0.0d, _o[idlayer][i]);
		}
	}

	return;
}

// Return derivative of activation for calculation of deltas
double network::_dactivate(int const id, vector<double> const &input,
	vector<vector<double>> const &w){

	double phiprime = 0.0d;

    for (uint i = 0; i != input.size(); ++i){
        phiprime += input[i] * w[i][id];
    }
    // Add bias
    phiprime += w[input.size()][id];

    phiprime = 1.0d / (1.0d + exp(-phiprime));

	return phiprime;

}

// Calculate deltas for back propagation
void network::_delta(uint const idlayer, vector<double> const &delta_old,
	vector<double> &delta_new){

	// number of elements in delta
	int nodes = _layers[idlayer].get_nodes();
	// resize vector delta
	vector<double> ().swap(delta_new);


	// Delta for output layer
	if (idlayer == _layers.size()-1){
		// fill delta elements
        for (int l = 0; l != nodes; l++){
			double phi = _dactivate(l, _o[idlayer-1], _W[idlayer-1]);
			double diff = (_o[idlayer][l] - _y[l]);
			//cout << l << " " << _o[idlayer][l] << " " << _y[l] << endl;
            delta_new.push_back(phi*(1.0-phi)*diff);
			//cout << delta_new[l] << endl;
        }
	// Delta for hidden layers
	} else if (idlayer < _layers.size()-1 && idlayer > 0 ) {
		// elements in delta old
        int nodes_old = delta_old.size();
		for (int l = 0; l != nodes; l++){
			double element = 0.0d;
			for (int i = 0; i != nodes_old; ++i){
				element += delta_old[i] * _W[idlayer][l][i];
			}
			double phi = _dactivate(l, _o[idlayer-1], _W[idlayer-1]);
			delta_new.push_back(phi*(1.0-phi)*element);
		}
	}

	return;
}

void network::_update_weights(){

	// Loop over layers
    for (int l = 0; l != _layers.size()-1; l++){
		// Loop over rows and columns
		int nrows = _W[l].size();
		int ncols = _W[l][0].size();
		for (int i = 0; i != nrows; ++i){
			for (int j = 0; j != ncols; ++j){
                _W[l][i][j] -= _momentum * _dWold[l][i][j] + _lrate * _dW[l][i][j];
			}
		}
    }

	return;
}

// Perform forward propagation (reset the outputs)
void network::_forward_prop(){
    // Propagate through the layers
    for (uint l = 1; l != _nlayers; l++){
		// Matrix product with inputs and weigth matrices
		_collapse(l);
		// Activate with activation function
		_activate(l);
    }

	return;
}

// Perform back propagation (calculate the delta weigths)
void network::_backward_prop(){
	//cout << "Doing back prop" << endl;
	vector<double> delta_old;
	vector<double> delta_new;
	// Start from last layer (output layer) and go back to first
	for (int idlayer = _layers.size()-1; idlayer != 0; idlayer--){
		//cout << idlayer << endl;
		// Calculate delta
		if (idlayer == _layers.size()-1){
			_delta(idlayer,delta_old,delta_old);
			delta_new = delta_old;
		} else {
            _delta(idlayer, delta_old, delta_new);
            delta_old = delta_new;
		}
		// Calculate delta weight
		int windex = idlayer-1;
		int nrows = _dW[windex].size();
		int ncols = _dW[windex][0].size();
		//cout << windex << " " << nrows << " " << ncols << endl;


        for (int l = 0; l != ncols; l++){
            for (int k = 0; k != nrows-1; k++){
                _dWold[windex][k][l] = _dW[windex][k][l];
                _dW[windex][k][l] = delta_new[l] * _o[windex][k];
            }
            _dWold[windex][nrows-1][l] = _dW[windex][nrows-1][l];
            _dW[windex][nrows-1][l] = delta_new[l];
        }
	}
	return;
}

void network::predict(vector<double> const &x, vector<double> &y){

    // Forward propagate input
    _o[0] = x;
    _forward_prop();
    y = _o[_layers.size()-1];

	return;
}


string network::get_structure(){
	string structure = "";
	for (vector<layer>::iterator l = _layers.begin(); l != _layers.end(); ++l){
		structure += "nodes: " + to_string((*l).get_nodes()) + ", activation: "
			+ (*l).get_activation() + "\n";
	}

	return structure;
}

void network::get_weights(int const i, vector<vector<double>> &w){
	w =  _W[i];
	return;
}

void network::get_dweights(int const i, vector<vector<double>> &dw){
	dw = _dW[i];
	return;
}

void network::get_output(int const idlayer, vector<double> &out){
    out = _o[idlayer];
	return;
}

void network::numerical_gradient_check(vector<double> const &X,
	vector<double> const &y){

	// First forward propagate normally and calculate loss and delta weights
	_forward_prop();
	_y = y; _backward_prop(); // don't update weights!
	double loss = 0.0d;
	for (int i = 0; i != y.size(); ++i)
		loss += 0.5d * (y[i] - _o[_layers.size()-1][i]) * (y[i] - _o[_layers.size()-1][i]);
	cout << loss << endl;

	// For each weight calculate the gradient numerically
	cout << "Checking gradient numerically..." << endl;
	double epsilon = 0.001d;
	for (int l = 0; l != _layers.size()-1; ++l){
		cout << "Layer " << l << endl;
		// Adjust weights +epsilon, forward propagate and calculate loss
		int nrows = _W[l].size();
		int ncols = _W[l][0].size();
		for (int i = 0; i != nrows; ++i){
			for (int j = 0; j != ncols; ++j){
				// Adjust weight
				_W[l][i][j] += epsilon;
				// Forward propagate
				_forward_prop();
				// Calculate loss
				double loss_up = 0.0d;
				for (int k = 0; k != y.size(); ++k)
					loss_up += 0.5d * (y[k] - _o[_layers.size()-1][k])
						* (y[k] - _o[_layers.size()-1][k]);
				// Adjust weight
				_W[l][i][j] -= 2*epsilon;
				// Forward propagate
				_forward_prop();
				// Calculate loss
				double loss_down = 0.0d;
				for (int k = 0; k != y.size(); ++k)
					loss_down += 0.5d * (y[k] - _o[_layers.size()-1][k])
						* (y[k] - _o[_layers.size()-1][k]);
				// Refix weights
				_W[l][i][j] += epsilon;
				// Calculate numerical gradient
				double dnum = (loss_up - loss_down) / (2.0d * epsilon);
				if (abs(dnum - _dW[l][i][j]) > 1E-7)
					cout << dnum - _dW[l][i][j] << endl;
			}
		}
	}



	return;
}
