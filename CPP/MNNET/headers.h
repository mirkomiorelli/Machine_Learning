

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

class layer;
class network;
void print_vector(std::vector<double> &);
void print_matrix(std::vector<std::vector<double>> &);

class layer{
	public:
		layer(const int,const std::string);
		~layer();
		// Get functions
		int get_nodes();
		std::string get_activation();
	private:
        int _nodes = -1;
        std::string _activation = "sigmoid";
};

class network{
	public:
		network(double,double);
		~network();
		void add_layer(layer const &);
		std::string get_structure();
		void get_weights(int const, std::vector<std::vector<double>> &);
		void get_dweights(int const, std::vector<std::vector<double>> &);
		void get_output(int const, std::vector<double> &);
		void train(std::vector<std::vector<double>> const &,
			std::vector<std::vector<double>> const &,
			int const);
		void numerical_gradient_check(std::vector<double> const &,
			std::vector<double> const &);
		void predict(std::vector<double> const &, std::vector<double> &);
	private:
		// Functions
		void _init_weights(int const, int const);
		void _forward_prop();
		void _collapse(int const);
		void _activate(int const);
		double _dactivate(int const, std::vector<double> const &,
			std::vector<std::vector<double>> const &);
		void _backward_prop();
		void _delta(uint const, std::vector<double> const &, std::vector<double> &);
		void _update_weights();
		// Variables
		// Random engine
		std::default_random_engine _generator;
		std::uniform_real_distribution<double> _distribution;
		// Network structure
		std::vector<layer> _layers;
		std::vector<layer>::size_type _nlayers;
		// Weight matrices
		std::vector<std::vector<std::vector<double>>> _W;
		std::vector<std::vector<std::vector<double>>> _dW;
		std::vector<std::vector<std::vector<double>>> _dWold;
		// Output state variables
		std::vector<std::vector<double>> _o;
		// Target vector
		std::vector<double> _y;
		// Learning rate
		double _lrate;
		double _momentum;
		// Epochs
		int _epochs;
};

