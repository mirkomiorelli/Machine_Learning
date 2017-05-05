#ifndef GUARD_NNET_H
#define GUARD_NNET_H

#include <map>
#include <string>
#include "c_types.h"

typedef std::vector<double>::size_type vsz;

class network{
	public:
        network();
        void initialize(vsz,vsz);	// initialize input and output layers
        void addLayer(vsz);			// Add a hidden layer
        void printLabels();			// Print structure labels
        void train(std::vector<double> const &,std::vector<double> const &);
        void numgradcheck();
	private:
		// Variables
        std::vector<vsz> _nodes;
        std::vector<std::string> _labels;
        std::map<std::string,cvector<double>> _inputs;
        std::map<std::string,cvector<double>> _outputs;
		std::map<std::string,cmatrix<double>> _W;
		std::map<std::string,cmatrix<double>> _dW;
		std::map<std::string,cvector<double>> _delta;
		// Functions
		void _forwardProp(std::vector<double> const &);
		void _collapse(cvector<double> &, cmatrix<double> &, cvector<double> &);
		void _activate(cvector<double> &, cvector<double> &);
		void _backProp(std::vector<double> const &);
		void _updateW();
};

#endif // GUARD_NNET_H
