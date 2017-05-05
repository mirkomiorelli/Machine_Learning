
#include <iostream>
#include "c_types.h"
#include "nnet.h"

int main(){

	cmatrix<double> A;
	A.initialize(2,3);
	A.fill(1.0);
	cmatrix<double> B;
	B.initialize(3,2);
	B.fill(2.5);
	cmatrix<double> C;
	C.initialize(2,2);

    C.dot(A,B);
    //C.print();

	network nnet;
	nnet.initialize(2,2);
	nnet.addLayer(3);
	nnet.addLayer(7);
	nnet.addLayer(8);
	//nnet.printLabels();
	std::vector<double> x;
	x.push_back(1.0);
	x.push_back(1.0);
	std::vector<double> y;
	y.push_back(0.9);
	y.push_back(0.1);


	nnet.train(x,y);
	nnet.numgradcheck();

	return 0;
}
