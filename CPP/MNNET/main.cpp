#include "headers.h"

#include <iostream>

using namespace std;

int main(){

	// Initialize network
    network nnet(0.1,0.9);
    // Add layers (and initialize weights at the same time)
    layer l1(2,"none");
    nnet.add_layer(l1);
	layer l2(50,"sigmoid");
	nnet.add_layer(l2);
	layer l3(40,"sigmoid");
	nnet.add_layer(l3);
    layer l4(2,"sigmoid");
    nnet.add_layer(l4);

    cout << nnet.get_structure() << endl;

	// Forward propagation

	//vector<vector<double>> w;
	//nnet.get_weights(0,w);
    //print_matrix(w); cout << endl;
	//nnet.get_weights(1,w);
    //print_matrix(w); cout << endl;

    // Input
    vector<vector<double>> input;
    double a = 0.1;
    double b = 0.5;
    vector<double> r;
    r.push_back(a); r.push_back(b);
    input.push_back(r);

    // Output
    vector<vector<double>> output;
    double c = 0.2;
    double d = 0.8;
    vector<double> ().swap(r);
    r.push_back(c); r.push_back(d);
    output.push_back(r);

	nnet.numerical_gradient_check(input[0], output[0]);

	//return 0;
	//for (int i = 0; i != 10; i++){
		nnet.train(input,output,100);

		//nnet.get_dweights(0,w);
		//print_matrix(w); cout << endl;
		//nnet.get_dweights(1,w);
		//print_matrix(w); cout << endl;

		vector<double> out_2;
		nnet.get_output(3, out_2);
		print_vector(out_2); cout << endl;
	//}


	return 0;
}

