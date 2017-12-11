#include "headers.h"

using namespace std;

void print_vector(vector<double> &v){
    for (vector<double>::iterator vi = v.begin(); vi != v.end(); vi++)
		cout << setprecision(10) << (*vi) << "\t";
	cout << endl;
	return;
}

void print_matrix(vector<vector<double>> &m){
	for (vector<vector<double>>::size_type i = 0; i != m.size(); i++)
        print_vector(m[i]);
	return;
}

