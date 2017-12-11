#include "headers.h"

using namespace std;

layer::layer(const int nodes,const std::string activation){
	_nodes = nodes;
    _activation = activation;
}

layer::~layer(){}

int layer::get_nodes(){return _nodes;}

string layer::get_activation(){return _activation;}
