
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>  // for high_resolution_clock
#include <iomanip>
#include <random>
#include <algorithm>

// Header to read the MNIST dataset, helper functions to normalize
// and format images to binary
#include "helper.h"
// Bernoulli mizture model and EM algorithm functions
#include "BMM.h"

int main(){

	// Initialize random engine (for data shuffling)
	std::random_device rd;
    std::mt19937 g(rd());

	// Read dataset: each row is an image
	std::vector<std::vector<double>> ar;
	std::size_t N = 784;
	std::size_t M = 60000;
	ReadMNIST(M,N,ar);

	// Normalize the data and convert to binary
	normalize(ar,M,N);
	getBinary(ar,M,N);

	// Declare variables
	std::size_t K = 10;	// classes
	std::vector<double> pi;
	std::vector<std::vector<double>> mu;
	std::vector<std::vector<double>> gamma;

	// Initialize coefficients
	initialize(pi,mu,gamma,M,K,N);

	auto start = std::chrono::high_resolution_clock::now();

	// Optimization
	int iterMax = 50;
	int i = 0;
	while(i != iterMax){
		std::cout << "Iteration: " << i << std::endl;
		std::shuffle(ar.begin(), ar.end(), g);
		Estep(gamma,pi,mu,ar,M,K,N);
		Mstep(gamma,pi,mu,ar,M,K,N);
		i++;
	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Time for optimization: " << elapsed.count() << "s" << std::endl;


	// Print to file
    std::ofstream file("mu.dat");
    for (std::size_t k = 0; k < K; k++){
		for (std::size_t n = 0; n < N; n++){
			file << mu[k][n] << "\t";
		}
		file << "\n";
    }
    file.close();

	return 0;
}

