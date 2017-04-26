
#include "BMM.h"
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <cmath>

void initialize(std::vector<double> &pi, std::vector<std::vector<double>> &mu,
	std::vector<std::vector<double>> &gamma,std::size_t M, std::size_t K, std::size_t N){

	std::cout << "Initializing mu and pi coefficients..." << std::endl;

	// Initialize pi
	double norm = 0.0d;
	for (std::size_t k = 0; k != K; ++k){
		pi.push_back(0.1d + (rand() % 100)/125.0d);
		norm += pi[k];
	}
	// Normalize pi
	for (std::size_t k = 0; k != K; ++k){
		pi[k] = pi[k] / norm;
	}
	// Initialize mu
	for (std::size_t k = 0; k != K; k++){
        std::vector<double> row;
        for (std::size_t n = 0; n != N; n++)
			row.push_back(0.1d + (rand() % 100)/125.0d);
		mu.push_back(row);
	}

	// Allocating prob and gamma matrices
	std::cout << "Allocating gamma matrix..." << std::endl;

	for (std::size_t k = 0; k != K; ++k){
		std::vector<double> row;
		for (std::size_t m = 0; m != M; ++m){
            row.push_back(0.0d);
		}
		gamma.push_back(row);
	}

	return;
}

void Estep(std::vector<std::vector<double>> &gamma, std::vector<double> const &pi,
	std::vector<std::vector<double>> const &mu, std::vector<std::vector<double>> const &X,
	std::size_t M, std::size_t K, std::size_t N){

	std::vector<double> norm;
	norm.resize(M);
	#pragma omp parallel for // reduction(+:norm)
	for (std::size_t m = 0; m < M; m++){
		norm[m] = 0.0d;
		for (std::size_t k = 0; k < K; k++){
			double temp = 1.0d;
			for (std::size_t n = 0; n < N; n++){
				temp *= pow(1.0d * mu[k][n],1.0d * X[m][n]) * pow((1.0d - mu[k][n]),(1.0d - X[m][n]));
			}
			gamma[k][m] =  pi[k] * temp;
			norm[m] += gamma[k][m];
		}
	}

	// Normalize gamma matrix
	#pragma omp parallel for
	for (std::size_t k = 0; k < K; k++){
		for (std::size_t m = 0; m < M; m++){
            gamma[k][m] = gamma[k][m] / norm[m];
		}
	}

	return;
}

void Mstep(std::vector<std::vector<double>> const &gamma, std::vector<double> &pi,
	std::vector<std::vector<double>> &mu,std::vector<std::vector<double>> const &X,
	std::size_t M, std::size_t K, std::size_t N){

	//cout << "Mstep -- Updating parameters..." << endl;
	//auto start = std::chrono::high_resolution_clock::now();

	// Find denominator and update pi coefficients
	std::vector<double> den;
    for (std::size_t k = 0; k < K; k++){
		double temp = 0.0d;
        #pragma omp parallel for reduction(+:temp)
		for (std::size_t m = 0; m < M; m++){
            temp += gamma[k][m];
		}
        den.push_back(temp);
        pi[k] = (temp) / (1.0d * M);
    }

    // Update mu matrix
    #pragma omp parallel for
    for (std::size_t k = 0; k < K; k++){
		for (std::size_t n = 0; n < N; n++){
			mu[k][n] = 0.0d;
			for (std::size_t m = 0; m < M; m++){
                mu[k][n] += gamma[k][m] * X[m][n];
            }
            mu[k][n] /= den[k];
		}
    }

    //auto finish = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double> elapsed = finish - start;
	//cout << elapsed.count() << endl;

	return;
}
