#ifndef GUARD_BMM_H
#define GUARD_BMM_H

#include <vector>

void initialize(std::vector<double> &, std::vector<std::vector<double>> &,
	std::vector<std::vector<double>> &, std::size_t, std::size_t, std::size_t);
void Estep(std::vector<std::vector<double>> &, std::vector<double> const &,
	std::vector<std::vector<double>> const &, std::vector<std::vector<double>> const &,
	std::size_t, std::size_t, std::size_t);
void Mstep(std::vector<std::vector<double>> const &, std::vector<double> &,
	std::vector<std::vector<double>> &, std::vector<std::vector<double>> const &,
	std::size_t, std::size_t, std::size_t);

#endif // GUARD_BMM_H
