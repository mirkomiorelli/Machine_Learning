#ifndef GUARD_HELPER_H
#define GUARD_HELPER_H

int ReverseInt(int);
void ReadMNIST(int,int,std::vector<std::vector<double>> &,std::string);
void normalize(std::vector<std::vector<double>> &, std::size_t,std::size_t);
void getBinary(std::vector<std::vector<double>> &, std::size_t,std::size_t);

#endif // GUARD_HELPER_H
