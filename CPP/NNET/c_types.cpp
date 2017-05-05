#include <iostream>

#include "c_types.h"


template <typename T>
cmatrix<T>::cmatrix(){

}

template<typename T>
void cmatrix<T>::initialize(size_t nr, size_t nc){
	_nr = nr;
	_nc = nc;
	_el.resize(_nr);
	for (size_t r = 0; r != _nr; r++){
		_el[r].resize(_nc);
	}
}

template <typename T>
void cmatrix<T>::fill(T val){
	for (typename std::vector<T>::size_type r = 0; r != _nr; r++){
		for (typename std::vector<T>::size_type c = 0; c != _nc; c++){
			_el[r][c] = val;
		}
	}

	return;
}

template <typename T>
void cmatrix<T>::fill_random_uniform(){
	for (typename std::vector<T>::size_type r = 0; r != _nr; r++){
		for (typename std::vector<T>::size_type c = 0; c != _nc; c++){
			_el[r][c] = (rand() % 1000) / 1000.0d;
		}
	}
	return;
}

// Reshape and fill with zeros
template<typename T>
void cmatrix<T>::reshape(size_t nr, size_t nc){
	for (size_t r = 0; r != _nr; r++){
		_el[r].resize(0);
	}
	_nr = nr;
	_nc = nc;
	_el.resize(_nr);
	for (size_t r = 0; r != _nr; r++){
		_el[r].resize(_nc);
	}

	return;
}

template <typename T>
void cmatrix<T>::print(){
	for (typename std::vector<T>::size_type r = 0; r != _nr; r++){
		for (typename std::vector<T>::size_type c = 0; c != _nc; c++){
			std::cout << _el[r][c] << "\t";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
	return;
};

template <typename T>
typename std::vector<std::vector<T>>::size_type cmatrix<T>::rows(){
	return _nr;
}

template <typename T>
typename std::vector<std::vector<T>>::size_type cmatrix<T>::cols(){
	return _nc;
}

template<typename T>
T cmatrix<T>::get(typename std::vector<std::vector<T>>::size_type i,
	typename std::vector<std::vector<T>>::size_type j){

	return _el[i][j];

}

template<typename T>
void cmatrix<T>::set(typename std::vector<std::vector<T>>::size_type i,
	typename std::vector<std::vector<T>>::size_type j, T val){

	_el[i][j] = val;

	return;
}

// Perform A*B
template<typename T>
void cmatrix<T>::dot(cmatrix<T> &A, cmatrix<T> &B){

	// Check dimensions
	if (A.rows() != rows() || A.cols() != B.rows() || B.cols() != cols()){
		std::cout << "Error in matrix dimensions!!!" << std::endl;
	} else {
        _vsz K = A.cols();
        for (_vsz r = 0; r != _nr; r++){
            for (_vsz c = 0; c != _nc; c++){
				_el[r][c] = 0;
				for (_vsz k = 0; k != K; k++){
                    _el[r][c] += A.get(r,k) * B.get(k,c);
				}
            }
        }
	}

	return;
}

template <typename T>
void cmatrix<T>::dot(cmatrix<T> &A, std::vector<T> &v){

	// Check dimensions
	if (A.rows() != rows() || A.cols() != v.size() || v.size() != cols()){
		std::cout << "Error in matrix dimensions!!!" << std::endl;
	} else {
        _vsz K = v.size();
        for (_vsz r = 0; r != _nr; r++){
            for (_vsz c = 0; c != _nc; c++){
				_el[r][c] = 0;
				for (_vsz k = 0; k != K; k++){
                    _el[r][c] += A.get(r,k) * v[c];
				}
            }
        }
	}

	return;
}

template <typename T>
void cmatrix<T>::dot(std::vector<T> &v, cmatrix<T> &A){

	// Check dimensions
	if (v.size() != rows() || A.rows() != v.size() || A.cols() != cols()){
		std::cout << "Error in matrix dimensions!!!" << std::endl;
	} else {
        _vsz K = v.size();
        for (_vsz r = 0; r != _nr; r++){
            for (_vsz c = 0; c != _nc; c++){
				_el[r][c] = 0;
				for (_vsz k = 0; k != K; k++){
                    _el[r][c] += A.get(k,c) * v[r];
				}
            }
        }
	}

	return;
}


template<typename T>
cvector<T>::cvector(){

}

template <typename T>
void cvector<T>::initialize(std::size_t nc){
    // Initialize as row vector
    _nc = nc;
    _el.resize(_nc);
    return;
}

template <typename T>
void cvector<T>::copy(std::vector<T> const &v){
	std::copy(v.begin(),v.end(),_el.begin());
	return;
}

template <typename T>
void cvector<T>::push_back(T el){
	_el.push_back(el);
	return;
}

template <typename T>
void cvector<T>::print(){
    for (typename std::vector<T>::iterator it = _el.begin(); it != _el.end(); ++it){
        std::cout << (*it) << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void cvector<T>::set(std::size_t idx, T val){
	_el.at(idx) = val;
	return;
}

template <typename T>
T cvector<T>::get(std::size_t idx){
	return _el[idx];
}

template <typename T>
std::size_t cvector<T>::size(){
	return _el.size();
}

template <typename T>
void cvector<T>::dot(cvector<T> &v, cmatrix<T> &M){
    if (v.size() != M.rows()){
		std::cout << "Number of rows of matrix different from number of elements vector!" << std::endl;
		return;
    }
    if (_nc != M.cols()){
		std::cout << "Output vector dimensions mismatch!" << std::endl;
		std::cout << "Input vector size: " << v.size() << std::endl;
		std::cout << "Input Matrix size: " << M.rows() << " " << M.cols() << std::endl;
		std::cout << "Output vector size: " << _el.size() << std::endl;

		return;
    }

	// NB: special product, last element if bias is untouched
	std::size_t nr = v.size();
	//std::size_t nc = M.cols();
	for (std::size_t c = 0; c != _nc; c++){
		_el[c] = 0;
		for (std::size_t r = 0; r != nr; r++){
            _el[c] += v.get(r) * M.get(r,c);
		}
    }

	return;
}


template class cmatrix<double>;
template class cmatrix<int>;
template class cvector<double>;
template class cvector<int>;
