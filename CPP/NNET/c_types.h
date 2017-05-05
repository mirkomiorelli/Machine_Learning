#ifndef GUARD_CTYPES_H
#define GUARD_CTYPES_H

#include <vector>

// Matrix class
template <typename T>
class cmatrix{
	public:
		// Initialization given shape
        cmatrix();
		void initialize(std::size_t,std::size_t);
		void reshape(std::size_t, std::size_t);
		void fill(T);	// Fill matrix with value
		void fill_random_uniform(); // Fill matrix with uniform random values
		void copy(std::vector<T> const &); 	// Copy vector in
		void copy(cmatrix<T> const &);		// Copy matrix in
		void print();	// Print matrix to screen
        typename std::vector<std::vector<T>>::size_type rows();	// Return number of rows
        typename std::vector<std::vector<T>>::size_type cols();	// Return number of columns
        void dot(cmatrix<T> &, cmatrix<T> &);	// Fill matrix with dot product of two matrices
        void dot(cmatrix<T> &, std::vector<T> &);
        void dot(std::vector<T> &,cmatrix<T> &);
        T get(typename std::vector<std::vector<T>>::size_type,
			typename std::vector<std::vector<T>>::size_type);	// Return element of matrix at given position
		void set(typename std::vector<std::vector<T>>::size_type,
			typename std::vector<std::vector<T>>::size_type, T); // Set element of matrix at given position
	private:
        std::vector<std::vector<T>> _el;
		typename std::vector<std::vector<T>>::size_type _nr, _nc;
		typedef typename std::vector<T>::size_type _vsz;
};

// Vector class
template<typename T>
class cvector{
	public:
		cvector();
        void initialize(std::size_t);
        void copy(std::vector<T> const &);
        void push_back(T);
        void print();
        void set(std::size_t, T);
        T get(std::size_t);
        std::size_t size();
        void dot(cmatrix<T> &, cvector<T> &);
        void dot(cvector<T> &, cmatrix<T> &);
	private:
		std::size_t _nc;
		std::vector<T> _el;
};

#endif // GUARD_CTYPES_H
