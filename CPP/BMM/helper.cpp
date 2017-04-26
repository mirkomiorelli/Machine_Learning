
/*********************************************************************************************
 The MNIST dataset can be found here:
 http://yann.lecun.com/exdb/mnist/
 This code to read the MNIST dataset is from
 http://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c/10409376#10409376
*********************************************************************************************/

#include <vector>
#include <iostream>
#include <fstream>

#include "helper.h"

int ReverseInt (int i){
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void ReadMNIST(int NumberOfImages, int DataOfAnImage,std::vector<std::vector<double>> &arr) {
    arr.resize(NumberOfImages,std::vector<double>(DataOfAnImage));
    std::ifstream file("data/train-images.idx3-ubyte",std::ios::binary);
    if (file.is_open()) {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        std::cout << magic_number << std::endl;
        std::cout << number_of_images << std::endl;
        std::cout << n_rows << " " << n_cols << std::endl;
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
}

void normalize(std::vector<std::vector<double>> &arr, std::size_t M,std::size_t N){

	std::cout << "Normalizing images..." << std::endl;
	for (std::size_t m = 0; m != M; ++m){
		for (std::size_t n = 0; n != N; ++n){
			arr[m][n] = arr[m][n] / 255.0d;
		}
	}
	return;
}

void getBinary(std::vector<std::vector<double>> &arr, std::size_t M, std::size_t N){

	std::cout << "Binary encoding of images..." << std::endl;
	for (std::size_t m = 0; m != M; ++m){
		for (std::size_t n = 0; n != N; ++n){
			if (arr[m][n] >= 0.5d)
				arr[m][n] = 1.0d;
			else
				arr[m][n] = 0.0d;
		}
	}
	return;
}

