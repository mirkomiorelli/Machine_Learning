OBJS = main.o c_types.o nnet.o
CC = g++
DEBUG = -g
CFLAGS = -Wall -c -std=c++11 -fopenmp -O3 #$(DEBUG)
LFLAGS = -Wall -std=c++11 -fopenmp #$(DEBUG)

nn.x : $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o nn.x

nnet.o: c_types.h nnet.h nnet.cpp
	$(CC) $(CFLAGS) nnet.cpp

main.o: c_types.h main.cpp
	$(CC) $(CFLAGS) main.cpp

c_types.o: c_types.h c_types.cpp
	$(CC) $(CFLAGS) c_types.cpp

clean:
	rm *.o *.x
