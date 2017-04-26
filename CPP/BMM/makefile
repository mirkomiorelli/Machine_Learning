OBJS = main.o helper.o BMM.o 
CC = g++
DEBUG = -g
CFLAGS = -Wall -c -std=c++11 -fopenmp -O3 #$(DEBUG)
LFLAGS = -Wall -fopenmp #$(DEBUG)

run.x : $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o run.x

helper.o: helper.cpp
	$(CC) $(CFLAGS) helper.cpp

BMM.o: BMM.cpp
	$(CC) $(CFLAGS) BMM.cpp

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

clean:
	rm *.o *.x
