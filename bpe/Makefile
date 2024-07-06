CFLAGS = -O2

all: mnbpe;

debug: CFLAGS = -g -Og -pg
debug: mnbpe;

mnbpe: mnbpe.cpp classes.o functions.o;
	g++ $(CFLAGS) -fopenmp classes.o functions.o mnbpe.cpp -o mnbpe

functions.o: functions.cpp functions.hpp classes.hpp;
	g++ $(CFLAGS) -c -fopenmp functions.cpp -o functions.o

classes.o: classes.cpp classes.hpp;
	g++ $(CFLAGS) -c classes.cpp -o classes.o

clean: 
	rm functions.o classes.o mnbpe
