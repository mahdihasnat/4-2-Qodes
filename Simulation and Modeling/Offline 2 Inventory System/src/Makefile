MAKEFLAGS += -j8

CC = g++ -std=c++17 -Wall
PARAMETERS = 
# compile every cpp file in this directory with g++ , and target

# the object file in the same directory
%.o : %.cpp %.h
	${CC} -c $< -o $@

clean :
	rm -f *.o
	rm -f *.out

.PHONY : clean

a.out : main.o
	${CC} *.o -o a.out

# run the executable
run : a.out
	./a.out ${PARAMETERS}

new.out : main.o
	${CC} main.o -o new.out

# compile and run
all : run
	
time : timer.o
	g++ -o time timer.o
	./time

# compile and debug memory-check
memory : a.out
	valgrind --leak-check=full -s ./a.out input.txt
# compile and run
.PHONY : run

# compile and run
.PHONY : all

# compile and run
.PHONY : clean

.PHONY : run
