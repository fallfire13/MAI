CC=g++
CFLAGS=-std=c++17 -pedantic -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable
LFLAGS=-lzmq -lpthread
SOURCES=diff.cpp diff.h linear_diff.h main.cpp structs.h
EXECUTABLE=main

OBJECTS=&(SOURCES:.cpp=.o)

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
        $(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LFLAGS) -Iinclude

.cpp.o:
        $(CC) $(CFLAGS) -c $< -o $@ $(LFLAGS) -Iinclude

clean:
        rm -f *.o $(EXECUTABLE)