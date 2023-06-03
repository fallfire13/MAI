#FLAGS=-std=c++11 -pedantic -Wall -Werror -Wno-sign-compare -Wno-long-long -O2
#CC=g++
#LINK=-lm
#
#all: node tree main
#
#main: main.cpp
#	$(CC) $(FLAGS) -c main.cpp
#	$(CC) $(FLAGS) -o TPatriciaTrieNode.o TPatriciaTrie.o main.o $(LINK)
#
#tree: PatriciaTrie.cpp
#	$(CC) $(FLAGS) -c TPatriciaTrie.cpp
#
#node: PatriciaTrieNode.cpp
#	$(CC) $(FLAGS) -c TPatriciaTrieNode.cpp
#
#clear:
#	rm -f *.o
#	rm -fr *.dSYM
#quick
CC = g++
FLAGS = -std=c++11 -w -O2 -Wextra -Wall -Werror -Wno-sign-compare -Wno-unused-result -pedantic -lm
FILES = main.cpp TPatriciaTrie.cpp TPatriciaTrieNode.cpp
PROG = solution

all: da2

da2:
	$(CC) $(FLAGS) -o $(PROG) $(FILES)

clean:
	rm -f *.o solution

# all:
# 	g++ -std=c++11 -pedantic -Wall -Werror -Wno-sign-compare -Wno-long-long main.cpp TPatriciaTrie.cpp TPatriciaTrieNode.cpp -o da2 -lm
#
# clean:
# 	rm -f *.o da2
