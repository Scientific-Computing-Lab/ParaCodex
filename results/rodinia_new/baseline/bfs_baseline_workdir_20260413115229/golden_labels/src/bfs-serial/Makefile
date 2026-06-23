# C compiler
CC = g++
ICC = icc
CC_FLAGS = -g -O2
CC_FLAGS += $(CFLAGS)

all: bfs

bfs: bfs.cpp
	$(CC) $(CC_FLAGS) bfs.cpp -o bfs

clean:
	rm -f bfs result.txt
