# C compiler
CC = gcc
CFLAGS ?= -g -O2


backprop: backprop.o facetrain.o imagenet.o backprop_kernel.o 
	$(CC) $(CFLAGS) backprop.o facetrain.o imagenet.o backprop_kernel.o -o backprop -lm

%.o: %.[ch]
	$(CC) $(CFLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CFLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CFLAGS) backprop.c -c

backprop_kernel.o: backprop_kernel.c backprop.h
	$(CC) $(CFLAGS) backprop_kernel.c -c

imagenet.o: imagenet.c backprop.h
	$(CC) $(CFLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
