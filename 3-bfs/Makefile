CFLAGS = -O3
CC = nvcc

#insert the ignore for deprecated gpu targets if using CUDA 8.0
CFLAGS += $(shell if [ `nvcc --version | grep 8.0 | wc -l` != 0 ]; then echo -Wno-deprecated-gpu-targets ; fi)

all: bfs

bfs: timer.o bfs.o
	$(CC) $(CFLAGS) -o bfs timer.o bfs.o

clean:
	rm -f *.o bfs

%.o: %.cc
	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	$(CC) $(CFLAGS) -o $@ -c $<
