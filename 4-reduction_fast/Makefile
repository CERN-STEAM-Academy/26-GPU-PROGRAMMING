CFLAGS = -O3
CC = nvcc

#insert the ignore for deprecated gpu targets if using CUDA 8.0
CFLAGS += $(shell if [ `nvcc --version | grep 8.0 | wc -l` != 0 ]; then echo -Wno-deprecated-gpu-targets ; fi)

all: reduction_fast

reduction_fast: timer.o reduction_fast.o
	$(CC) $(CFLAGS) -o reduction_fast timer.o reduction_fast.o

clean:
	rm -f *.o reduction_fast

%.o: %.cc
	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	$(CC) $(CFLAGS) -o $@ -c $<
