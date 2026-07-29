#include <stdio.h>

extern "C" {
    void start_timer();
    void stop_timer(float *time);
    __global__ void matrix_sum_kernel(float *c, float *a, int nrows, int ncols);
}

int compare_arrays(float *c, float *d, int n);

void matrix_sum(float *c, float *a, int nrows, int ncols) {
    for (int i=0; i<nrows; i++) {
    	for (int j=0; j<ncols; j++) {
    		c[i] += a[i*ncols+j];
    	}
    }
}

__global__ void matrix_sum_kernel(float *c, float *a, int nrows, int ncols) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x < nrows) {
        //To do: change for-loop to correctly fetch the row elements
    	for (int i = 0; i < ncols; i++) {
    		c[x] += a[x*ncols+i];
    	}
    }
}


int main() {

    int nrows = 1e8; //Number of rows in matrices
    int ncols = 4; //Number of columns in matrices
    float time;
    cudaError_t err;

    //Allocate arrays and fill them
    float *a = (float *) malloc(nrows * ncols * sizeof(float));
    float *c = (float *) malloc(nrows * sizeof(float));
    float *d = (float *) malloc(nrows * sizeof(float));
    for (int i=0; i<(nrows * ncols); i++) {
        a[i] = 1.0 / rand();
    }
    for (int i=0; i<nrows; i++) {
    	c[i] = 0.0;
        d[i] = 0.0;
    }

    float *e = (float *) malloc(nrows*ncols*sizeof(float));

    //Measure the CPU function
    start_timer();
    matrix_sum(c, a, nrows, ncols);
    stop_timer(&time);
    printf("matrix_add took %.3f ms\n", time);

    //To do: rearrange data in matrix a to coalesce memory accesses
    //You can use matrix e for this
    for (int i=0; i<nrows; i++)
    {
    	for (int j=0; j<ncols; j++)
    	{
    	    e[i*ncols+j] = a[i*ncols+j];
    	}
    }
    a = e;

    //Allocate GPU memory
    float *d_a; float *d_c;
    err = cudaMalloc((void **)&d_a, nrows*ncols*sizeof(float));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_a: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_c, nrows*sizeof(float));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_c: %s\n", cudaGetErrorString( err ));

    //Copy the input data to the GPU
    err = cudaMemcpy(d_a, a, nrows*ncols*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device a: %s\n", cudaGetErrorString( err ));

    //Zero the output array
    err = cudaMemset(d_c, 0, nrows*sizeof(float));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemset c: %s\n", cudaGetErrorString( err ));

    //Setup the grid and thread blocks
    int block_size = 1024;                          	//thread block size
    int nblocks = int(ceilf(nrows/(float)block_size));  //nrows divided by thread block size rounded up
    dim3 grid(nblocks, 1);
    dim3 threads(block_size, 1, 1);

    //Measure the GPU function
    cudaDeviceSynchronize();
    start_timer();
    matrix_sum_kernel<<<grid, threads>>>(d_c, d_a, nrows, ncols);
    cudaDeviceSynchronize();
    stop_timer(&time);
    printf("matrix_sum_kernel took %.3f ms\n", time);

    //Check to see if all went well
    err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "Error during kernel launch matrix_sum_kernel: %s\n", cudaGetErrorString( err ));

    //Copy the result back to host memory
    err = cudaMemcpy(d, d_c, nrows*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy device to host c: %s\n", cudaGetErrorString( err ));

    //Check the result
    int errors = compare_arrays(c, d, nrows);
    if (errors > 0) {
        printf("TEST FAILED!\n");
    } else {
        printf("TEST PASSED!\n");
    }


    //Clean up
    cudaFree(d_a);
    cudaFree(d_c);
    free(a);
    free(c);
    free(d);

    return 0;
}



int compare_arrays(float *a1, float *a2, int n) {
    int errors = 0;
    int print = 0;

    for (int i=0; i<n; i++) {

        if (isnan(a1[i]) || isnan(a2[i])) {
            errors++;
            if (print < 10) {
                print++;
                fprintf(stderr, "Error NaN detected at i=%d,\t a1= %10.7e \t a2= \t %10.7e\n",i,a1[i],a2[i]);
            }
        }

        unsigned int int_a1 = *(unsigned int *)(a1+i);
        unsigned int int_a2 = *(unsigned int *)(a2+i);
        unsigned int dist = (unsigned int)0;
        if (int_a1 > int_a2) {
            dist = int_a1 - int_a2;
        } else {
            dist = int_a2 - int_a1;
        }
        if (dist > 0) {
            errors++;
            if (print < 10) {
                print++;
                fprintf(stderr, "Error detected at i=%d, \t a1= \t %10.7e \t a2= \t %10.7e \t ulp_dist=\t %u\n",i,a1[i],a2[i],dist);
            }
        }

    }

    return errors;
}
