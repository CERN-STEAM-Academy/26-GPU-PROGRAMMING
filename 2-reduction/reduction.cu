#include <stdio.h>

extern "C" {
    void start_timer();
    void stop_timer(float *time);
    __global__ void reduce_kernel(float *out_array, float *in_array, int n);
}

//a naive summation in C
float sum_floats(float *in_array, int n) {
    float sum = 0.0;
    for (int i=0; i<n; i++) {
        sum += in_array[i];
    }
    return sum;
}

//Kahan summation to avoid floating-point precision errors
float sum_floats_kahan(float *in_array, int n) {
    float sum = 0.0;
    float c = 0.0;
    for (int i=0; i<n; i++) {
        float v = in_array[i] - c;
        float t = sum + v;
        c = (t - sum) - v;
        sum = t;
    }
    return sum;
}

//CUDA kernel for parallel reduction
__global__ void reduce_kernel(float *out_array, float *in_array, int n) {
	//produce an output array that is half the size of the input array.
	//let half of the threads add up elements in the input array,
	//and write the result as one element in the output array
	//be careful not to ignore elements in case the size of the input array is odd!
}


int main() {

    int n = (int)5e7; //problem size
    float time;
    cudaError_t err;

    //allocate arrays and fill them
    float *in_array = (float *) malloc(n * sizeof(float));
    float out_result = 0;
    for (int i=0; i<n; i++) {
        in_array[i] = (rand() % 10000) / 100000.0;
    }

    //measure the CPU function
    start_timer();
    float sum = sum_floats(in_array, n);
    stop_timer(&time);
    printf("sum_floats took %.3f ms\n", time);

    //setup the grid and thread blocks
    int block_size_x = 1024;                               //thread block size
    int num_blocks = int(ceilf(n/2/(float)block_size_x));  //half the problem size divided by thread block size rounded up
    dim3 grid(num_blocks, 1);
    dim3 threads(block_size_x, 1, 1);

    //allocate GPU memory
    float *d_in, *d_out, *d_tmp;
    err = cudaMalloc((void **)&d_in, n*sizeof(float));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_out, num_blocks*block_size_x*sizeof(float));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString( err ));

    //copy the input data to the GPU
    err = cudaMemcpy(d_in, in_array, n*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device: %s\n", cudaGetErrorString( err ));

    //zero the output array
    err = cudaMemset(d_out, 0, num_blocks*sizeof(float));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemset: %s\n", cudaGetErrorString( err ));

    //measure the GPU function
    cudaDeviceSynchronize();
    start_timer();
    cudaDeviceSynchronize();
    for (int i = n; i > 1; i /= 2) {
    	fprintf(stdout, "size of input array: %d\n", i);
    	reduce_kernel<<<grid, threads>>>(d_out, d_in, i); //call the kernel for each reduction iteration
        cudaDeviceSynchronize();
        // swap in and output
        d_tmp = d_out;
    	d_out = d_in;
    	d_in = d_tmp;
    }
    stop_timer(&time);
    printf("reduce_kernel took %.3f ms\n", time);

    //check to see if all went well
    err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "Error during kernel launch: %s\n", cudaGetErrorString( err ));

    //copy the result back to host memory
    err = cudaMemcpy(&out_result, d_in, 1*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy device to host: %s\n", cudaGetErrorString( err ));

    //compute a reliable reference answer on the host
    float sum2 = sum_floats_kahan(in_array, n);

    //check the result
    float diff = abs(out_result - sum2);
    printf("cpu: %f, corrected: %f\n", sum, sum2);
    printf("gpu: %f\n", out_result);

    if (diff < 1.0) {
        printf("TEST PASSED!\n");
    } else {
        printf("TEST FAILED!\n");
    }

    //clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(in_array);

    return 0;
}
