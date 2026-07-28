#include <stdio.h>
#include <queue>
#include <set>

using namespace std;

// Enum type to indicate exploration mode of a state.
enum State_mode {NEW, IN_OPEN, IN_CLOSED};

extern "C" {
    void start_timer();
    void stop_timer(float *time);
    __global__ void bfs_step_kernel(int *offsets, int *trans, State_mode *mode, int n, bool *update);
}

int bfs(int *offsets, int *trans) {
    queue<int> openq;
    set<int> closed;

    openq.push(0);
    closed.insert(0);
    
    int source, target;
    while (!openq.empty()) {
        source = openq.front();
        openq.pop();
        // iterate over outgoing transitions
        for (int t = offsets[source]; t < offsets[source+1]; t++) {
            target = trans[t];
            if (closed.find(target) == closed.end()) {
                openq.push(target);
                closed.insert(target);
            }
        }
    }
    return closed.size();
}

__global__ void bfs_step_kernel(int *offsets, int *trans, State_mode *mode, int n, bool *progress) {
    int step_size = gridDim.x * blockDim.x;
    int target;

    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < n; x += step_size) {
        //to do: perform one step for state x in bfs
    }
}

__global__ void count_closed_states(int *count, State_mode *mode, int n) {
    int local_count = 0;
    
    for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < n; x += gridDim.x * blockDim.x) {
        if (mode[x] == IN_CLOSED) {
            local_count++;
        }
    }
    atomicAdd(count, local_count);
}

int main() {

    int n = 1e6; //number of states in the graph
    int out, outsum = 0, outmin = 4, outmax = 20; //min and max number of outgoing transitions of a state
    float time;
    cudaError_t err;

    //allocate arrays and fill them
    int *offsets = (int *) malloc((n+1) * sizeof(int));
    offsets[0] = 0;
    //fill offsets array
    for (int i=0; i < n; i++) {
        out = (rand() % (outmax - outmin)) + outmin;
        outsum += out;
        offsets[i+1] = outsum;
    }
    int *trans = (int *) malloc(outsum * sizeof(int));
    //fill trans array
    for (int i=0; i < outsum; i++) {
        trans[i] = rand() % n;
    }
    printf("graph has %d transitions\n", outsum);
    //progress flag
    bool progress = true;
    //count variable
    int count = 0;
    
    //measure the CPU function
    start_timer();
    int seqcount = bfs(offsets, trans);
    stop_timer(&time);
    printf("bfs took %.3f ms\n", time);

    //allocate GPU memory
    int *d_offsets, *d_trans;
    State_mode *d_mode;
    //progress flag
    bool *d_progress;
    //count variable to count number of closed states
    int *d_count;
    err = cudaMalloc((void **)&d_offsets, (n+1)*sizeof(int));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_offsets: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_trans, outsum*sizeof(int));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_trans: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_mode, n*sizeof(State_mode));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_mode: %s\n", cudaGetErrorString( err ));     
    err = cudaMalloc((void **)&d_progress, sizeof(bool));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_progress: %s\n", cudaGetErrorString( err ));     
    err = cudaMalloc((void **)&d_count, sizeof(int));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_count: %s\n", cudaGetErrorString( err ));     
    
    //copy the input data to the GPU
    err = cudaMemcpy(d_offsets, offsets, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device offsets: %s\n", cudaGetErrorString( err ));
    err = cudaMemcpy(d_trans, trans, outsum*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device trans: %s\n", cudaGetErrorString( err ));

    //zero the open and closed arrays
    err = cudaMemset(d_mode, NEW, n*sizeof(State_mode));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemset d_mode: %s\n", cudaGetErrorString( err ));
    //open state 0
    State_mode M = IN_OPEN;
    err = cudaMemcpy(&(d_mode[0]), &M, sizeof(State_mode), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device d_mode 0: %s\n", cudaGetErrorString( err ));    
    
    
    //setup the grid and thread blocks
    int block_size = 1024;                          	//thread block size
    int nblocks = int(ceilf(n/(float)block_size));  //n divided by thread block size rounded up
    dim3 grid(nblocks, 1);
    dim3 threads(block_size, 1, 1);

    //measure the GPU function
    cudaDeviceSynchronize();
    start_timer();
    int iterations = 0;
    bool F = false;
    while (progress) {
        iterations++;
        err = cudaMemcpy(d_progress, &F, sizeof(bool), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device d_progress: %s\n", cudaGetErrorString( err ));   
        bfs_step_kernel<<<grid, threads>>>(d_offsets, d_trans, d_mode, n, d_progress);
        err = cudaMemcpy(&progress, d_progress, sizeof(bool), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy device to host d_progress: %s\n", cudaGetErrorString( err ));   
    }
    cudaDeviceSynchronize();
    stop_timer(&time);
    printf("bfs_step_kernel took %.3f ms, %d iterations\n", time, iterations);

    //check to see if all went well
    err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "Error during kernel launch bfs_step_kernel: %s\n", cudaGetErrorString( err ));

    //count the number of closed states
    err = cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device d_count: %s\n", cudaGetErrorString( err ));       
    count_closed_states<<<grid, threads>>>(d_count, d_mode, n);
    err = cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy device to host d_count: %s\n", cudaGetErrorString( err ));   
    
    //check the result
    if (count != seqcount) {
        printf("TEST FAILED!\n");
    } else {
        printf("TEST PASSED!\n");
    }

    //clean up
    cudaFree(d_offsets);
    cudaFree(d_trans);
    cudaFree(d_mode);
    free(offsets);
    free(trans);

    return 0;
}
