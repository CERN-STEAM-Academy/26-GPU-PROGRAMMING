#pragma once

#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include "multi_gpu_iterator.h"
#include "gpu_config.h"
#include "multi_gpu_policy.h"
#include <algorithm>
#include <vector>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

template <typename T>
class multi_gpu_vector {
private:
    T* gpu_ptrs[GPU_COUNT];
    size_t total_size;
    size_t ceil_gpu_size;

    void sync_gpus() {
        for (size_t i = 0; i < GPU_COUNT; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
    }

    void allocate(size_t n) {
        total_size = n;
        ceil_gpu_size = (n + GPU_COUNT - 1) / GPU_COUNT;

        for (int i = 0; i < GPU_COUNT; i++) {
            cudaSetDevice(i);
            cudaMalloc(&gpu_ptrs[i], ceil_gpu_size * sizeof(T));
        }
    }

    void set_peer_to_peer() {
        for (size_t i = 0; i < GPU_COUNT; i++) {
            cudaSetDevice(i);
            for (size_t j = 0; j < GPU_COUNT; j++) {
                if (i !=j)
                    cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }

public:
    multi_gpu_vector() {
        total_size = 0;
        ceil_gpu_size = 0;
        for (size_t i = 0; i < GPU_COUNT; i++)
            gpu_ptrs[i] = nullptr;
        set_peer_to_peer();
    }

    multi_gpu_vector(size_t size) : total_size(size) {
        ceil_gpu_size = (size + GPU_COUNT - 1) / GPU_COUNT;

        for (int i = 0; i < GPU_COUNT; i++) {
            cudaSetDevice(i);
            cudaMalloc(&gpu_ptrs[i], ceil_gpu_size * sizeof(T));
        }
        
        set_peer_to_peer();
    }

    ~multi_gpu_vector() {
        for (int i = 0; i < GPU_COUNT; ++i) {
            cudaSetDevice(i);
            cudaFree(gpu_ptrs[i]);
            gpu_ptrs[i] = nullptr;
        }
        cudaSetDevice(0);
    }

    multi_gpu_iterator<T> begin() {
        return multi_gpu_iterator<T>(gpu_ptrs, 0, ceil_gpu_size, total_size);
    }

    multi_gpu_iterator<T> end() {
        return multi_gpu_iterator<T>(gpu_ptrs, total_size, ceil_gpu_size, total_size);
    }

    size_t size() const {
        return total_size;
    }

    T operator[](size_t index) const {
        T val;
        size_t gpu_id = index / ceil_gpu_size;
        size_t gpu_idx = index % ceil_gpu_size;
        cudaSetDevice(index / ceil_gpu_size);
        printf("[] => %zu, ", index/ceil_gpu_size);
        cudaMemcpy(
            &val, 
            &gpu_ptrs[gpu_id][gpu_idx], 
            sizeof(T),
            cudaMemcpyDeviceToHost
        );

        return val;
    }

    void resize(size_t n) {
        if (n == total_size) return; // ignore

        // create new vector
        // TODO: Parallelize
        size_t new_ceil = (n + GPU_COUNT - 1) / GPU_COUNT;
        T* new_gpu_ptrs[GPU_COUNT];
        for (int i = 0; i < GPU_COUNT; i++) {
            cudaSetDevice(i);
            cudaMalloc(&new_gpu_ptrs[i], new_ceil * sizeof(T));
        }

        size_t num_to_copy = std::min(n, total_size);
        
        for (size_t i = 0; i < num_to_copy; i++) {
            size_t cur_gpu_id = i / ceil_gpu_size;
            size_t cur_gpu_idx = i % ceil_gpu_size;
            size_t new_gpu_id = i / new_ceil;
            size_t new_gpu_idx = i % new_ceil;
            printf("resize => %zu, ", new_gpu_id);
            cudaSetDevice(new_gpu_id);
            if (cur_gpu_id == new_gpu_id) {
                cudaMemcpy(
                    &new_gpu_ptrs[new_gpu_id][new_gpu_idx],
                    &gpu_ptrs[cur_gpu_id][cur_gpu_idx],
                    sizeof(T),
                    cudaMemcpyDeviceToDevice
                );
            } else {
                cudaMemcpyPeer(
                    &new_gpu_ptrs[new_gpu_id][new_gpu_idx],
                    new_gpu_id,
                    &gpu_ptrs[cur_gpu_id][cur_gpu_idx],
                    cur_gpu_id,
                    sizeof(T)
                );
            }
        }

        // free cur memory
        for (size_t i = 0; i < GPU_COUNT; i++) {
            cudaSetDevice(i);
            cudaFree(gpu_ptrs[i]);
        }

        // update vars
        total_size = n;
        ceil_gpu_size = new_ceil;
        for (size_t i = 0; i < GPU_COUNT; i++)
            gpu_ptrs[i] = new_gpu_ptrs[i];

        sync_gpus();
    }

    void operator=(thrust::device_vector<T>& from) {
        printf("=device\n");
        clear();
        cudaSetDevice(0);
        total_size = from.size();
        ceil_gpu_size = (total_size + GPU_COUNT - 1) / GPU_COUNT;
        T* d_ptr = from.data().get();

        // Local GPU Copy
        cudaSetDevice(0);
        if (cudaMalloc(&gpu_ptrs[0], ceil_gpu_size * sizeof(T)) != cudaSuccess)
            printf("Malloc Failure\n");
        if (cudaMemcpy(
                gpu_ptrs[0],
                d_ptr,
                ceil_gpu_size * sizeof(T),
                cudaMemcpyDeviceToDevice
        ) != cudaSuccess) printf("Failure ceil part\n");

        // Other copies
        for (size_t i = 1; i < GPU_COUNT - 1; i++) {
            printf("=device => %zu\n", i);
            cudaSetDevice(i);
            if (cudaMalloc(&gpu_ptrs[i], ceil_gpu_size * sizeof(T)) != cudaSuccess) {
                printf("Malloc Failure\n");
            }
            // TODO: IMplement GPU COPY to prevent 0 to 0
            if (cudaMemcpyPeer(
                gpu_ptrs[i],
                i,
                d_ptr + i * ceil_gpu_size,
                0,
                ceil_gpu_size * sizeof(T)
            ) != cudaSuccess) printf("Failure ceil part\n");
        }

        // Final off by one copy
        size_t i = GPU_COUNT - 1;
        size_t final_size = total_size - i * ceil_gpu_size;
        printf("=device => %zu\n", i);
        cudaSetDevice(i);
        if (cudaMalloc(&gpu_ptrs[i], ceil_gpu_size * sizeof(T)) != cudaSuccess) {
            printf("Malloc Failure\n");
        }
        if (cudaMemcpyPeer(
            gpu_ptrs[i],
            i, 
            d_ptr + i * ceil_gpu_size,
            0, 
            final_size * sizeof(T)
        ) != cudaSuccess) printf("Failure final part\n");
        cudaSetDevice(0);
	}

    void operator=(thrust::host_vector<T>& from) {
        printf("=host\n");
        clear();
        total_size = from.size();
        ceil_gpu_size = (total_size + GPU_COUNT - 1) / GPU_COUNT;
        T* h_ptr = from.data();
        for (size_t i = 0; i < GPU_COUNT - 1; i++) {
            cudaSetDevice(i);
            if (cudaMalloc(&gpu_ptrs[i], ceil_gpu_size * sizeof(T)) != cudaSuccess) {
                printf("Malloc Failure\n");
            }
            if (cudaMemcpy(
                gpu_ptrs[i], 
                h_ptr + i * ceil_gpu_size, 
                ceil_gpu_size * sizeof(T), 
                cudaMemcpyHostToDevice
            ) != cudaSuccess) printf("Failure ceil part\n");
        }

        size_t i = GPU_COUNT - 1;
        size_t final_size = total_size - i * ceil_gpu_size;
        cudaSetDevice(i);
        if (cudaMalloc(&gpu_ptrs[i], ceil_gpu_size * sizeof(T)) != cudaSuccess) {
            printf("Malloc Failure\n");
        }
        if (cudaMemcpy(
            gpu_ptrs[i], 
            h_ptr + i * ceil_gpu_size, 
            final_size * sizeof(T), 
            cudaMemcpyHostToDevice
        ) != cudaSuccess) printf("Failure final part\n");
	}

    void operator=(thrust::universal_vector<T>& from) {
        printf("=universal\n");
        assert(false);
	}

    operator thrust::device_vector<T>() {
        printf("device()\n");
        cudaSetDevice(0); // Test destination memcpy theory
        thrust::device_vector<T> v = thrust::device_vector<T>(total_size);
        T* d_ptr = v.data().get();

        // Local GPU Copy
        cudaSetDevice(0);
        if (cudaMemcpy(
                d_ptr,
                gpu_ptrs[0],
                ceil_gpu_size * sizeof(T),
                cudaMemcpyDeviceToDevice
        ) != cudaSuccess) printf("Failure ceil part\n");

        // Other copies
        for (size_t i = 1; i < GPU_COUNT - 1; i++) {
            printf("device() => %zu\n", i);
            // TODO: IMplement GPU COPY to prevent 0 to 0
            if (cudaMemcpyPeer(
                d_ptr + i * ceil_gpu_size,
                0,
                gpu_ptrs[i],
                i,
                ceil_gpu_size * sizeof(T)
            ) != cudaSuccess) printf("Failure ceil part\n");
        }

        // Off by one copy
        size_t i = GPU_COUNT - 1;
        size_t final_size = total_size - i * ceil_gpu_size;
        printf("device() => %zu\n", i);
        if (cudaMemcpyPeer(
            d_ptr + i * ceil_gpu_size,
            0,
            gpu_ptrs[i],
            i,
            final_size * sizeof(T)
        ) != cudaSuccess) printf("Failure final part\n");
        cudaGetLastError();
        return v;
    }

    operator thrust::host_vector<T>() {
        printf("host()\n");
        thrust::host_vector<T> h = thrust::host_vector<T>(total_size);
        T* h_ptr = h.data();
        for (size_t i = 0; i < GPU_COUNT - 1; i++) {
            cudaSetDevice(i);
            if (cudaMemcpy(
                h_ptr + i * ceil_gpu_size, 
                gpu_ptrs[i], 
                ceil_gpu_size * sizeof(T), 
                cudaMemcpyDeviceToHost
            ) != cudaSuccess) printf("Failure ceil part\n");
        }
        size_t i = GPU_COUNT - 1;
        size_t final_size = total_size - i * ceil_gpu_size;
        cudaSetDevice(i);
        if (cudaMemcpy(
            h_ptr + i * ceil_gpu_size, 
            gpu_ptrs[i], 
            final_size * sizeof(T), 
            cudaMemcpyDeviceToHost
        ) != cudaSuccess) printf("Failure final part\n");
        return h;
    }

    operator thrust::universal_vector<T>() {
        printf("universal()\n");
        assert(false);
        thrust::universal_vector<T> u = thrust::universal_vector<T>(total_size);
        return u;
    }

    void clear() {
        for (size_t i = 0; i < GPU_COUNT; i++) {
            cudaSetDevice(i);
            cudaFree(gpu_ptrs[i]);
        }
        total_size = 0;
        ceil_gpu_size = 0;
        sync_gpus();
        cudaSetDevice(0);
    }

    void push_back() {
        printf("push_back\n");
        assert(false);
    }

    T* data () {
        printf("data\n");
        assert(false);
        thrust::host_vector<T> h = thrust::host_vector<T>(total_size); 
        // thrust::copy(multi_gpu_policy, begin(), end(), h.begin());
        return h.data();
    }

    void swap (std::vector<T>& other) {
        printf("***swap\n");
        // std::vector<T> v = std::vector<T>(other.size());
        // thrust::copy(multi_gpu_policy, other.begin(), other.end(), v.begin());
        // thrust::copy(multi_gpu_policy, begin(), end(), other.begin());
        // thrust::copy(multi_gpu_policy, v.begin(), v.end(), begin());
        assert(false);
    }
};