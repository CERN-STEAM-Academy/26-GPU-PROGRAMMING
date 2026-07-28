#pragma once

#include <thrust/iterator/iterator_facade.h>
#include <cuda_runtime.h>
#include "gpu_config.h"

// Derived, Value, System, Traversal, Reference, (Difference)
template <typename T>
class multi_gpu_iterator
    : public thrust::iterator_facade<
        multi_gpu_iterator<T>,
        T,
        multi_gpu_tag,
        thrust::random_access_traversal_tag,
        T
      > 
{
private:
    friend class thrust::iterator_core_access;

    T* ptrs[GPU_COUNT];
    size_t idx;
    size_t gpu_size;
    size_t total_size;

    // Minimal functions
    void increment() {
        idx++;
    }
    void decrement() {
        idx--;
    }

    void advance(std::ptrdiff_t n) {
        idx += n;
    }
    
    std::ptrdiff_t distance_to(const multi_gpu_iterator& it) const {
        return static_cast<std::ptrdiff_t>(it.idx) - static_cast<std::ptrdiff_t>(this->idx);
    }

    bool equal(const multi_gpu_iterator& it) const {
        return this->idx == it.idx;
    }

    __host__ __device__ T dereference() const {
        size_t id = idx / gpu_size;
        size_t offset = idx % gpu_size;
        #ifdef __CUDA_ARCH__
        return ptrs[id][offset];
        #else
        T el;
        cudaSetDevice(id);
        cudaMemcpy(&el, ptrs[id] + offset, sizeof(T), cudaMemcpyDeviceToHost);
        return el;
        #endif
    }

    __host__ __device__ multi_gpu_iterator& operator=(const T& val) {
        size_t id = idx / gpu_size;
        size_t offset = idx % gpu_size;
        #ifdef __CUDA_ARCH__
        ptrs[id][offset] = val;
        #else
        cudaSetDevice(id);
        cudaMemcpy(ptrs[id] + offset, &val, sizeof(T), cudaMemcpyHostToDevice);
        #endif
        return *this;
    }

public:
    multi_gpu_iterator(T* const* gpu_ptrs, size_t idx, size_t gpu_size, size_t size) 
        : idx(idx), gpu_size(gpu_size), total_size(size) {
            for(size_t i = 0; i < GPU_COUNT; i++)
                ptrs[i] = gpu_ptrs[i];
        }
        
    T* get_gpu_ptr(size_t id) const {
        return ptrs[id];
    }
    
    size_t get_idx() const {
        return idx;
    }

    size_t get_size() const {
        return total_size;
    }
};
