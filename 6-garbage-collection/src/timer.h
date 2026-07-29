#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

 // +/- 0.5 micro sec deviation -> +/- 500 ns
#define CUDA_VOID_WRAPPER(ORG, func) \
    cudaStream_t stream; \
    cudaEvent_t start, stop; \
    cudaStreamCreate(&stream); \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start, stream); \
    func; \
    cudaEventRecord(stop, stream); \
    cudaStreamSynchronize(stream); \
    float time = 0.0f; \
    cudaEventElapsedTime(&time, start, stop); \
    std::cout << "@" << ORG << "|" << #func << "|time->" << time * 1000000 << " ns\n"; \
    cudaEventDestroy(start); \
    cudaEventDestroy(stop); \
    cudaStreamDestroy(stream);

// unknown deviation!
#define CHRONO_VOID_WRAPPER(ORG, func) \
	const auto start = std::chrono::high_resolution_clock::now(); \
    func; \
	const auto stop = std::chrono::high_resolution_clock::now(); \
	const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count(); \
    std::cout << "@" << ORG << "|" << #func << "|time->" << time << " ns\n";
