/*
 * Based on thrust::gather_if
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/gather.h
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <chrono>

namespace thrust::system::benchmark {
	template <typename Policy, typename InputIterator1, typename InputIterator2, typename RandomAccessIterator, typename OutputIterator>
	OutputIterator gather_if(benchmark<Policy> exec, InputIterator1 map_first, InputIterator1 map_last, InputIterator2 stencil, RandomAccessIterator input_first,
							 OutputIterator result)
	{
		auto size = std::distance(map_first, map_last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		result = thrust::gather_if(policy, map_first, map_last, stencil, input_first, result);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[gather_if] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}
} // namespace thrust::system::benchmark
