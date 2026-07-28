/*
 * Based on thrust::scatter
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/scatter.h
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/scatter.h>
#include <chrono>

namespace thrust::system::benchmark {
	template <typename Policy, typename InputIterator1, typename InputIterator2, typename RandomAccessIterator>
	void scatter(benchmark<Policy> exec, InputIterator1 first, InputIterator1 last, InputIterator2 map, RandomAccessIterator result)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		thrust::scatter(policy, first, last, map, result);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[scatter] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);
	}
} // namespace thrust::system::benchmark
