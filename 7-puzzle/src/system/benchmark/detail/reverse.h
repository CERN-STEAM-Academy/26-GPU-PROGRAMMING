/*
 * Based on thrust::reverse
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/reverse.h
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/reverse.h>
#include <chrono>

namespace thrust::system::benchmark {
	template <typename Policy, typename BidirectionalIterator>
	void reverse(benchmark<Policy> exec, BidirectionalIterator first, BidirectionalIterator last)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		thrust::reverse(policy, first, last);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[reverse] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);
	}
} // namespace thrust::system::benchmark
