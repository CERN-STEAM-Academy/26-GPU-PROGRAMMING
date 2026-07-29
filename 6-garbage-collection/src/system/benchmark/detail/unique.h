/*
 * Based on thrust::unique
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/unique.h
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <chrono>

namespace thrust::system::benchmark {
	template <typename Policy, typename ForwardIterator>
	ForwardIterator unique(benchmark<Policy> exec, ForwardIterator first, ForwardIterator last)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		auto result = thrust::unique(policy, first, last);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[unique] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}

	template <typename Policy, typename ForwardIterator, typename BinaryPredicate>
	ForwardIterator unique(benchmark<Policy> exec, ForwardIterator first, ForwardIterator last, BinaryPredicate binary_pred)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		auto result = thrust::unique(policy, first, last, binary_pred);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[unique] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}
} // namespace thrust::system::benchmark
