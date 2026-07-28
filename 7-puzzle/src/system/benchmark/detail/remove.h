/*
 * Based on thrust::remove_if
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/remove.h
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <chrono>

namespace thrust::system::benchmark {
	template <typename Policy, typename ForwardIterator, typename Predicate>
	ForwardIterator remove_if(benchmark<Policy> exec, ForwardIterator first, ForwardIterator last, Predicate pred)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		auto result = thrust::remove_if(policy, first, last, pred);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[remove_if] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}

	template <typename Policy, typename ForwardIterator, typename InputIterator, typename Predicate>
	ForwardIterator remove_if(benchmark<Policy> exec, ForwardIterator first, ForwardIterator last, InputIterator stencil, Predicate pred)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		auto result = thrust::remove_if(policy, first, last, stencil, pred);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[remove_if] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}
} // namespace thrust::system::benchmark
