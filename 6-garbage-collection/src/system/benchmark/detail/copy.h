/*
 * Based on thrust::copy
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/copy.h
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <chrono>

namespace thrust::system::benchmark {
	template <typename Policy, typename InputIterator, typename OutputIterator>
	OutputIterator copy(benchmark<Policy> exec, InputIterator first, InputIterator last, OutputIterator result)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		result = thrust::copy(policy, first, last, result);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[copy] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}

	template <typename Policy, typename InputIterator, typename OutputIterator, typename Predicate>
	OutputIterator copy_if(benchmark<Policy> exec, InputIterator first, InputIterator last, OutputIterator result, Predicate pred)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		result = thrust::copy_if(policy, first, last, result, pred);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[copy_if] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}
} // namespace thrust::system::benchmark
