/*
 * Based on thrust::merge
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/merge.h
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <chrono>

namespace thrust::system::benchmark {
	template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
	OutputIterator merge(benchmark<Policy> exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result)
	{
		auto size = std::distance(first1, last1) + std::distance(first2, last2);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		result = thrust::merge(policy, first1, last1, first2, last2, result);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[merge] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}

	template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename StrictWeakCompare>
	OutputIterator merge(benchmark<Policy> exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result,
						 StrictWeakCompare comp)
	{
		auto size = std::distance(first1, last1) + std::distance(first2, last2);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		result = thrust::merge(policy, first1, last1, first2, last2, result, comp);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[merge] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}
} // namespace thrust::system::benchmark
