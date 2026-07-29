/*
 * Based on thrust::lower_bound
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/binary_search.h
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <chrono>

namespace thrust::system::benchmark {
	template <typename Policy, typename ForwardIterator, typename LessThanComparable>
	ForwardIterator lower_bound(benchmark<Policy> exec, ForwardIterator first, ForwardIterator last, const LessThanComparable& value)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		auto result = thrust::lower_bound(policy, first, last, value);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[lower_bound] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}

	template <typename Policy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
	ForwardIterator lower_bound(benchmark<Policy> exec, ForwardIterator first, ForwardIterator last, const T& value, StrictWeakOrdering comp)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		auto result = thrust::lower_bound(policy, first, last, value, comp);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[lower_bound] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}

	template <typename Policy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
	OutputIterator lower_bound(benchmark<Policy> exec, ForwardIterator first, ForwardIterator last, InputIterator values_first, InputIterator values_last, OutputIterator result)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		result = thrust::lower_bound(policy, first, last, values_first, values_last, result);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[lower_bound] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}

	template <typename Policy, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
	OutputIterator lower_bound(benchmark<Policy> exec, ForwardIterator first, ForwardIterator last, InputIterator values_first, InputIterator values_last, OutputIterator result,
							   StrictWeakOrdering comp)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		result = thrust::lower_bound(policy, first, last, values_first, values_last, result, comp);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[lower_bound] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}
} // namespace thrust::system::benchmark
