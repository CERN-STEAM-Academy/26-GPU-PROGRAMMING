/*
 * Based on thrust::transform
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/transform.h
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <chrono>

namespace thrust::system::benchmark {
	template <typename Policy, typename InputIterator, typename OutputIterator, typename UnaryFunction>
	OutputIterator transform(benchmark<Policy> exec, InputIterator first, InputIterator last, OutputIterator result, UnaryFunction op)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		result = thrust::transform(policy, first, last, result, op);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[transform] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}

	template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
	OutputIterator transform(benchmark<Policy> exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, BinaryFunction op)
	{
		auto size = std::distance(first1, last1);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		result = thrust::transform(policy, first1, last1, first2, result, op);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[transform] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}
} // namespace thrust::system::benchmark
