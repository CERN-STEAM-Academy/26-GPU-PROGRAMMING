/*
 * Based on thrust::exclusive_scan
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/scan.h
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <chrono>

namespace thrust::system::benchmark {
	template <typename Policy, typename InputIterator, typename OutputIterator>
	OutputIterator exclusive_scan(benchmark<Policy> exec, InputIterator first, InputIterator last, OutputIterator result)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		result = thrust::exclusive_scan(policy, first, last, result);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[exclusive_scan] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}

	template <typename Policy, typename InputIterator, typename OutputIterator, typename T>
	OutputIterator exclusive_scan(benchmark<Policy> exec, InputIterator first, InputIterator last, OutputIterator result, T init)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		result = thrust::exclusive_scan(policy, first, last, result, init);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[exclusive_scan] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}

	template <typename Policy, typename InputIterator, typename OutputIterator, typename T, typename AssociativeOperator>
	OutputIterator exclusive_scan(benchmark<Policy> exec, InputIterator first, InputIterator last, OutputIterator result, T init, AssociativeOperator binary_op)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		result = thrust::exclusive_scan(policy, first, last, result, init, binary_op);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[exclusive_scan] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}
} // namespace thrust::system::benchmark
