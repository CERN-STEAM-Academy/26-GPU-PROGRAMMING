/*
 * Based on thrust::for_each
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/for_each.h
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <chrono>

namespace thrust::system::benchmark {
	template <typename Policy, typename InputIterator, typename UnaryFunction>
	InputIterator for_each(benchmark<Policy> exec, InputIterator first, InputIterator last, UnaryFunction f)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		auto result = thrust::for_each(policy, first, last, f);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[for_each] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}
} // namespace thrust::system::benchmark
