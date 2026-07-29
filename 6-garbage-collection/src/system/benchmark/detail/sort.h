/*
 * Based on thrust::sort
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/sort.h
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <chrono>

namespace thrust::system::benchmark {
	template <typename Policy, typename RandomAccessIterator>
	void sort(benchmark<Policy> exec, RandomAccessIterator first, RandomAccessIterator last)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		thrust::sort(policy, first, last);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[sort] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);
	}

	template <typename Policy, typename RandomAccessIterator, typename StrictWeakOrdering>
	void sort(benchmark<Policy> exec, RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		thrust::sort(policy, first, last, comp);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[sort] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);
	}

	template <typename Policy, typename RandomAccessIterator>
	void stable_sort(benchmark<Policy> exec, RandomAccessIterator first, RandomAccessIterator last)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		thrust::stable_sort(policy, first, last);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[stable_sort] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);
	}

	template <typename Policy, typename RandomAccessIterator, typename StrictWeakOrdering>
	void stable_sort(benchmark<Policy> exec, RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		thrust::stable_sort(policy, first, last, comp);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[stable_sort] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);
	}

	template <typename Policy, typename RandomAccessIterator1, typename RandomAccessIterator2>
	void sort_by_key(benchmark<Policy> exec, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
	{
		auto size = std::distance(keys_first, keys_last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		thrust::sort_by_key(policy, keys_first, keys_last, values_first);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[sort_by_key] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);
	}

	template <typename Policy, typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
	void sort_by_key(benchmark<Policy> exec, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, StrictWeakOrdering comp)
	{
		auto size = std::distance(keys_first, keys_last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		thrust::sort_by_key(policy, keys_first, keys_last, values_first, comp);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[sort_by_key] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);
	}

	template <typename Policy, typename RandomAccessIterator1, typename RandomAccessIterator2>
	void stable_sort_by_key(benchmark<Policy> exec, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
	{
		auto size = std::distance(keys_first, keys_last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		thrust::stable_sort_by_key(policy, keys_first, keys_last, values_first);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[stable_sort_by_key] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);
	}

	template <typename Policy, typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
	void stable_sort_by_key(benchmark<Policy> exec, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, StrictWeakOrdering comp)
	{
		auto size = std::distance(keys_first, keys_last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		thrust::stable_sort_by_key(policy, keys_first, keys_last, values_first, comp);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[stable_sort_by_key] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);
	}

	template <typename Policy, typename ForwardIterator>
	bool is_sorted(benchmark<Policy> exec, ForwardIterator first, ForwardIterator last)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		bool result = thrust::is_sorted(policy, first, last);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[is_sorted] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}

	template <typename Policy, typename ForwardIterator, typename Compare>
	bool is_sorted(benchmark<Policy> exec, ForwardIterator first, ForwardIterator last, Compare comp)
	{
		auto size = std::distance(first, last);
		auto start = std::chrono::system_clock::now();

		Policy policy;
		bool result = thrust::is_sorted(policy, first, last, comp);

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		printf("[is_sorted] Time Elapsed: %10ldns, Size: %10d, Policy: %s\n", elapsed.count(), size, exec.label);

		return result;
	}
} // namespace thrust::system::benchmark
