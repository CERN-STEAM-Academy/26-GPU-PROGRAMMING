/*
 * Based on thrust::sort
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/sort.h
 */
#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace thrust::system::multicore_cpu {
	template <typename RandomAccessIterator>
	void sort(multicore_cpu, RandomAccessIterator first, RandomAccessIterator last)
	{
		thrust::sort(thrust::omp::par, first, last);
	}

	template <typename RandomAccessIterator, typename StrictWeakOrdering>
	void sort(multicore_cpu, RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
	{
		thrust::sort(thrust::omp::par, first, last, comp);
	}

	template <typename RandomAccessIterator>
	void stable_sort(multicore_cpu, RandomAccessIterator first, RandomAccessIterator last)
	{
		thrust::stable_sort(thrust::omp::par, first, last);
	}

	template <typename RandomAccessIterator, typename StrictWeakOrdering>
	void stable_sort(multicore_cpu, RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
	{
		thrust::stable_sort(thrust::omp::par, first, last, comp);
	}

	template <typename RandomAccessIterator1, typename RandomAccessIterator2>
	void sort_by_key(multicore_cpu, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
	{
		thrust::sort_by_key(thrust::omp::par, keys_first, keys_last, values_first);
	}

	template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
	void sort_by_key(multicore_cpu, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, StrictWeakOrdering comp)
	{
		thrust::sort_by_key(thrust::omp::par, keys_first, keys_last, values_first, comp);
	}

	template <typename RandomAccessIterator1, typename RandomAccessIterator2>
	void stable_sort_by_key(multicore_cpu, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
	{
		thrust::stable_sort_by_key(thrust::omp::par, keys_first, keys_last, values_first);
	}

	template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
	void stable_sort_by_key(multicore_cpu, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, StrictWeakOrdering comp)
	{
		thrust::stable_sort_by_key(thrust::omp::par, keys_first, keys_last, values_first, comp);
	}

	template <typename ForwardIterator>
	bool is_sorted(multicore_cpu, ForwardIterator first, ForwardIterator last)
	{
		return thrust::is_sorted(thrust::omp::par, first, last);
	}

	template <typename ForwardIterator, typename Compare>
	bool is_sorted(multicore_cpu, ForwardIterator first, ForwardIterator last, Compare comp)
	{
		return thrust::is_sorted(thrust::omp::par, first, last, comp);
	}
} // namespace thrust::system::multicore_cpu
