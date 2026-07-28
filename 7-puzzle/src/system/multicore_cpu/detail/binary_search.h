/*
 * Based on thrust::lower_bound
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/binary_search.h
 */
#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

namespace thrust::system::multicore_cpu {
	template <typename ForwardIterator, typename LessThanComparable>
	ForwardIterator lower_bound(multicore_cpu, ForwardIterator first, ForwardIterator last, const LessThanComparable& value)
	{
		return thrust::lower_bound(thrust::omp::par, first, last, value);
	}

	template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
	ForwardIterator lower_bound(multicore_cpu, ForwardIterator first, ForwardIterator last, const T& value, StrictWeakOrdering comp)
	{
		return thrust::lower_bound(thrust::omp::par, first, last, value, comp);
	}

	template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
	OutputIterator lower_bound(multicore_cpu, ForwardIterator first, ForwardIterator last, InputIterator values_first, InputIterator values_last, OutputIterator result)
	{
		return thrust::lower_bound(thrust::omp::par, first, last, values_first, values_last, result);
	}

	template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
	OutputIterator lower_bound(multicore_cpu, ForwardIterator first, ForwardIterator last, InputIterator values_first, InputIterator values_last, OutputIterator result,
							   StrictWeakOrdering comp)
	{
		return thrust::lower_bound(thrust::omp::par, first, last, values_first, values_last, result, comp);
	}
} // namespace thrust::system::multicore_cpu
