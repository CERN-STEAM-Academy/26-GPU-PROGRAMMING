/*
 * Based on thrust::merge
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/merge.h
 */
#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>

namespace thrust::system::multicore_cpu {
	template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
	OutputIterator merge(multicore_cpu, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result)
	{
		return thrust::merge(thrust::omp::par, first1, last1, first2, last2, result);
	}

	template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename StrictWeakCompare>
	OutputIterator merge(multicore_cpu, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, StrictWeakCompare comp)
	{
		return thrust::merge(thrust::omp::par, first1, last1, first2, last2, result, comp);
	}
} // namespace thrust::system::multicore_cpu
