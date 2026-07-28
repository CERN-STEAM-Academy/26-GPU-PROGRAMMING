/*
 * Based on thrust::copy
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/copy.h
 */
#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

namespace thrust::system::multicore_cpu {
	template <typename InputIterator, typename OutputIterator>
	OutputIterator copy(multicore_cpu, InputIterator first, InputIterator last, OutputIterator result)
	{
		return thrust::copy(thrust::tbb::par, first, last, result);
	}

	template <typename InputIterator, typename OutputIterator, typename Predicate>
	OutputIterator copy_if(multicore_cpu, InputIterator first, InputIterator last, OutputIterator result, Predicate pred)
	{
		return thrust::copy_if(thrust::tbb::par, first, last, result, pred);
	}
} // namespace thrust::system::multicore_cpu
