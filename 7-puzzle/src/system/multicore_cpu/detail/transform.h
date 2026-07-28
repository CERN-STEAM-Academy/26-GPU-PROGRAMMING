/*
 * Based on thrust::transform
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/transform.h
 */
#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace thrust::system::multicore_cpu {
	template <typename InputIterator, typename OutputIterator, typename UnaryFunction>
	OutputIterator transform(multicore_cpu, InputIterator first, InputIterator last, OutputIterator result, UnaryFunction op)
	{
		return thrust::transform(thrust::omp::par, first, last, result, op);
	}

	template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
	OutputIterator transform(multicore_cpu, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, BinaryFunction op)
	{
		return thrust::transform(thrust::omp::par, first1, last1, first2, result, op);
	}
} // namespace thrust::system::multicore_cpu
