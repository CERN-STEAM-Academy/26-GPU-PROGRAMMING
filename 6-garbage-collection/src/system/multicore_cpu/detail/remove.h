/*
 * Based on thrust::remove_if
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/remove.h
 */
#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

namespace thrust::system::multicore_cpu {
	template <typename ForwardIterator, typename Predicate>
	ForwardIterator remove_if(multicore_cpu, ForwardIterator first, ForwardIterator last, Predicate pred)
	{
		return thrust::remove_if(thrust::omp::par, first, last, pred);
	}

	template <typename ForwardIterator, typename InputIterator, typename Predicate>
	ForwardIterator remove_if(multicore_cpu, ForwardIterator first, ForwardIterator last, InputIterator stencil, Predicate pred)
	{
		return thrust::remove_if(thrust::omp::par, first, last, stencil, pred);
	}
} // namespace thrust::system::multicore_cpu
