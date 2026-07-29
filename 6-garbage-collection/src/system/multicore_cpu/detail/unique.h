/*
 * Based on thrust::unique
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/unique.h
 */
#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>

namespace thrust::system::multicore_cpu {
	template <typename ForwardIterator>
	ForwardIterator unique(multicore_cpu, ForwardIterator first, ForwardIterator last)
	{
		return thrust::unique(thrust::tbb::par, first, last);
	}

	template <typename ForwardIterator, typename BinaryPredicate>
	ForwardIterator unique(multicore_cpu, ForwardIterator first, ForwardIterator last, BinaryPredicate binary_pred)
	{
		return thrust::unique(thrust::tbb::par, first, last, binary_pred);
	}
} // namespace thrust::system::multicore_cpu
