/*
 * Based on thrust::reverse
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/reverse.h
 */
#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/reverse.h>

namespace thrust::system::multicore_cpu {
	template <typename BidirectionalIterator>
	void reverse(multicore_cpu, BidirectionalIterator first, BidirectionalIterator last)
	{
		thrust::reverse(thrust::omp::par, first, last);
	}
} // namespace thrust::system::multicore_cpu
