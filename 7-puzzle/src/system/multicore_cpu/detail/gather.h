/*
 * Based on thrust::gather_if
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/gather.h
 */
#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>

namespace thrust::system::multicore_cpu {
	template <typename InputIterator1, typename InputIterator2, typename RandomAccessIterator, typename OutputIterator>
	OutputIterator gather_if(multicore_cpu, InputIterator1 map_first, InputIterator1 map_last, InputIterator2 stencil, RandomAccessIterator input_first, OutputIterator result)
	{
		return thrust::gather_if(thrust::omp::par, map_first, map_last, stencil, input_first, result);
	}
} // namespace thrust::system::multicore_cpu
