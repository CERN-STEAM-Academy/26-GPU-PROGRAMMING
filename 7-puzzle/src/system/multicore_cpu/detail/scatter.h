/*
 * Based on thrust::scatter
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/scatter.h
 */
#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/scatter.h>

namespace thrust::system::multicore_cpu {
	template <typename InputIterator1, typename InputIterator2, typename RandomAccessIterator>
	void scatter(multicore_cpu, InputIterator1 first, InputIterator1 last, InputIterator2 map, RandomAccessIterator result)
	{
		thrust::scatter(thrust::omp::par, first, last, map, result);
	}
} // namespace thrust::system::multicore_cpu
