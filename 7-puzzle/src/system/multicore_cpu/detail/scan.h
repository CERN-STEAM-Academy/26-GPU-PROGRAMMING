/*
 * Based on thrust::exclusive_scan
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/scan.h
 */
#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace thrust::system::multicore_cpu {
	template <typename InputIterator, typename OutputIterator>
	OutputIterator exclusive_scan(multicore_cpu, InputIterator first, InputIterator last, OutputIterator result)
	{
		return thrust::exclusive_scan(thrust::omp::par, first, last, result);
	}

	template <typename InputIterator, typename OutputIterator, typename T>
	OutputIterator exclusive_scan(multicore_cpu, InputIterator first, InputIterator last, OutputIterator result, T init)
	{
		return thrust::exclusive_scan(thrust::omp::par, first, last, result, init);
	}

	template <typename InputIterator, typename OutputIterator, typename T, typename AssociativeOperator>
	OutputIterator exclusive_scan(multicore_cpu, InputIterator first, InputIterator last, OutputIterator result, T init, AssociativeOperator binary_op)
	{
		return thrust::exclusive_scan(thrust::omp::par, first, last, result, init, binary_op);
	}
} // namespace thrust::system::multicore_cpu
