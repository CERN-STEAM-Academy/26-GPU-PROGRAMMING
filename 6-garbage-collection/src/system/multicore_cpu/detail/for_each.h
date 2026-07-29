/*
 * Based on thrust::for_each
 *
 * https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/for_each.h
 */
#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

namespace thrust::system::multicore_cpu {
	template <typename InputIterator, typename UnaryFunction>
	InputIterator for_each(multicore_cpu, InputIterator first, InputIterator last, UnaryFunction f)
	{
		return thrust::for_each(thrust::tbb::par, first, last, f);
	}
} // namespace thrust::system::multicore_cpu
