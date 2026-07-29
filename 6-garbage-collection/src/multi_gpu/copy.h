#pragma once
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

template <typename InputIterator, typename OutputIterator>
OutputIterator copy(multi_gpu_tag, InputIterator first, InputIterator last, OutputIterator result)
{
    printf("copy\n");
	assert(false);
	auto res = thrust::copy(thrust::omp::par, first, last, result);;
	return res;
}

template <typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator copy_if(multi_gpu_tag, InputIterator first, InputIterator last, OutputIterator result, Predicate pred)
{
    printf("copy_if\n");
    assert(false);
    auto res = thrust::copy_if(thrust::omp::par, first, last, result, pred);
    return res;
}
