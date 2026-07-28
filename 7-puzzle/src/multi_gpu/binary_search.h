#pragma once
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator lower_bound(multi_gpu_tag, ForwardIterator first, ForwardIterator last, const LessThanComparable& value)
{
    printf("lower_bound1\n");
    assert(false);
    auto result = thrust::lower_bound(thrust::omp::par, first, last, value);
    return result;
}

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator lower_bound(multi_gpu_tag, ForwardIterator first, ForwardIterator last, const T& value, StrictWeakOrdering comp)
{
    printf("lower_bound2\n");
    assert(false);
    auto result = thrust::lower_bound(thrust::omp::par, first, last, value, comp);
    return result;
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator lower_bound(multi_gpu_tag, ForwardIterator first, ForwardIterator last, InputIterator values_first, InputIterator values_last, OutputIterator result)
{
    printf("lower_bound3\n");
    assert(false);
    auto res = thrust::lower_bound(thrust::omp::par, first, last, values_first, values_last, result);
    return res;
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator lower_bound(multi_gpu_tag, ForwardIterator first, ForwardIterator last, InputIterator values_first, InputIterator values_last, OutputIterator result,
                            StrictWeakOrdering comp)
{
    printf("lower_bound4\n");
    assert(false);
    auto res = thrust::lower_bound(thrust::omp::par, first, last, values_first, values_last, result, comp);
    return res;
}
