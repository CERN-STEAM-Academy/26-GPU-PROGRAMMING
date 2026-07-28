#pragma once
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan(multi_gpu_tag, InputIterator first, InputIterator last, OutputIterator result)
{
    printf("exclusive_scan1\n");
    assert(false);
    auto res = thrust::exclusive_scan(thrust::omp::par, first, last, result);
    return res;
}

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator exclusive_scan(multi_gpu_tag, InputIterator first, InputIterator last, OutputIterator result, T init)
{
    printf("exclusive_scan2\n");
    assert(false);
    auto res = thrust::exclusive_scan(thrust::omp::par, first, last, result, init);
    return res;
}

template <typename InputIterator, typename OutputIterator, typename T, typename AssociativeOperator>
OutputIterator exclusive_scan(multi_gpu_tag, InputIterator first, InputIterator last, OutputIterator result, T init, AssociativeOperator binary_op)
{
    printf("exclusive_scan3\n");
    assert(false);
    auto res = thrust::exclusive_scan(thrust::omp::par, first, last, result, init, binary_op);
    return res;
}