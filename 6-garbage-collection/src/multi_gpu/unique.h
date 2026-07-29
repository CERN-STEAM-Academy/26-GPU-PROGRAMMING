#pragma once
#include <thrust/execution_policy.h>
#include <thrust/unique.h>

template <typename ForwardIterator>
ForwardIterator unique(multi_gpu_tag, ForwardIterator first, ForwardIterator last)
{
    printf("unique1\n");
    auto result = thrust::unique(thrust::omp::par, first, last);
    return result;
}

template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator unique(multi_gpu_tag, ForwardIterator first, ForwardIterator last, BinaryPredicate binary_pred)
{
    printf("unique2\n");
    auto result = thrust::unique(thrust::omp::par, first, last, binary_pred);
    return result;
}