#pragma once
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

template <typename ForwardIterator, typename Predicate>
ForwardIterator remove_if(multi_gpu_tag, ForwardIterator first, ForwardIterator last, Predicate pred)
{
    printf("remove_if2\n");
    assert(false);
    auto result = thrust::remove_if(thrust::omp::par, first, last, pred);
    return result;
}

template <typename ForwardIterator, typename InputIterator, typename Predicate>
ForwardIterator remove_if(multi_gpu_tag, ForwardIterator first, ForwardIterator last, InputIterator stencil, Predicate pred)
{
    printf("remove_if2\n");
    assert(false);
    auto result = thrust::remove_if(thrust::omp::par, first, last, stencil, pred);
    return result;
}