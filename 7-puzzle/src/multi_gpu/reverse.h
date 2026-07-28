#pragma once
#include <thrust/execution_policy.h>
#include <thrust/reverse.h>

template <typename BidirectionalIterator>
void reverse(multi_gpu_tag, BidirectionalIterator first, BidirectionalIterator last)
{
    printf("reverse\n");
    assert(false);
    thrust::reverse(thrust::omp::par, first, last);
}