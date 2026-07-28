#pragma once
#include <thrust/execution_policy.h>
#include <thrust/scatter.h>

template <typename InputIterator1, typename InputIterator2, typename RandomAccessIterator>
void scatter(multi_gpu_tag, InputIterator1 first, InputIterator1 last, InputIterator2 map, RandomAccessIterator result)
{
    printf("scatter\n");
    assert(false);
    thrust::scatter(thrust::omp::par, first, last, map, result);
}
