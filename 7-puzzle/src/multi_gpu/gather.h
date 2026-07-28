#pragma once
#include <thrust/execution_policy.h>
#include <thrust/gather.h>

template <typename InputIterator1, typename InputIterator2, typename RandomAccessIterator, typename OutputIterator>
OutputIterator gather_if(multi_gpu_tag, InputIterator1 map_first, InputIterator1 map_last, InputIterator2 stencil, RandomAccessIterator input_first, OutputIterator result)
{
    printf("gather_if\n");
    assert(false);
    auto res = thrust::gather_if(thrust::omp::par, map_first, map_last, stencil, input_first, result);
    return res;
}
