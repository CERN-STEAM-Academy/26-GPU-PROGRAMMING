#pragma once
#include <thrust/execution_policy.h>
#include <thrust/merge.h>

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator merge(multi_gpu_tag, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result)
{
    printf("merge1\n");
    auto res = thrust::merge(thrust::omp::par, first1, last1, first2, last2, result);
    return res;
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename StrictWeakCompare>
OutputIterator merge(multi_gpu_tag, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, StrictWeakCompare comp)
{
    printf("merge2\n");
    auto res = thrust::merge(thrust::omp::par, first1, last1, first2, last2, result, comp);
    return res;
}