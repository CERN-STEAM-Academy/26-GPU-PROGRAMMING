#pragma once
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

template <typename InputIterator, typename OutputIterator, typename UnaryFunction>
OutputIterator transform(multi_gpu_tag, InputIterator first, InputIterator last, OutputIterator result, UnaryFunction op)
{
    printf("transform1\n");
    assert(false);
    return result;
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
OutputIterator transform(multi_gpu_tag, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, BinaryFunction op)
{
    printf("transform2\n");
    assert(false);
    return result;
}