#pragma once
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

template <typename InputIterator, typename UnaryFunction>
InputIterator for_each(multi_gpu_tag, InputIterator first, InputIterator last, UnaryFunction f)
{
    printf("for_each\n");
    assert(false);
    return first;
}
