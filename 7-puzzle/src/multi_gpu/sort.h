#pragma once
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

template <typename RandomAccessIterator>
void sort(multi_gpu_tag, RandomAccessIterator first, RandomAccessIterator last)
{
    printf("sort1\n");
    assert(false);
    thrust::sort(thrust::omp::par, first, last);
}

template <typename RandomAccessIterator, typename StrictWeakOrdering>
void sort(multi_gpu_tag, RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
{
    printf("sort2\n");
    assert(false);
    thrust::sort(thrust::omp::par, first, last, comp);
}

template <typename RandomAccessIterator>
void stable_sort(multi_gpu_tag, RandomAccessIterator first, RandomAccessIterator last)
{
    printf("stable_sort1\n");
    assert(false);
    thrust::stable_sort(thrust::omp::par, first, last);
}

template <typename RandomAccessIterator, typename StrictWeakOrdering>
void stable_sort(multi_gpu_tag, RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
{
    printf("stable_sort2\n");
    assert(false);
    thrust::stable_sort(thrust::omp::par, first, last, comp);
}

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void sort_by_key(multi_gpu_tag, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
{
    printf("sort_by_key1\n");
    assert(false);
    thrust::sort_by_key(thrust::omp::par, keys_first, keys_last, values_first);
}

template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
void sort_by_key(multi_gpu_tag, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, StrictWeakOrdering comp)
{
    printf("sort_by_key2\n");
    assert(false);
    thrust::sort_by_key(thrust::omp::par, keys_first, keys_last, values_first, comp);
}

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void stable_sort_by_key(multi_gpu_tag, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
{
    printf("stable_sort_by_key1\n");
    assert(false);
    thrust::stable_sort_by_key(thrust::omp::par, keys_first, keys_last, values_first);
}

template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
void stable_sort_by_key(multi_gpu_tag, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, StrictWeakOrdering comp)
{
    printf("stable_sort_by_key2\n");
    assert(false);
    thrust::stable_sort_by_key(thrust::omp::par, keys_first, keys_last, values_first, comp);
}

template <typename ForwardIterator>
bool is_sorted(multi_gpu_tag, ForwardIterator first, ForwardIterator last)
{
    printf("is_sorted1\n");
    assert(false);
    bool result = thrust::is_sorted(thrust::omp::par, first, last);
    return result;
}

template <typename ForwardIterator, typename Compare>
bool is_sorted(multi_gpu_tag, ForwardIterator first, ForwardIterator last, Compare comp)
{
    printf("is_sorted2\n");
    assert(false);
    bool result = thrust::is_sorted(thrust::omp::par, first, last, comp);
    return result;
}
