#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "timer.h"

namespace thrust {
	namespace system {
		namespace measured_par {
			struct measured_par : thrust::host_execution_policy<measured_par> {};

			template <typename RandomAccessIterator, typename StrictWeakOrdering>
			void sort(measured_par, RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
			{
				CHRONO_VOID_WRAPPER("par", thrust::sort(thrust::omp::par, first, last, comp))
			}

			template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
			void sort_by_key(measured_par, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, StrictWeakOrdering comp)
			{
				CHRONO_VOID_WRAPPER("par", thrust::sort_by_key(thrust::omp::par, keys_first, keys_last, values_first, comp))
			}

			template <typename DerivedPolicy>
			struct host_execution_policy : thrust::system::omp::execution_policy<DerivedPolicy> {};
		} // namespace measured_par

		namespace measured_device {
			struct measured_device : thrust::device_execution_policy<measured_device> {};

			template <typename RandomAccessIterator, typename StrictWeakOrdering>
			void sort(measured_device, RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
			{
				CUDA_VOID_WRAPPER("device", thrust::sort(thrust::device, first, last, comp))
			}

			template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
			void sort_by_key(measured_device, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, StrictWeakOrdering comp)
			{
				CUDA_VOID_WRAPPER("device", thrust::sort_by_key(thrust::device, keys_first, keys_last, values_first, comp))
			}

			template <typename DerivedPolicy>
			struct device_execution_policy : thrust::device_execution_policy<DerivedPolicy> {};
		} // namespace measured_device

		namespace measured_host {
			struct measured_host : thrust::host_execution_policy<measured_host> {};

			template <typename RandomAccessIterator, typename StrictWeakOrdering>
			void sort(measured_host, RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
			{
				CHRONO_VOID_WRAPPER("host", thrust::sort(thrust::host, first, last, comp))
			}

			template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
			void sort_by_key(measured_host, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, StrictWeakOrdering comp)
			{
				CHRONO_VOID_WRAPPER("host", thrust::sort_by_key(thrust::host, keys_first, keys_last, values_first, comp))
			}

			template <typename DerivedPolicy>
			struct host_execution_policy : thrust::host_execution_policy<DerivedPolicy> {};
		} // namespace measured_host

	} // namespace system
	static const system::measured_par::measured_par measured_par;
	static const system::measured_device::measured_device measured_device;
	static const system::measured_host::measured_host measured_host;
} // namespace thrust
