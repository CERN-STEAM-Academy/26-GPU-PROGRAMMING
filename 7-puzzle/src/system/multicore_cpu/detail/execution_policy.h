#pragma once

#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace thrust {
	namespace system::multicore_cpu {
		// Execution Policy Definition: multicore_cpu
		struct multicore_cpu : thrust::host_execution_policy<multicore_cpu> {};

		// Derived Execution Policy: Host (thrust::omp::par / thrust::tbb)
		template <typename DerivedPolicy>
		struct host_execution_policy : thrust::system::omp::execution_policy<DerivedPolicy> {};
		// struct host_execution_policy : thrust::system::tbb::execution_policy<DerivedPolicy> {};

		// Derived Execution Policy: Device (thrust::device)
		template <typename DerivedPolicy>
		struct device_execution_policy : thrust::device_execution_policy<DerivedPolicy> {};
	} // namespace system::multicore_cpu

	static const system::multicore_cpu::multicore_cpu multicore_cpu;
} // namespace thrust
