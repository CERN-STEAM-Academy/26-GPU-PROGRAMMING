#pragma once

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/system/omp/detail/execution_policy.h>
#include <thrust/system/tbb/detail/execution_policy.h>
#include "../../multicore_cpu/execution_policy.h"

namespace thrust {
	namespace system::benchmark {
		// Execution Policy Definition: benchmark
		template <typename Policy>
		struct benchmark : thrust::host_execution_policy<benchmark<Policy>> {
			const char* label;

			benchmark(const char* label) : label(label) {}
		};

		// Derived Execution Policy: Host (thrust::host)
		template <typename DerivedPolicy>
		struct host_execution_policy : thrust::host_execution_policy<DerivedPolicy> {};

		// Derived Execution Policy: Device (thrust::device)
		template <typename DerivedPolicy>
		struct device_execution_policy : thrust::device_execution_policy<DerivedPolicy> {};
	} // namespace system::benchmark

	static const system::benchmark::benchmark<decltype(thrust::omp::par)> benchmark_omp("thrust::omp::par");
	static const system::benchmark::benchmark<decltype(thrust::tbb::par)> benchmark_tbb("thrust::tbb::par");
	static const system::benchmark::benchmark<decltype(thrust::multicore_cpu)> benchmark_custom("thrust::multicore_cpu");
} // namespace thrust
