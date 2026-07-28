#include "administrators.h"
#include <stdexcept>
#include <cassert>

namespace sro{
	data_info::data_info(unsigned int _size, data_location _location) : size(_size), location(_location) {}

	data_location data_administrator::get_data_location(data_info info) {
		throw std::runtime_error("Data location requested on base class.");
	}

	data_policy data_administrator::get_data_policy(data_info info) {
		throw std::runtime_error("Data policy requested on base class.");
	}
	
	bool data_administrator::is_default_administrator() {
		throw std::runtime_error("Default administrator check on base class.");
	}

	data_location host_administrator::get_data_location(data_info info) {
		return HOST;
	}

	data_policy host_administrator::get_data_policy(data_info info) {
		assert(info.location == HOST);

		if (info.size < switch_to_multi_cpu_at) {
			return SINGLE_CORE_CPU;
		}
		else {
			return MULTI_CORE_CPU;
		}
	}

	bool host_administrator::is_default_administrator() {
		return false;
	}

	data_location thrust_administrator::get_data_location(data_info info) {
		if (info.size < switch_to_gpu_at){
			return HOST;
		}
		else {
			return DEVICE;
		}
	}

	data_policy thrust_administrator::get_data_policy(data_info info) {
		if (info.location == HOST) {
			if (info.size < switch_to_multi_cpu_at) {
				return SINGLE_CORE_CPU;
			}
			else {
				return MULTI_CORE_CPU;
			}
		}
		else {
			return THRUST_GPU;
		}
	}

	bool thrust_administrator::is_default_administrator() {
		return switch_to_multi_cpu_at == 5000 && switch_to_gpu_at == 10000;
	}
}