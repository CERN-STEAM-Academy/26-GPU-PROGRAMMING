#ifndef SRO_PERSISTENT_QUANTIFIER_STATE
#define SRO_PERSISTENT_QUANTIFIER_STATE

#include <cstddef>
#include <cstdint>
#include <vector>
#include <thrust/tuple.h>

namespace sro {
	struct persistent_quantifier_statistics {
		std::vector<uint32_t> states_per_layer;
		uint32_t peak_state_count = 0;
		uint32_t peak_state_layer = 0;
		size_t peak_layer_member_count = 0;
		uint32_t peak_layer_member_count_layer = 0;
	};

	struct persistent_state_member {
		uint32_t state;
		uint32_t node;

		__host__ __device__
		bool operator==(const persistent_state_member& other) const {
			return state == other.state && node == other.node;
		}
	};

	struct persistent_state_arc {
		uint32_t low;
		uint32_t high;
	};

	struct persistent_state_member_order {
		__host__ __device__
		bool operator()(const persistent_state_member& lhs, const persistent_state_member& rhs) const {
			return thrust::make_tuple(lhs.state, lhs.node) < thrust::make_tuple(rhs.state, rhs.node);
		}
	};

}

#endif
