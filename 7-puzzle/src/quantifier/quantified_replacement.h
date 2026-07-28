#ifndef SRO_QUANTIFIED_REPLACEMENT
#define SRO_QUANTIFIED_REPLACEMENT

#include "../layer/layer.h"

namespace sro {
	struct quantified_replacement {
		uint32_t origin;
		node_product replacement;
	};

	struct quantified_replacement_origin_order {
		__host__ __device__
		bool operator()(const quantified_replacement& lhs, const quantified_replacement& rhs) const {
			return lhs.origin < rhs.origin;
		}
	};
}

#endif
