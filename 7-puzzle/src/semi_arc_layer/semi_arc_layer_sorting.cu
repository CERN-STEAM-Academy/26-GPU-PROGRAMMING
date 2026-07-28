// Sorting functions for semi arc layer

#include "semi_arc_layer.h"

using namespace sro;

struct sort_on_semi_arc_children{
	__host__ __device__
	bool operator() (const semi_arc& a, const semi_arc& b) const {
		auto a_t = thrust::make_tuple(a.b.high, a.b.low, a.a.high, a.a.low);
		auto b_t = thrust::make_tuple(b.b.high, b.b.low, b.a.high, b.a.low);
		return a_t < b_t;
	}
};

void semi_arc_layer::sort_on_children() {
	semi_arcs.sort(sort_on_semi_arc_children());
}

template<typename List>
void semi_arc_layer::sort_on_children(List values) {
	semi_arcs.sort_by_key(values, sort_on_semi_arc_children());
}

void semi_arc_layer::sort_on_parents() {
	semi_arcs.sort(sort_on_semi_arc_parents());
}