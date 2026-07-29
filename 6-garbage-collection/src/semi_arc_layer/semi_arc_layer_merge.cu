// Sorting functions for semi arc layer

#include "semi_arc_layer.h"

#include <thrust/set_operations.h>

using namespace sro;

struct semi_arc_to_b{
	__host__ __device__
	node_product operator() (semi_arc in) const {
		return in.b;
	}
};

struct sort_node_product_high_low{
	__host__ __device__
	bool operator() (node_product a, node_product b) const {
		return thrust::make_tuple(a.high, a.low) < thrust::make_tuple(b.high, b.low);
	}
};

data_manager<node_product> semi_arc_layer::merge_semi_arcs_with_layer(layer& l) {
	assert(top_heavy);

	// Make sure the semi arcs are sorted on children, as we will do a set union on the children.
	// TODO: examine why it is not already sorted.
	sort_on_children();

	// Swap children from layer
	data_manager<node_product> children(get_data_location(), l.node_count() + semi_arcs.size(), administrator);
	l.children.swap(children);

	children.with_vector([&](auto policy, auto& result_v, auto& semi_arcs_v, auto& l_v){
		// Transform iterator from semi-arcs to node products
		// Precondition is that the semi-arcs are sorted on b.high
		auto new_products = thrust::make_transform_iterator(semi_arcs_v.begin(), semi_arc_to_b());
		
		// Perform merge
		auto iterator = thrust::set_union(
			policy, 
			result_v.begin(), result_v.end(),
			new_products, new_products + semi_arcs.size(),
			l_v.begin(),
			sort_node_product_high_low()
		);
		l_v.resize(iterator - l_v.begin());
	}, semi_arcs, l.children);

	// This line can probably be deleted.
	l.children.sort(sort_node_product_high_low());
	l.children.unique();

	return children;
}