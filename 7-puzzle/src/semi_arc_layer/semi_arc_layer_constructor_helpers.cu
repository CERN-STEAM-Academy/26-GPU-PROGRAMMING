// Helper functions for constructors of semi arc layer

#include "semi_arc_layer.h"

using namespace sro;

struct sort_node_product_high_low{
	__host__ __device__
	bool operator() (node_product a, node_product b) const {
		return thrust::make_tuple(a.high, a.low) < thrust::make_tuple(b.high, b.low);
	}
};

struct sort_arc_low{
	__host__ __device__
	bool operator() (arc a, arc b) {
		auto a_t = thrust::make_tuple(a.low.high, a.low.low, a.parent.high, a.parent.low, a.high.high, a.high.low);
		auto b_t = thrust::make_tuple(b.low.high, b.low.low, b.parent.high, b.parent.low, b.high.high, b.high.low);
		return a_t < b_t;
	}
};

struct sort_arc_high{
	__host__ __device__
	bool operator() (const arc a, const arc b) const {
		auto a_t = thrust::make_tuple(a.high.high, a.high.low, a.parent.high, a.parent.low, a.low.high, a.low.low);
		auto b_t = thrust::make_tuple(b.high.high, b.high.low, b.parent.high, b.parent.low, b.low.high, b.low.low);
		return a_t < b_t;
	}
};

struct sort_product{
	__host__ __device__
	bool operator() (node_product a, node_product b) {
		return thrust::make_tuple(a.high, a.low) < thrust::make_tuple(b.high, b.low);
	}
};

struct semi_arc_to_a{
	__host__ __device__
	node_product operator() (semi_arc in) const {
		return in.a;
	}
};

struct semi_arc_to_b{
	__host__ __device__
	node_product operator() (semi_arc in) const {
		return in.b;
	}
};

struct arc_to_low{
	__host__ __device__
	node_product operator() (arc in) const {
		return in.low;
	}
};

struct arc_to_high{
	__host__ __device__
	node_product operator() (arc in) const {
		return in.high;
	}
};

struct transform_arc_low{
	__host__ __device__
	arc operator() (arc a, uint32_t idx){
		a.low.low = idx;
		a.low.high = 0;
		return a; 
	}
};

struct transform_arc_high{
	__host__ __device__
	semi_arc operator() (arc a, uint32_t idx){
		return {
			a.parent.low,
			a.parent.high,
			a.low.low,
			idx
		};
	}
};

data_manager<uint32_t> semi_arc_layer::fill_semi_arcs(arc_layer& parent_arcs, semi_arc_layer& child_arcs, layer& merge_with) {
	child_arcs.sort_on_parents();

	// Compute lower bound of semi arcs to merged children
	data_manager<uint32_t> semi_arc_indices_to_layer(get_data_location(), child_arcs.semi_arcs.size(), administrator);

	child_arcs.semi_arcs.with_vector([&](auto policy,  auto& semi_arcs_v, auto& indices_to_layer_v, auto& merge_with_v){
		auto new_products = thrust::make_transform_iterator(semi_arcs_v.begin(), semi_arc_to_b());

		using VecSemiArc = std::remove_reference_t<decltype(semi_arcs_v)>;
		static_assert(std::is_same_v<vector_value_type_t<VecSemiArc>, sro::semi_arc>, "Types are the same.");

		using VecUint = std::remove_reference_t<decltype(merge_with_v)>;
		static_assert(std::is_same_v<vector_value_type_t<VecUint>, sro::node_product>, "Types are the same.");

		using VecProduct = std::remove_reference_t<decltype(indices_to_layer_v)>;
		static_assert(std::is_same_v<vector_value_type_t<VecProduct>, uint32_t>, "Types are the same.");

		thrust::lower_bound(
			policy,
			merge_with_v.begin(), merge_with_v.end(),
			new_products, new_products + semi_arcs_v.size(),
			indices_to_layer_v.begin(),
			sort_node_product_high_low()
		);
	}, semi_arc_indices_to_layer, merge_with.children);

	// Sort semi-arcs on parents, with semi_arc_indices_to_layer as payload
	child_arcs.sort_on_parents(semi_arc_indices_to_layer);

	// Fill low arcs
	parent_arcs.arcs.sort(sort_arc_low());

	assert(semi_arc_indices_to_layer.size() > 0);
	child_arcs.semi_arcs.with_vector([&](auto policy,  auto& semi_arcs_v, auto& indices_to_layer_v, auto& parent_arcs_v){
		using Vec = std::remove_reference_t<decltype(semi_arcs_v)>;
		using IndicesVec = rebind_vector_t<Vec, uint32_t>;

		IndicesVec parent_arc_low_indices_to_semi_arcs(parent_arcs_v.size());
		auto parent_arcs_low = thrust::make_transform_iterator(parent_arcs_v.begin(), arc_to_low());
		auto child_arcs_parent = thrust::make_transform_iterator(semi_arcs_v.begin(), semi_arc_to_a());
		thrust::lower_bound(
			policy,
			child_arcs_parent, child_arcs_parent + semi_arcs_v.size(),
			parent_arcs_low, parent_arcs_low + parent_arcs_v.size(),
			parent_arc_low_indices_to_semi_arcs.begin(),
			sort_product()
		);
	
		auto permuted_low = thrust::make_permutation_iterator(indices_to_layer_v.begin(), parent_arc_low_indices_to_semi_arcs.begin());
		assert(parent_arc_low_indices_to_semi_arcs.size() == parent_arcs_v.size());
		thrust::transform(
			policy,
			parent_arcs_v.begin(), parent_arcs_v.end(),
			permuted_low,
			parent_arcs_v.begin(),
			transform_arc_low()
		);
	}, semi_arc_indices_to_layer, parent_arcs.arcs);

	// Fill high arcs
	parent_arcs.arcs.sort(sort_arc_high());
	

	child_arcs.semi_arcs.with_vector([&](auto policy,  auto& semi_arcs_v, auto& indices_to_layer_v, auto& parent_arcs_v, auto& target_semi_arcs_v){
		using Vec = std::remove_reference_t<decltype(semi_arcs_v)>;
		using IndicesVec = rebind_vector_t<Vec, uint32_t>;

		IndicesVec parent_arc_high_indices_to_semi_arcs(parent_arcs_v.size());
		auto parent_arcs_high = thrust::make_transform_iterator(parent_arcs_v.begin(), arc_to_high());
		auto child_arcs_parent2 = thrust::make_transform_iterator(semi_arcs_v.begin(), semi_arc_to_a());
		thrust::lower_bound(
			policy,
			child_arcs_parent2, child_arcs_parent2 + child_arcs.semi_arcs.size(),
			parent_arcs_high, parent_arcs_high + parent_arcs_v.size(),
			parent_arc_high_indices_to_semi_arcs.begin(),
			sort_product()
		);
		
		semi_arcs.resize(parent_arcs_v.size());
		assert(parent_arc_high_indices_to_semi_arcs.size() == parent_arcs_v.size());
		assert(indices_to_layer_v.size() > 0);
		uint32_t value = parent_arc_high_indices_to_semi_arcs[0];
		assert(indices_to_layer_v.size() >= 
		*thrust::max_element(policy, parent_arc_high_indices_to_semi_arcs.begin(),
								parent_arc_high_indices_to_semi_arcs.end()) + 1);
		
		auto permuted_high = thrust::make_permutation_iterator(indices_to_layer_v.begin(), parent_arc_high_indices_to_semi_arcs.begin());
		thrust::transform(
			policy,
			parent_arcs_v.begin(), parent_arcs_v.end(),
			permuted_high,
			target_semi_arcs_v.begin(),
			transform_arc_high()
		);
	}, semi_arc_indices_to_layer, parent_arcs.arcs, semi_arcs);

	child_arcs.sort_on_children();
	return semi_arc_indices_to_layer;
} 