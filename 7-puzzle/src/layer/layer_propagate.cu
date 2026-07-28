// Propagating layers

#include "layer.h"

#include <thrust/set_operations.h>
#include <thrust/iterator/constant_iterator.h>

using namespace sro;


template <typename M>
struct apply_map_func{
	M begin;
	
	apply_map_func(M _begin) : begin(_begin) {}

	__host__ __device__
	void operator() (node_product& in) {
		in.apply_map(begin);
	}
};

struct sort_node_product_high_low{
	__host__ __device__
	bool operator() (node_product a, node_product b) const {
		return thrust::make_tuple(a.high, a.low) < thrust::make_tuple(b.high, b.low);
	}
};

struct sort_node_product_low{
	__host__ __device__
	bool operator() (node_product a, node_product b) {
		return thrust::make_tuple(a.low, a.high) < thrust::make_tuple(b.low, b.high);
	}
};

struct transform_product_to_low{
	__host__ __device__
	uint32_t operator() (node_product in) const {
		return in.low;
	}
};

struct transform_low_children{
	__host__ __device__
	node_product operator() (node_product parent, uint32_t new_low) const {
		parent.low = new_low;
		return parent;
	}
};

struct sort_product{
	__host__ __device__
	bool operator() (node_product a, node_product b) {
		return thrust::make_tuple(a.high, a.low) < thrust::make_tuple(b.high, b.low);
	}
};

struct transform_product_to_high{
	__host__ __device__
	uint32_t operator() (node_product in) const {
		return in.high;
	}
};

struct transform_high_children{
	__host__ __device__
	node_product operator() (node_product parent, uint32_t new_high) const {
		parent.high = new_high;
		return parent;
	}
};

data_manager<uint32_t> layer::update_child_ids(data_manager<node_product>& old_children, layer& child_layer) {
	data_manager<uint32_t> merged_layer_indices(get_data_location(), old_children.size(), administrator);

	// Compute lower bound of original layer with merged layer
	old_children.with_vector([&](auto policy, auto& v, auto& children_v, auto& merged_layer_indices_v){
		thrust::lower_bound(
			policy, 
			children_v.begin(), children_v.end(),
			v.begin(), v.end(),
			merged_layer_indices_v.begin(),
			sort_node_product_high_low()
		);
	}, child_layer.children, merged_layer_indices);

	// Update low children
	children.sort(sort_node_product_low());
	children.with_vector([&](auto policy, auto& parent_v, auto& merged_layer_indices_v){
		auto low_arcs = thrust::make_transform_iterator(parent_v.begin(), transform_product_to_low());
		auto permuted_low = thrust::make_permutation_iterator(merged_layer_indices_v.begin(), low_arcs);
		thrust::transform(
			policy, 
			parent_v.begin(), parent_v.end(),
			permuted_low,
			parent_v.begin(),
			transform_low_children()
		);
	}, merged_layer_indices);

	// Update high children
	children.sort(sort_product());
	children.with_vector([&](auto policy, auto& parent_v, auto& merged_layer_indices_v){
		auto high_arcs = thrust::make_transform_iterator(parent_v.begin(), transform_product_to_high());
		auto permuted_high = thrust::make_permutation_iterator(merged_layer_indices_v.begin(), high_arcs);
		thrust::transform(
			policy, 
			parent_v.begin(), parent_v.end(),
			permuted_high,
			parent_v.begin(),
			transform_high_children()
		);
	}, merged_layer_indices);
	return merged_layer_indices;
}

struct node_product_to_low{
	__host__ __device__
	uint32_t operator() (node_product product) {
		return product.low;
	}
};

struct node_product_to_high{
	__host__ __device__
	uint32_t operator() (node_product product) {
		return product.high;
	}
};

// Propagate boolean filter towards terminals
data_manager<bool> layer::propagate_encoding(data_manager<bool>& encoding, layer& child_layer) {
	encoding.to_location(get_data_location());
	data_manager<bool> new_encoding(get_data_location(), child_layer.children.size(), false, administrator);

	children.with_vector([&](auto policy, auto& children_v, auto& encoding_v, auto& new_encoding_v){
		using Vec = std::remove_reference_t<decltype(children_v)>;

		Vec filtered_nodes(children_v.size());
		auto it = thrust::copy_if(
			policy,
			children_v.begin(), children_v.end(),
			encoding_v.begin(),
			filtered_nodes.begin(),
			::cuda::std::identity()
		);
		filtered_nodes.resize(it - filtered_nodes.begin());

		using IdxVec = rebind_vector_t<Vec, uint32_t>;
		IdxVec indices(filtered_nodes.size() * 2);
		thrust::transform(
			policy,
			filtered_nodes.begin(), filtered_nodes.end(),
			indices.begin(),
			node_product_to_low()
		);
		thrust::transform(
			policy,
			filtered_nodes.begin(), filtered_nodes.end(),
			indices.begin() + filtered_nodes.size(),
			node_product_to_high()
		);


		auto true_it = thrust::make_constant_iterator(true);
		thrust::scatter(
			policy,
			true_it, true_it + indices.size(),
			indices.begin(),
			new_encoding_v.begin()
		);
	}, encoding, new_encoding);
	return new_encoding;
}

void layer::propagate_separation_down(layer& separated, data_manager<bool>& stencil, data_manager<bool>& stencil_separated, data_manager<bool>& child_stencil, data_manager<bool>& child_stencil_separated) {
	// Copy separated children to new layer
	separated.children.with_vector([](auto policy, auto& children_separated_v, auto& children_v, auto& stencil_separated_v){
		children_separated_v.resize(thrust::count(policy, stencil_separated_v.begin(), stencil_separated_v.end(), true));
		
		thrust::copy_if(
			policy,
			children_v.begin(), children_v.end(),
			stencil_separated_v.begin(),
			children_separated_v.begin(),
			::cuda::std::identity()
		);
	}, children, stencil_separated);
	
	// Remove children that are not part of constancy
	children.with_vector([](auto policy, auto& children_v, auto& stencil_v){
		auto it = thrust::remove_if(
			policy,
			children_v.begin(), children_v.end(),
			stencil_v.begin(),
			::cuda::std::logical_not()
		);
		children_v.resize(it - children_v.begin());
	}, stencil);

	// Generate separated child stencil
	separated.children.with_vector([](auto policy, auto& children_v, auto& stencil_v){
		using Vec = std::remove_reference_t<decltype(children_v)>;
		using IdxVec = rebind_vector_t<Vec, uint32_t>;
		
		IdxVec indices(children_v.size() * 2);

		thrust::transform(
			policy,
			children_v.begin(), children_v.end(),
			indices.begin(),
			node_product_to_low()
		);

		thrust::transform(
			policy,
			children_v.begin(), children_v.end(),
			indices.begin() + children_v.size(),
			node_product_to_high()
		);

		auto it = thrust::unique(
			policy,
			indices.begin(), indices.end()
		);

		indices.resize(it - indices.begin());

		auto true_it = thrust::make_constant_iterator(true);
		thrust::scatter(
			policy,
			true_it, true_it + indices.size(),
			indices.begin(),
			stencil_v.begin()
		);
	}, child_stencil_separated);

	// Generate constancy child stencil
	children.with_vector([](auto policy, auto& children_v, auto& stencil_v){
		using Vec = std::remove_reference_t<decltype(children_v)>;
		using IdxVec = rebind_vector_t<Vec, uint32_t>;
		
		IdxVec indices(children_v.size() * 2);

		thrust::transform(
			policy,
			children_v.begin(), children_v.end(),
			indices.begin(),
			node_product_to_low()
		);

		thrust::transform(
			policy,
			children_v.begin(), children_v.end(),
			indices.begin() + children_v.size(),
			node_product_to_high()
		);

		auto it = thrust::unique(
			policy,
			indices.begin(), indices.end()
		);

		indices.resize(it - indices.begin());

		auto true_it = thrust::make_constant_iterator(true);
		thrust::scatter(
			policy,
			true_it, true_it + indices.size(),
			indices.begin(),
			stencil_v.begin()
		);
	}, child_stencil);

	// Update indices of separated layer using child_stencil_separated
	separated.children.with_vector([](auto policy, auto& separated_v, auto& stencil_v){
		using Vec = std::remove_reference_t<decltype(separated_v)>;
		using MapVec = rebind_vector_t<Vec, uint32_t>;

		MapVec map(stencil_v.size());

		thrust::exclusive_scan(
			policy,
			stencil_v.begin(), stencil_v.end(),
			map.begin(),
			0u,
			thrust::plus<uint32_t>()
		);

		thrust::for_each(
			policy, 
			separated_v.begin(), separated_v.end(),
			apply_map_func(map.begin())
		);
	}, child_stencil_separated);

	// Update indices of constancy layer using child_stencil
	children.with_vector([](auto policy, auto& constancy_v, auto& stencil_v){
		using Vec = std::remove_reference_t<decltype(constancy_v)>;
		using MapVec = rebind_vector_t<Vec, uint32_t>;

		MapVec map(stencil_v.size());

		thrust::exclusive_scan(
			policy,
			stencil_v.begin(), stencil_v.end(),
			map.begin(),
			0u,
			thrust::plus<uint32_t>()
		);

		thrust::for_each(
			policy, 
			constancy_v.begin(), constancy_v.end(),
			apply_map_func(map.begin())
		);
	}, child_stencil);
}

void layer::propagate_duplication_down(layer& duplicated, data_manager<bool>& stencil, data_manager<bool>& child_stencil, bool bottom) {
	// Copy duplicated children to new layer
	duplicated.children.with_vector([](auto policy, auto& children_duplicated_v, auto& children_v, auto& stencil_duplicated_v){
		children_duplicated_v.resize(thrust::count(policy, stencil_duplicated_v.begin(), stencil_duplicated_v.end(), true));
		
		thrust::copy_if(
			policy,
			children_v.begin(), children_v.end(),
			stencil_duplicated_v.begin(),
			children_duplicated_v.begin(),
			::cuda::std::identity()
		);
	}, children, stencil);

	if (bottom) return; // The bottom (true/false for BDDS, more elements for MTBDDS) never changes the number of elements

	// Generate duplicated child stencil
	duplicated.children.with_vector([](auto policy, auto& children_v, auto& stencil_v){
		using Vec = std::remove_reference_t<decltype(children_v)>;
		using IdxVec = rebind_vector_t<Vec, uint32_t>;
		
		IdxVec indices(children_v.size() * 2);

		thrust::transform(
			policy,
			children_v.begin(), children_v.end(),
			indices.begin(),
			node_product_to_low()
		);

		thrust::transform(
			policy,
			children_v.begin(), children_v.end(),
			indices.begin() + children_v.size(),
			node_product_to_high()
		);

		auto it = thrust::unique(
			policy,
			indices.begin(), indices.end()
		);

		indices.resize(it - indices.begin());

		auto true_it = thrust::make_constant_iterator(true);
		thrust::scatter(
			policy,
			true_it, true_it + indices.size(),
			indices.begin(),
			stencil_v.begin()
		);
	}, child_stencil);

	// Update indices of duplicated layer using child_stencil
	duplicated.children.with_vector([](auto policy, auto& separated_v, auto& stencil_v){
		using Vec = std::remove_reference_t<decltype(separated_v)>;
		using MapVec = rebind_vector_t<Vec, uint32_t>;

		MapVec map(stencil_v.size());

		thrust::exclusive_scan(
			policy,
			stencil_v.begin(), stencil_v.end(),
			map.begin(),
			0u,
			thrust::plus<uint32_t>()
		);

		thrust::for_each(
			policy, 
			separated_v.begin(), separated_v.end(),
			apply_map_func(map.begin())
		);
	}, child_stencil);
}