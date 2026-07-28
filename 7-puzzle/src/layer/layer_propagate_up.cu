// Propagating layers up to the roots

#include "layer.h"

#include <thrust/set_operations.h>
#include <thrust/gather.h>

using namespace sro;

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

struct permute_node_product_high{
	uint32_t* data;

	permute_node_product_high(uint32_t* _data) : data(_data) {}

	__host__ __device__
	void operator() (node_product& in) {
		in.high = data[in.high];
	}
};

struct permute_node_product_low{
	uint32_t* data;

	permute_node_product_low(uint32_t* _data) : data(_data) {}

	__host__ __device__
	void operator() (node_product& in) {
		in.low = data[in.low];
	}
};

// Merges "merge" into itself, and generates children to be merged into parent layer.
data_manager<node_product> layer::propagate_quantifier_up(layer& parent_layer, data_manager<node_product>& merge, data_manager<bool>& parent_stencil, data_manager<bool>& child_stencil) {
	assert(children.size() == child_stencil.size());
	assert(parent_layer.size() == parent_stencil.size());
	
	uint32_t stencil_count = parent_stencil.count_if(::cuda::std::identity());
	data_manager<node_product> result(parent_layer.get_data_location(), stencil_count, administrator);

	// Copy all parents that are involved in quantifier
	result.with_vector([](auto policy, auto& result_v, auto& parents_v, auto& stencil_v) {
		thrust::copy_if(
			policy,
			parents_v.begin(), parents_v.end(),
			stencil_v.begin(),
			result_v.begin(),
			::cuda::std::identity()
		);
	}, parent_layer.children, parent_stencil);

	// Merge new nodes into child layer
	data_manager<node_product> old_children(get_data_location(), children.size() + merge.size(), administrator);
	children.with_vector([](auto policy, auto& children_v, auto& old_children_v, auto& merge_v){
		thrust::copy(policy, children_v.begin(), children_v.end(), old_children_v.begin());
		thrust::copy(policy, merge_v.begin(), merge_v.end(), old_children_v.begin() + children_v.size());
		thrust::sort(
			policy,
			old_children_v.begin(), old_children_v.end(),
			sort_node_product_high_low() 
		);

		auto it = thrust::unique(
			old_children_v.begin(), old_children_v.end()
		);
		old_children_v.resize(it - old_children_v.begin());
	}, old_children, merge);
	children.swap(old_children);

	// Compute lower bound of merge.
	data_manager<uint32_t> merge_indices(merge.get_data_location(), merge.size(), administrator);
	merge_indices.with_vector([](auto policy, auto& indices_v, auto& merge_v, auto& children_v){
		thrust::lower_bound(
			policy,
			children_v.begin(), children_v.end(),
			merge_v.begin(), merge_v.end(),
			indices_v.begin(),
			sort_node_product_high_low()
		);
	}, merge, children);
	merge.clear();

	// Compute lower bound of old children.
	data_manager<uint32_t> old_children_indices(old_children.get_data_location(), old_children.size(), administrator);
	old_children_indices.with_vector([](auto policy, auto& indices_v, auto& old_children_v, auto& children_v){
		thrust::lower_bound(
			policy,
			children_v.begin(), children_v.end(),
			old_children_v.begin(), old_children_v.end(),
			indices_v.begin(),
			sort_node_product_high_low()
		);
	}, old_children, children);
	old_children.clear();

	// Update indices of parent nodes
	uint32_t* old_children_indices_data = old_children_indices.with_vector_no_policy([](auto& v){
		return thrust::raw_pointer_cast(v.data());
	});

	children.sort(sort_node_product_low());
	parent_layer.children.with_vector([&](auto policy, auto& parent_v, auto& indices_v){
		thrust::for_each(
			policy,
			parent_v.begin(), parent_v.end(),
			permute_node_product_low(old_children_indices_data)
		);
	}, old_children_indices);

	children.sort(sort_node_product_high_low());
	parent_layer.children.with_vector([&](auto policy, auto& parent_v, auto& indices_v){
		thrust::for_each(
			policy,
			parent_v.begin(), parent_v.end(),
			permute_node_product_high(old_children_indices_data)
		);
	}, old_children_indices);
	
	// Create mapping from old child ids to merged ids
	data_manager<uint32_t> child_stencil_prefix_sum(child_stencil.get_data_location(), child_stencil.size(), administrator);
	child_stencil_prefix_sum.with_vector([](auto policy, auto& v, auto& child_stencil_v){
		thrust::exclusive_scan(
			policy,
			child_stencil_v.begin(), child_stencil_v.end(),
			v.begin(),
			0,
			thrust::plus<uint32_t>()
		);
	}, child_stencil);

	// We now have a mapping from the original merged list.
	// We have to pad it to create a mapping from the original child nodes
	data_manager<uint32_t> padded_merge_indices(child_stencil_prefix_sum.get_data_location(), child_stencil.size(), 0xFFFFFFFF, administrator);
	padded_merge_indices.with_vector([](auto policy, auto& padded_v, auto& original_v, auto& stencil_prefix_v, auto& stencil_v){
		thrust::gather_if(
			policy,
			stencil_prefix_v.begin(), stencil_prefix_v.end(),
			stencil_v.begin(),
			original_v.begin(),
			padded_v.begin()
		);
	}, merge_indices, child_stencil_prefix_sum, child_stencil);

	child_stencil.clear();
	child_stencil_prefix_sum.clear();
	merge_indices.clear();

	uint32_t* padded_merge_indices_data = padded_merge_indices.with_vector_no_policy([](auto& v){
		return thrust::raw_pointer_cast(v.data());
	});

	// update indices of result
	result.sort(sort_node_product_low());

	// Update low ids
	result.with_vector([&](auto policy, auto& result_v, auto& indices_v){
		thrust::for_each(
			policy, 
			result_v.begin(), result_v.end(),
			permute_node_product_low(padded_merge_indices_data)
		);
	}, padded_merge_indices);

	result.sort(sort_node_product_high_low());

	// Update high ids
	result.with_vector([&](auto policy, auto& result_v, auto& indices_v){
		thrust::for_each(
			policy, 
			result_v.begin(), result_v.end(),
			permute_node_product_high(padded_merge_indices_data)
		);
	}, padded_merge_indices);

	result.sort(sort_node_product_high_low());

	return result;
}

void layer::propagate_merge_up(layer& merge_with, layer& parent, layer& merge_with_parent) {
	// Merge layers
	data_manager<node_product> merged(children.get_data_location(), children.size() + merge_with.size(), administrator);
	merged.with_vector([](auto policy, auto& merged_v, auto& children_v, auto& other_v){
		auto it = thrust::set_union(
			policy,
			children_v.begin(), children_v.end(),
			other_v.begin(), other_v.end(),
			merged_v.begin(),
			sort_node_product_high_low()
		);
		merged_v.resize(it - merged_v.begin());
	}, children, merge_with.children);

	data_manager<uint32_t> map(children.get_data_location(), children.size(), administrator);
	map.with_vector([](auto policy, auto& map_v, auto& children_v, auto& merged_v) {
		thrust::lower_bound(
			policy,
			merged_v.begin(), merged_v.end(),
			children_v.begin(), children_v.end(),
			map_v.begin(),
			sort_node_product_high_low()
		);
	}, children, merged);

	parent.apply_ordered_map(map);
	children.clear();

	map.resize(merge_with.children.size());
	map.with_vector([](auto policy, auto& map_v, auto& children_v, auto& merged_v) {
		thrust::lower_bound(
			policy,
			merged_v.begin(), merged_v.end(),
			children_v.begin(), children_v.end(),
			map_v.begin(),
			sort_node_product_high_low()
		);
	}, merge_with.children, merged);

	merge_with_parent.apply_ordered_map(map);
	children.swap(merged);
}
