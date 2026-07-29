// Removal and insertion of elements into layers

#include "layer.h"

#include <thrust/set_operations.h>

using namespace sro;

// Remove and replace elements, and return map vector
data_manager<uint32_t> layer::remove_elements(std::vector<uint32_t> indices, std::vector<uint32_t> replacement_indices) {
	// Fill stencil
	thrust::host_vector<uint8_t> stencil(size());
	for (int i = 0; i < stencil.size(); i++) stencil[i] = 0;
	for (uint32_t index : indices) {
		stencil[index] = 1;
	}

	data_manager<uint32_t> map(get_data_location(), stencil.size(), administrator);

	children.with_vector([&](auto policy, auto& children_v, auto& map_v){
		// Compiler computes types
		using Vec = std::remove_reference_t<decltype(children_v)>;
		using StencilVec = rebind_vector_t<Vec, uint8_t>;
		using IndicesVec = rebind_vector_t<Vec, uint8_t>;

		// Remove elements
		StencilVec d_stencil = stencil;
		auto new_end = thrust::remove_if(
			policy, 
			children_v.begin(), children_v.end(),
			d_stencil.begin(),
			::cuda::std::identity{}
		);
		children.resize(new_end - children_v.begin());

		// Apply prefix sum on stencil to compute map
		thrust::transform(
			policy, 
			d_stencil.begin(),
			d_stencil.end(),
			d_stencil.begin(),
			thrust::logical_not<bool>()
		);
		thrust::exclusive_scan(
			policy, 
			d_stencil.begin(), d_stencil.end(), 
			map_v.begin()
		);

		// Copy replacement indices into the map
		thrust::host_vector<uint32_t> h_indices = indices;
		IndicesVec d_indices = h_indices;
		thrust::host_vector<uint32_t> h_replacement_indices = replacement_indices;
		IndicesVec d_replacement_indices = h_replacement_indices;

		thrust::scatter(
			policy, 
			d_replacement_indices.begin(), d_replacement_indices.end(),
			d_indices.begin(),
			map_v.begin()
		);
	}, map);

	assert(check_validity());

	return map;
}

std::vector<uint32_t> layer::remove_elements(std::vector<uint32_t> indices) {
	// Fill stencil
	thrust::host_vector<uint8_t> stencil(size());
	for (int i = 0; i < stencil.size(); i++) stencil[i] = 0;
	for (uint32_t index : indices) {
		stencil[index] = 1;
	}

	// Remove elements
	children.with_vector([&](auto policy, auto& v) {
		// Compiler computes types
		using Vec = std::remove_reference_t<decltype(v)>;
		using StencilVec = rebind_vector_t<Vec, uint8_t>;

		StencilVec d_stencil = stencil;
		auto new_end = thrust::remove_if(
			v.begin(), v.end(),
			d_stencil.begin(),
			::cuda::std::identity{}
		);
		children.resize(new_end - v.begin());
	});

	// Return map
	std::vector<uint32_t> result(stencil.size());
	uint32_t map = 0;
	for (int i = 0; i < stencil.size(); i++) {
		if (stencil[i] == 0) result[i] = map++;
		else result[i] = map;
		// else result[i] = 0xFFFFFFFF;
	}
	return result;
}

struct sort_node_product_high_low{
	__host__ __device__
	bool operator() (node_product a, node_product b) const {
		return thrust::make_tuple(a.high, a.low) < thrust::make_tuple(b.high, b.low);
	}
};

data_manager<uint32_t> layer::insert_elements(data_manager<node_product> elements) {
	// Ensure uniqueness of elements
	elements.unique();

	data_manager<uint32_t> result(children.get_data_location(), size(), 0, administrator);

	// Perform union of elements with children
	data_manager<node_product> old_children(children.get_data_location(), children.size() + elements.size(), administrator);
	children.with_vector([](auto policy, auto& children_v, auto& elements_v, auto& result_v){
		using Vec = std::remove_reference_t<decltype(children_v)>;
		Vec target_v(children_v.size() + elements_v.size());

		auto it = thrust::set_union(
			policy,
			children_v.begin(), children_v.end(),
			elements_v.begin(), elements_v.end(),
			target_v.begin(),
			sort_node_product_high_low()
		);
		target_v.resize(it - target_v.begin());
		children_v.swap(target_v);

		thrust::lower_bound(
			policy,
			children_v.begin(), children_v.end(),
			target_v.begin(), target_v.end(),
			result_v.begin(),
			sort_node_product_high_low()
		);
	}, elements, result);

	return result;
}
 