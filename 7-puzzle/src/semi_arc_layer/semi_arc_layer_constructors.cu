// Constructors of semi arc layer

#include "semi_arc_layer.h"

using namespace sro;

struct sort_node_product_high_low{
	__host__ __device__
	bool operator() (node_product a, node_product b) const {
		return thrust::make_tuple(a.high, a.low) < thrust::make_tuple(b.high, b.low);
	}
};

struct semi_arc_to_b{
	__host__ __device__
	node_product operator() (semi_arc in) const {
		return in.b;
	}
};

struct monotonic_semi_arc_to_parent{
	__host__ __device__
	uint32_t operator() (const semi_arc& in) {
		return in.a.high;
	}
};

struct transform_semi_arc_parents{
	__host__ __device__
	semi_arc operator() (semi_arc arc, uint32_t new_value) const {
		arc.a.high = new_value;
		arc.a.low = new_value;
		return arc;
	}
};

semi_arc_layer::semi_arc_layer(
		arc_layer& parent_arcs, 
		semi_arc_layer& child_arcs, 
		layer& merge_with, 
		layer& parent_layer, 
		bool monotonic
	) 
		: top_heavy(true), administrator(merge_with.administrator), semi_arcs(merge_with.administrator)
	{
	// Ensure correct data locality
	auto target = parent_layer.get_data_location();
	if (most_general_data_compatibility(
		parent_arcs.get_data_location(), 
		child_arcs.get_data_location(), 
		merge_with.get_data_location(), 
		parent_layer.get_data_location()
	) == NONE) {
		// We move from terminal node (childs) to roots (parents).
		// To avoid cascading, we move the childs to the location of the parents.
		merge_with.set_data_location(target);
		child_arcs.set_data_location(target);
		parent_arcs.set_data_location(target);
	}
	semi_arcs.to_location(target);

	// Add the children to the layer
	auto children = child_arcs.merge_semi_arcs_with_layer(merge_with);
	
	// Fill semi-arcs
	fill_semi_arcs(parent_arcs, child_arcs, merge_with);

	// Update child ids of parents. 
	// This is done using thrust::lower_bound to deduce the indices after merging.
	data_manager<uint32_t> merged_layer_indices = parent_layer.update_child_ids(children, merge_with);
	
	// Update child ids of monotonic semi_arcs
	if (monotonic) {
		semi_arcs.with_vector([&](auto policy, auto& parent_v, auto& merged_layer_indices_v){
			auto low_arcs = thrust::make_transform_iterator(parent_v.begin(), monotonic_semi_arc_to_parent());
			auto permuted_low = thrust::make_permutation_iterator(merged_layer_indices_v.begin(), low_arcs);
			thrust::transform(
				policy, 
				parent_v.begin(), parent_v.end(),
				permuted_low,
				parent_v.begin(),
				transform_semi_arc_parents()
			);
		}, merged_layer_indices);
	}

	assert(semi_arcs.size() > 0);
}

semi_arc_layer::semi_arc_layer(semi_arc_layer& child_arcs, layer& merge_with, bdd& b) : top_heavy(true), administrator(b.administrator), semi_arcs(b.administrator) {
	semi_arcs.to_location(child_arcs.get_data_location());
	assert(semi_arcs.get_data_location() == child_arcs.semi_arcs.get_data_location());
	
	auto children = child_arcs.merge_semi_arcs_with_layer(merge_with);

	// Update nodes in BDD
	b.update_bdd_nodes(children, merge_with);
	assert(child_arcs.semi_arcs.size() == 1);
	semi_arc a = child_arcs.semi_arcs[0];
	uint32_t position = merge_with.children.with_vector([&](auto policy, auto& v) {
		auto it = thrust::lower_bound(
			policy,
			v.begin(), v.end(),
			a.b,
			sort_node_product_high_low()
		);
		return it - v.begin();

	});
	b.add_root(position);
}

template<typename T>
struct apply_map_to_node_product_high{
	const uint32_t* data;
	const T iterator;

	apply_map_to_node_product_high(const uint32_t* _data, const T it) : data(_data), iterator(it) {}

	__host__ __device__
	void operator() (thrust::tuple<node_product&, uint32_t> in) {
		thrust::get<0>(in).high = *(iterator + data[thrust::get<1>(in)]);
	}
};

template<typename T>
struct apply_map_to_node_product_low{
	const uint32_t* data;
	const T iterator;

	apply_map_to_node_product_low(const uint32_t* _data, const T it) : data(_data), iterator(it) {}

	__host__ __device__
	void operator() (thrust::tuple<node_product&, uint32_t> in) {
		thrust::get<0>(in).low = *(iterator + data[thrust::get<1>(in)]);
	}
};

struct is_semi_arc_monotonic{
	__host__ __device__
	bool operator() (const semi_arc& arc) {
		return arc.a.high == arc.a.low && arc.b.high == arc.b.low;
	}
};

struct node_product_to_high_product{
	__host__ __device__
	node_product operator() (const node_product& p) {
		return {p.high, p.high};
	}
};

struct product_to_low{
	__host__ __device__
	uint32_t operator() (const node_product& p) {
		return p.low;
	}
};

struct product_to_high{
	__host__ __device__
	uint32_t operator() (const node_product& p) {
		return p.high;
	}
};

struct monotonic_semi_arc_to_product{
	__host__ __device__
	node_product operator() (const semi_arc& arc) {
		return {arc.a.low, arc.b.low};
	}
};

struct int_to_monotonic_node_product{
	node_product operator() (uint32_t in) {
		return {in, in};
	}
};

data_manager<node_product> semi_arc_layer::merge_semi_arcs_with_quantifier_layer(layer& child_layer, layer& parent_layer, data_manager<bool> stencil) {
	data_manager<node_product> children = merge_semi_arcs_with_layer(child_layer);
	data_manager<uint32_t> merged_layer_indices = parent_layer.update_child_ids(children, child_layer);
		
	// Update child ids of monotonic semi_arcs
	semi_arcs.with_vector([&](auto policy, auto& parent_v, auto& merged_layer_indices_v){
		auto low_arcs = thrust::make_transform_iterator(parent_v.begin(), monotonic_semi_arc_to_parent());
		auto permuted_low = thrust::make_permutation_iterator(merged_layer_indices_v.begin(), low_arcs);
		thrust::transform(
			policy, 
			parent_v.begin(), parent_v.end(),
			permuted_low,
			parent_v.begin(),
			transform_semi_arc_parents()
		);
	}, merged_layer_indices);
	
	// Convert monotonic semi arcs to products
	data_manager<node_product> products(semi_arcs.get_data_location(), semi_arcs.size(), administrator);
	products.with_vector([](auto policy, auto& products_v, auto& semi_arcs_v) {
		thrust::transform(
			policy,
			semi_arcs_v.begin(), semi_arcs_v.end(),
			products_v.begin(),
			monotonic_semi_arc_to_product()
		);
	}, semi_arcs);

	// Lower bound of semi arc children into child layer
	data_manager<uint32_t> target_indices(semi_arcs.get_data_location(), semi_arcs.size(), administrator);
	target_indices.with_vector([](auto policy, auto& target_v, auto& arcs_v, auto& children_v) {
		auto arcs_it = thrust::make_transform_iterator(arcs_v.begin(), semi_arc_to_b());
		thrust::lower_bound(
			policy,
			children_v.begin(), children_v.end(),
			arcs_it, arcs_it + arcs_v.size(),
			target_v.begin(),
			sort_node_product_high_low()
		);
	}, semi_arcs, child_layer.children);

	semi_arcs.clear();

	// Compute size of relevant nodes
	uint32_t relevant_nodes_size = stencil.with_vector([](auto policy, auto& v) {
		return thrust::count(
			v.begin(), v.end(),
			true
		);
	});

	// Apply stencil to find relevant nodes
	data_manager<node_product> relevant_nodes(parent_layer.get_data_location(), relevant_nodes_size, administrator);
	relevant_nodes.with_vector([](auto policy, auto& relevant_nodes_v, auto& parents_v, auto& stencil_v) {
		thrust::copy_if(
			policy,
			parents_v.begin(), parents_v.end(),
			stencil_v.begin(),
			relevant_nodes_v.begin(),
			::cuda::std::identity()
		);
	}, parent_layer.children, stencil);

	// Convert relevant nodes using map
	relevant_nodes.with_vector([](auto policy, auto& relevant_nodes_v, auto& products_v, auto& targets_v){
		using Vec = std::remove_reference_t<decltype(products_v)>;
		using IndicesVec = rebind_vector_t<Vec, uint32_t>;

		IndicesVec map(relevant_nodes_v.size());


		auto from_it = thrust::make_transform_iterator(products_v.begin(), product_to_low());
		auto to_it = thrust::make_transform_iterator(products_v.begin(), product_to_high());

		// Compute lower bound of the high ids
		auto high_id = thrust::make_transform_iterator(relevant_nodes_v.begin(), product_to_high());
		thrust::lower_bound(
			policy,
			from_it, from_it + products_v.size(),
			high_id, high_id + relevant_nodes_v.size(),
			map.begin()
		);

		// Apply lower bound of high ids
		auto zip_high = thrust::make_zip_iterator(relevant_nodes_v.begin(), thrust::make_counting_iterator(0));
		thrust::for_each(
			policy,
			zip_high, zip_high + relevant_nodes_v.size(),
			apply_map_to_node_product_high(thrust::raw_pointer_cast(map.data()), targets_v.begin())
		);

		// Sort on low ids
		// thrust::sort(
		// 	policy,
		// 	relevant_nodes_v.begin(), relevant_nodes_v.end(),
		// 	node_product_sort_on_low()
		// );

		// Compute lower bound of the low ids
		auto low_id = thrust::make_transform_iterator(relevant_nodes_v.begin(), product_to_low());
		thrust::lower_bound(
			policy,
			from_it, from_it + products_v.size(),
			low_id, low_id + relevant_nodes_v.size(),
			map.begin()
		);

		// Apply lower bound of low ids
		auto zip_low = thrust::make_zip_iterator(relevant_nodes_v.begin(), thrust::make_counting_iterator(0));
		thrust::for_each(
			policy,
			zip_low, zip_low + relevant_nodes_v.size(),
			apply_map_to_node_product_low(thrust::raw_pointer_cast(map.data()), targets_v.begin())
		);

		// Sort back on high ids again
		// thrust::sort(
		// 	policy,
		// 	relevant_nodes_v.begin(), relevant_nodes_v.end(),
		// 	node_product_sort_on_high()
		// );
	}, products, target_indices);

	// Return relevant nodes
	return relevant_nodes;
}

uint32_t semi_arc_layer::merge_semi_arcs_with_quantifier_layer(layer& child_layer, bdd& b) {
	data_manager<node_product> children = merge_semi_arcs_with_layer(child_layer);

	semi_arc a = semi_arcs[0];
	uint32_t position = child_layer.children.with_vector([&](auto policy, auto& v) {
		auto it = thrust::lower_bound(
			policy,
			v.begin(), v.end(),
			a.b,
			sort_node_product_high_low()
		);
		return it - v.begin();
	});

	// Shift root ids to account for insertion
	if (children[position] != a.b){ // Check if inserted value is new. If not, no shift occurs
		for (auto& [key, value] : b.roots) {
			if (value >= position) {
				value++;
			}
		}
	}

	uint32_t result = b.add_root(position);

	return result;
}
