// Queries and modifications of the overall structure of the layer

#include "layer.h"

using namespace sro;

struct sum_children{
	const uint64_t* array;

	sum_children(uint64_t* _array) : array(_array) {}

	__host__ __device__
	uint64_t operator() (node_product in){
		return array[in.low] + array[in.high];
	}
};

/**
 * Counts the number of satisfying assignments of this layer, based on the number of satisfying assignments of the layer
 * below. 
 * 
 * @param lower_count The list of satisfying assignment counts of the layer below.
 * 
 * @returns The list of satisfying assignment counts of this layer. The `i`-th element of this list is equal to:
 * `lower_count[children[i].low] + lower_count[children[i].high]`
 * 
 * @pre ∀ sro::node_product child ∈ children : child.low < lower_count.size()
 * 
 * @pre ∀ sro::node_product child ∈ children : child.high < lower_count.size()
 */
data_manager<uint64_t> layer::satisfying_assignments(data_manager<uint64_t>& lower_count) {
	lower_count.to_location(get_data_location());
	data_manager<uint64_t> result(get_data_location(), children.size(), administrator);

	lower_count.with_vector([](auto policy, auto& lower_count_v, auto& result_v, auto& children_v){
		thrust::transform(
			policy,
			children_v.begin(), children_v.end(), 
			result_v.begin(), 
			sum_children(thrust::raw_pointer_cast(lower_count_v.data()))
		);
	}, result, children);

	return result;
}

/**
 * Counts the number of nodes in this layer. The result might differ from other BDD libraries, as GPUdecide works with 
 * Quasi-Reduced Ordered Binary Decision Diagrams (QROBDD). Equivalent to layer::size.
 * 
 * @returns `children.size()`
 */
uint32_t layer::node_count() {
	return children.size();
}

struct children_unequal{
	__device__ __host__
	bool operator() (node_product in) {
		return in.low != in.high;
	}
};

/**
 * Counts the number of irreducible nodes in this layer. A node is irreducible iff its low child is unequal to its high
 * child. The result should be equal to the number of nodes in the (fully) Reduced Ordered Binary Decision Diagram (ROBDD).
 * 
 * @returns `children.count_if(children_unequal())`; the number of irreducible nodes.
 */
uint32_t layer::irreducible_node_count() {
	return children.count_if(children_unequal());
}

template <typename M>
struct apply_map_func{
	M begin;
	
	apply_map_func(M _begin) : begin(_begin) {}

	__host__ __device__
	void operator() (node_product& in) {
		in.apply_map(begin);
	}
};

struct sort_product_on_g{
	__host__ __device__
	bool operator() (node_product a, node_product b) {
		return thrust::make_tuple(a.high, a.low) < thrust::make_tuple(b.high, b.low); 
	}
};

// Apply a map to the layer, and return the map for the layer above
data_manager<uint32_t> layer::apply_map(data_manager<uint32_t>& map) {
	// Apply input map
	children.with_vector([&](auto policy, auto& v, auto& map_v) {
		thrust::for_each(
			policy, 
			v.begin(), v.end(),
			apply_map_func(map_v.begin())
		);
	}, map);
	
	data_manager<uint32_t> new_map_inverse(get_data_location(), children.size(), administrator);
	new_map_inverse.with_vector([&](auto policy, auto& v) {
		thrust::sequence(v.begin(), v.end());
	});

	// Sort keys along with new map
	// We get as a result the inverse map from the sorted indices to the original indices
	children.with_vector(
		[](auto policy, auto& v, auto& payload_v){
			thrust::sort_by_key(
				policy,
				v.begin(), v.end(),
				payload_v.begin(),
				sort_product_on_g()
			);
		},
		new_map_inverse
	);

	// Reverse generated permutation to get map from old to new indices
	data_manager<uint32_t> new_map(get_data_location(), children.size(), administrator);
	new_map.with_vector([&](auto policy, auto& v) {
		thrust::sequence(v.begin(), v.end());
	});
	new_map_inverse.with_vector([](auto policy, auto& inv_v, auto& v){
		thrust::sort_by_key(
			policy,
			inv_v.begin(), inv_v.end(),
			v.begin()
		);
	}, new_map);

	return new_map;
}

void layer::apply_ordered_map(data_manager<uint32_t>& mapping) {
	// Update indices of constancy layer using child_stencil
	children.with_vector([](auto policy, auto& children_v, auto& mapping_v){
		thrust::for_each(
			policy, 
			children_v.begin(), children_v.end(),
			apply_map_func(mapping_v.begin())
		);
	}, mapping);
}

struct is_product_removable{
	__host__ __device__
	bool operator() (node_product& p) {
		return p.removable();
	}
};

/**
 * TODO: this function is not implemented
 * 
 * Returns if every element of this layer is reducible. If they are, this layer can be removed, and the preceding layer
 * can be attached directly to the proceeding layer.
 */
bool layer::is_removable() {
	// return children.with_vector([](auto policy, auto& v){
	// 	return thrust::all_of(
	// 		policy,
	// 		v.begin(), v.end(),
	// 		is_product_removable()
	// 		// ::cuda::std::mem_fn(&node_product::removable)
	// 	);
	// });
	return false;
}

struct removable_node_product{
	__host__ __device__
	node_product operator() (uint32_t in) {
		return {in, in};
	}
};

/**
 * Fills this layer with `number_of_elements` reducible nodes.
 * 
 * @post children.size() == number_of_elements
 * 
 * @post ∀ 0 <= uint32_t i < number_of_elements : children[i].low == i 
 * 
 * @post ∀ 0 <= uint32_t i < number_of_elements : children[i].high == i 
 */
void layer::make_removable_layer(uint32_t number_of_elements) {
	children.resize(number_of_elements);

	children.with_vector([](auto policy, auto& v){
		auto it = thrust::make_counting_iterator(0u);

		thrust::transform(
			policy,
			it, it + v.size(),
			v.begin(),
			removable_node_product()
		);
	});
}

__host__ __device__
void node_product::project_stencil(uint32_t* stencil) {
	stencil[low] = 0;
	stencil[high] = 0;
}

#include "../../collect_garbage.h"

/**
 * Removes nodes that are not in use anymore. Creates a stencil of all nodes that are used, and then removes nodes not
 * covered by this stencil. This function is called when a root node is destructed.
 * 
 * @pre `_label > from._label`
 * 
 * @param from The parent layer of this layer. 
 */
void layer::collect_garbage(layer& from) {
	auto original_data_location = from.get_data_location();
	from.set_data_location(get_data_location());
	children.with_vector([&](auto policy, auto& v, auto& parent_v) {
		garbage_collection(policy, v, parent_v);
		// // Compiler computes types
		// using Vec = std::remove_reference_t<decltype(v)>;
		// using StencilVec = rebind_vector_t<Vec, uint32_t>;
		
		// // Generate list of referenced nodes
		// StencilVec stencil(v.size(), 1);
		// thrust::for_each(
		// 	policy,
		// 	parent_v.begin(), parent_v.end(),
		// 	Wrapper<&node_product::project_stencil, uint32_t*>(thrust::raw_pointer_cast(stencil.data()))
		// );

		// // Compute lower bound into old list
		// auto new_end = thrust::remove_if(
		// 	policy,
		// 	v.begin(), v.end(), 
		// 	stencil.begin(),
		// 	::cuda::std::identity{}
		// );
		// v.resize(new_end - v.begin());
		
		// // Update parent layer's children to new lower bound by applying stencil
		// thrust::exclusive_scan(stencil.begin(), stencil.end(), stencil.begin());
		// auto count_it = thrust::make_counting_iterator<uint32_t>(0);
		// auto map = thrust::make_transform_iterator(
		// 	thrust::make_zip_iterator(thrust::make_tuple(count_it, stencil.begin())),
		// 	minus()
		// );

		// thrust::for_each(
		// 	policy, 
		// 	parent_v.begin(), parent_v.end(),
		// 	apply_map_func(map)
		// );
	}, from.children);
	from.set_data_location(original_data_location);
}

bool layer::check_validity() {
	return children.is_sorted(sort_product_on_g());
}

/**
 * Returns the size of this layer. Equivalent to layer::node_count.
 * 
 * @returns `children.size()`
 */
uint32_t layer::size() {
	return children.size();
}

/**
 * Returns whether or not the layer starts with the false BDD. Due to the way ordering works, the false BDD will always 
 * be at the beginning, so this is equivalent to whether false occurs in this layer.
 * 
 * @returns True if layer starts with the false BDD. False otherwise.
 */
bool layer::starts_with_false() {
	node_product p = children[0];
	return p.low == 0 && p.high == 0;
}

/**
 * Gets the child idx of the layer at `idx`. If `high`, it returns the high child index, otherwise it returns the low child
 * index.
 * 
 * @param idx The index of the child.
 * 
 * @param high Whether to return the high child.
 * 
 * @returns high ? children[idx].high : children[idx].low 
 */
uint32_t layer::get_child(uint32_t idx, bool high) {
	node_product p = children[idx];
	return high ? p.high : p.low;
}

struct sort_node_product_high_low{
	__host__ __device__
	bool operator() (node_product a, node_product b) const {
		return thrust::make_tuple(a.high, a.low) < thrust::make_tuple(b.high, b.low);
	}
};

/**
 * Returns the index of a node product, or if it does not occur in `children`, the index it would have if it were inserted.
 * 
 * @param p The node product to search for
 * 
 * @returns The index of `p` in `children`, or the index `p` would have if it were inserted into `children`.
 */
uint32_t layer::get_index_of(node_product p) {
	return children.with_vector([&](auto policy, auto& v) {
		auto it = thrust::lower_bound(
			policy,
			v.begin(), v.end(),
			p,
			sort_node_product_high_low()
		);
		return it - v.begin();
	});
}

/**
 * Get the index of the variable this layer corresponds to.
 * 
 * @returns _label
 */
 uint16_t layer::get_label() {
	return _label;
}

/**
 * Set the data location of the data in this layer.
 * 
 * @param l The location to which this layer should be moved.
 */
void layer::set_data_location(data_location l) {
	children.to_location(l);
}

/**
 * Get the data location of the data in this layer.
 * 
 * @returns The data location of the data in this layer.
 */
data_location layer::get_data_location() {
	return children.get_data_location();
}


#if USE_STORAGE
bool layer::is_spilled() {
    return children.is_spilled();
}
#endif

void layer::update_irreducible_count() {
    _irreducible_count = irreducible_node_count();
}

/**
 * Returns a vector of the data associated with this layer.
 * 
 * @returns std::vector<node_product> containing the children of this layer.
 */
std::vector<node_product> layer::get_data() {
	thrust::host_vector<node_product> result = children;
	return std::vector<node_product>(result.begin(), result.end());
}

/**
 * Sets the data associated with this layer.
 * 
 * @param data The new children for this layer.
 */
void layer::set_data(std::vector<node_product> data) {
	children = data;
}

/**
 * Prints the low and high children for this layer. One line is printed for the low children, and one line is printed for
 * the high children. Alignment is arranged through tabs. An extra newline is printed at the end.
 */
void layer::print() {
	printf("Low: \t\t");
	for (node_product n : children) {
		printf("%5u ", n.low);
	}
	printf("\n");
	printf("High: \t\t");
	for (node_product n : children) {
		printf("%5u ", n.high);
	}
	printf("\n");
}
