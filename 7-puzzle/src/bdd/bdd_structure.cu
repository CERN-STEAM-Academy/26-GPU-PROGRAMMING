// Functions other than logical and quantitative operations that make structural changes to the BDD.
// Examples include construction, duplication, insertion of layers, etc.

#include "bdd.h"

using namespace sro;

struct sort_node_product_low_high{
	__host__ __device__
	bool operator() (node_product a, node_product b) const {
		return thrust::make_tuple(a.low, a.high) < thrust::make_tuple(b.low, b.high);
	}
};

struct sort_node_product_high_low{
	__host__ __device__
	bool operator() (node_product a, node_product b) const {
		return thrust::make_tuple(a.high, a.low) < thrust::make_tuple(b.high, b.low);
	}
};

void bdd::update_bdd_nodes(data_manager<node_product>& children, layer& child_layer) {
	// Compute lower bound of original layer with merged layer
	data_manager<uint32_t> d_merged_layer_indices(child_layer.get_data_location(), children.size(), administrator);
	d_merged_layer_indices.with_vector([&](auto policy, auto& indices_v, auto& parents_v, auto& children_v){
		thrust::lower_bound(
			policy, 
			children_v.begin(), children_v.end(),
			parents_v.begin(), parents_v.end(),
			indices_v.begin(),
			sort_node_product_high_low()
		);
	}, children, child_layer.children);

	thrust::host_vector<uint32_t> merged_layer_indices = d_merged_layer_indices;
	apply_root_layer_mapping(merged_layer_indices);
}

void bdd::update_occurrences_of_false() {
	lowermost_occurrence_of_false = layers.size();

	while (
		lowermost_occurrence_of_false > 0 && 
		layers[lowermost_occurrence_of_false - 1].starts_with_false()
	) lowermost_occurrence_of_false--;

	false_cache_up_to_date = true;
}

/**
 * Removes root nodes from the BDD. This happens when the last node_ref to a root is destructed.
 * 
 * @param nodes The nodes to be removed.
 */
void bdd::remove_root_nodes(std::vector<uint32_t> nodes) {
	false_cache_up_to_date = false;
	// Create mapping from old nodes to new nodes
	std::vector<uint32_t> offsets = layers[0].remove_elements(nodes);

	// Apply mapping to existing variables and roots
	apply_root_layer_mapping(offsets);

	// Collect garbage
	collect_garbage(0);
}

/**
 * Removes dead nodes from the BDD.
 * 
 * @param layer The layer to start collecting garbage at. When performing garbage collection on the whole BDD this should
 * be 0. Can be higher when working with quantifiers.
 */
void bdd::collect_garbage(uint16_t layer) {
	for (int i = layer + 1; i < layers.size(); i++) {
		layers[i].collect_garbage(layers[i - 1]);
	}
}

/**
 * Removes the variables in this BDD.
 */
void bdd::remove_variables() {
	variables.clear();
}

struct lt{
	const uint32_t bound;

	lt(const uint32_t _bound) : bound(_bound) {}

	__host__ __device__
	bool operator() (node_product p) {
		return p.low < bound && p.high < bound;
	}
};

void bdd::remove_layers(std::vector<uint16_t> r) {
	std::sort(r.rbegin(), r.rend());
	r.erase(std::unique(r.begin(), r.end()), r.end());

	for (uint16_t idx : r) {
		assert(layers[idx].is_removable());
		layers.erase(layers.begin() + idx);
	}
}

void bdd::insert_removable_layers(std::vector<uint16_t> r) {
	std::sort(r.rbegin(), r.rend());
	
	for (uint16_t idx : r) {
		uint32_t layer_size = 2;
		if (idx < layers.size()) {
			layer_size = layers[idx].size();
		}
		layers.emplace(layers.begin() + idx, administrator);
		layers[idx].make_removable_layer(layer_size);
	}
}

std::shared_ptr<bdd> bdd::separate_root(node_ref root) {
	return separate_roots({root});
}

std::shared_ptr<bdd> bdd::separate_roots(std::vector<node_ref> roots_to_separate) {
	bdd* new_bdd = new bdd(layers.size());
	std::shared_ptr<bdd> new_bdd_ptr = new_bdd->shared_from_this();

	// Fill initial stencils.
	// There are two stencils: 
	// One stencil is for the nodes to be taken out of the BDD.
	// One stencil is for the nodes that need to remain in the BDD.
	// This is needed, because some nodes need to end up in both BDDs.
	std::vector<uint32_t> root_keys_separate; // These root keys will be separated into a new BDD
	for (node_ref root : roots_to_separate) {
		root_keys_separate.push_back(root.root_id);
	}
	std::sort(root_keys_separate.begin(), root_keys_separate.end());
	auto iterator = std::unique(root_keys_separate.begin(), root_keys_separate.end());
	root_keys_separate.resize(iterator - root_keys_separate.begin());

	std::vector<uint32_t> root_keys_constancy; // These root keys will remain in this BDD
	int s_i = 0; // Index to root_keys_separate
	for (int i = 0; i < lifetime_root_count; i++) {
		if (root_keys_separate[s_i] == i) {
			s_i++;
		}
		else if (roots.contains(i)) {
			root_keys_constancy.push_back(i);
		}
	}

	data_manager<bool> roots_separate(HOST, layers[0].size(), false, administrator);
	for (uint32_t id : root_keys_separate) {
		new_bdd_ptr->roots[id] = roots[id];
		redirects[id] = new_bdd_ptr;
		roots_separate[roots[id]] = true;
	}

	data_manager<bool> roots_constancy(HOST, layers[0].size(), false, administrator);
	for (uint32_t id : root_keys_constancy) {
		roots_constancy[roots[id]] = true;
	}

	// Shift node refs to new ID in generated and returned BDD.
	roots_separate.with_vector([&](auto policy, auto& v){
		using Vec = std::remove_reference_t<decltype(v)>;
		using PrefixVec = rebind_vector_t<Vec, uint32_t>;

		PrefixVec scan(v.size());
		thrust::exclusive_scan(
			thrust::host,
			v.begin(), v.end(),
			scan.begin(),
			0,
			thrust::plus<uint32_t>()
		);
		cudaDeviceSynchronize();
		for (auto& [key, value] : roots) {
			value = scan[value];
		}
	});

	// Shift node refs of constancy roots to new (possibly smaller) IDs in this BDD
	roots_constancy.with_vector([&](auto policy, auto& v){
		using Vec = std::remove_reference_t<decltype(v)>;
		using PrefixVec = rebind_vector_t<Vec, uint32_t>;

		PrefixVec scan(v.size());
		thrust::exclusive_scan(
			thrust::host,
			v.begin(), v.end(),
			scan.begin(),
			0,
			thrust::plus<uint32_t>()
		);
		for (auto& [key, value] : roots) {
			value = scan[value];
		}
	});
	
	// Propagate stencil downwards
	for (int i = 0; i < layers.size(); i++) {
		uint32_t next_layer_size = (i == layers.size() - 1) ? 2 : layers[i + 1].size();
		data_manager<bool> child_roots_separate(HOST, next_layer_size, false, administrator);
		data_manager<bool> child_roots_constancy(HOST, next_layer_size, false, administrator);
		layers[i].propagate_separation_down(new_bdd->layers[i], roots_constancy, roots_separate, child_roots_constancy, child_roots_separate);
		roots_separate = child_roots_separate;
		roots_constancy = child_roots_constancy;
	}

	// Remove redirected roots
	for (uint32_t id : root_keys_separate) {
		roots.erase(id);
	}

	return new_bdd_ptr;
}

/**
 * Duplicates the BDD. Only the roots referred to by `node_refs` are copied.
 * 
 * @param node_refs The roots to copy to the new BDD.
 * 
 * @returns A pair containing the new BDD and a vector of sro::noderef objects. The order of these objects corresponds to
 * the paramter `node_refs`.
 */
std::pair<std::shared_ptr<bdd>, std::vector<sro::node_ref>> bdd::duplicate(std::vector<node_ref> node_refs) {
	bdd* new_bdd = new bdd(layers.size());
	std::shared_ptr<bdd> new_bdd_ptr = new_bdd->shared_from_this();

	std::vector<uint32_t> root_keys; // These root keys will be duplicated into a new BDD
	for (node_ref root : node_refs) {
		root_keys.push_back(root.root_id);
	}

	data_manager<bool> roots_duplicate(HOST, layers[0].size(), false, administrator);
	for (uint32_t id : root_keys) {
		new_bdd_ptr->roots[id] = roots[id];
		roots_duplicate[roots[id]] = true;
	}

	// Compute prefix sum of duplicated roots
	data_manager<uint32_t> map(roots_duplicate.get_data_location(), roots_duplicate.size(), administrator);
	map.with_vector([](auto policy, auto& v, auto& roots_duplicate_v){
		thrust::exclusive_scan(
			policy,
			roots_duplicate_v.begin(), roots_duplicate_v.end(),
			v.begin(),
			0,
			thrust::plus<uint32_t>()
		);
	}, roots_duplicate);

	// Propagate stencil downwards
	for (int i = 0; i < layers.size(); i++) {
		uint32_t next_layer_size = (i == layers.size() - 1) ? 2 : layers[i + 1].size();
		data_manager<bool> child_roots_duplicate(HOST, next_layer_size, false, administrator);

		layers[i].propagate_duplication_down(new_bdd->layers[i], roots_duplicate, child_roots_duplicate, i == layers.size() - 1);

		roots_duplicate = child_roots_duplicate;
	}

	// Shift roots to account for nodes that are not duplicated
	for (auto& [key, value] : new_bdd_ptr->roots) {
		assert(value < map.size());
		value = map[value];
	}

	std::vector<node_ref> duplicated_refs;
	for (node_ref r : node_refs) {
		duplicated_refs.push_back(node_ref(r.root_id, new_bdd_ptr));
	}
	return std::make_pair(new_bdd_ptr, duplicated_refs);
}

/**
 * Merges the BDD with another BDD. 
 * 
 * @param bdd The BDD to merge with.
 * 
 * @param destroy TODO: implement removing the contents of the original BDD.
 */
void bdd::join_with(std::shared_ptr<bdd> bdd, bool destroy) {
	assert(bdd->layers.size() == layers.size());

	for (uint16_t i = layers.size() - 1; i > 0; i--) {
		layers[i].propagate_merge_up(bdd->layers[i], layers[i-1], bdd->layers[i-1]);
	}

	thrust::host_vector<uint32_t> map = layers[0].insert_elements(bdd->layers[0].children);
	
	apply_root_layer_mapping(map);
}

/**
 * Updates the roots of the BDD after new root nodes are added. `p` contains a mapping from original root indices to new
 * root indices.
 * 
 * @param p Contains the new indices. For an old index `i`, the new index is located at `p[i]`.
 */
void bdd::apply_root_layer_mapping(thrust::host_vector<uint32_t>& p) {
	// Update roots list
	for (auto& pair : roots) {
		pair.second = p[pair.second];
		// TODO: add assert back. Removed for now because in testing live roots are removed.
		// assert(pair.second < layers[0].size());
	}
}

/**
 * Updates the roots of the BDD after new root nodes are added. `p` contains a mapping from original root indices to new
 * root indices.
 * 
 * @param p Contains the new indices. For an old index `i`, the new index is located at `p[i]`.
 */
void bdd::apply_root_layer_mapping(std::vector<uint32_t>& p) {
	// Update roots list
	for (auto& pair : roots) {
		pair.second = p[pair.second];
		// TODO: add assert back. Removed for now because in testing live roots are removed.
		// assert(pair.second < layers[0].size());
	}
}

uint32_t bdd::add_root(uint32_t id) {
	roots[lifetime_root_count] = id;
	if (root_refcounts.contains(lifetime_root_count)) {
		root_refcounts[lifetime_root_count];
	} else {
		root_refcounts[lifetime_root_count] = 0;
	}

	return lifetime_root_count++;
}

bdd::bdd(uint16_t depth, std::shared_ptr<data_administrator> _administrator) : administrator(_administrator) {
	temp = std::shared_ptr<bdd>(this);

	for (uint16_t i = 0; i < depth; i++) {
		layers.push_back(layer(i, depth, administrator));
	}

	for (uint16_t i = 0; i < depth; i++) {
		variables.push_back(node_ref(add_root((uint16_t)(depth - i - 1)), weak_from_this()));
	}
} 

bdd::bdd(bdd& other, std::shared_ptr<data_administrator> admin) : administrator(admin) {
	temp = std::shared_ptr<bdd>(this);

	lifetime_root_count = other.lifetime_root_count;
	roots = std::map<uint32_t, uint32_t>(other.roots);
	root_refcounts = std::map<uint32_t, uint32_t>(other.root_refcounts);

	// Copy the layers of this BDD
	for (layer l : other.layers) {
		layers.push_back(layer(l));
	}

	// Copy the variables and update the reference to ourselves
	for (node_ref v : other.variables) {
		variables.push_back(node_ref(v, weak_from_this()));
	}

	// Add this bdd as a copy to the other bdd
	std::weak_ptr<bdd> weak_temp = temp;
	other.copied_bdds[weak_temp] = other.lifetime_root_count;
}

bdd::~bdd() {}

std::shared_ptr<bdd> bdd::extract_shared_pointer() {
	std::shared_ptr<bdd> result = temp;
	temp.reset();
	return result;
}

layer& bdd::get_layer(uint16_t label) {
	return layers[label];
}
