// Bdd operations for input and output of dddmp files

#include <ranges>
#include <fstream>
#include "bdd.h"

#include <thrust/reverse.h>


using namespace sro;

/**
 * Construct a BDD using an input stream of dddmp format. This function converts the ROBDD in the input stream to a 
 * QROBDD.
 * 
 * @param stream The input stream containing the BDD in dddmp format.
 */
bdd::bdd(std::istream& stream) {
	std::string ver_p, ver;
	stream >> ver_p >> ver;
	assert(ver_p == ".ver");

	std::string mode_p, mode;
	stream >> mode_p >> mode;
	assert(mode_p == ".mode");

	std::string varinfo_p;
	int varinfo;
	stream >> varinfo_p >> varinfo;
	assert(varinfo_p == ".varinfo");

	std::string nnodes_p;
	int nnodes;
	stream >> nnodes_p >> nnodes;
	assert(nnodes_p == ".nnodes");

	std::string nvars_p;
	int nvars;
	stream >> nvars_p >> nvars;
	assert(nvars_p == ".nvars");

	std::string nsuppvars_p;
	int nsuppvars;
	stream >> nsuppvars_p >> nsuppvars;
	assert(nsuppvars_p == ".nsuppvars");

	std::string ids_p;
	std::string ids_s;
	std::vector<uint32_t> ids;
	stream >> ids_p;
	assert(ids_p == ".ids");
	stream.seekg(1, std::ios::cur); // Skip whitespace
	std::getline(stream, ids_s);
    for (auto part : std::views::split(ids_s, ' ')) {
		std::string result =  std::string(part.begin(), part.end());
		ids.push_back(std::stoul(result));
    }

	std::string permids_s;
	std::getline(stream, permids_s);

	std::string nroots_p;
	int nroots;
	stream >> nroots_p >> nroots;
	assert(nroots_p == ".nroots");

	std::string rootids_p;
	std::string rootids_s;
	std::vector<uint32_t> rootids;
	stream >> rootids_p;
	assert(rootids_p == ".rootids");
	stream.seekg(1, std::ios::cur); // Skip whitespace
	std::getline(stream, rootids_s);
    for (auto part : std::views::split(rootids_s, ' ')) {
        std::string result =  std::string(part.begin(), part.end());
		rootids.push_back(std::stoul(result));
    }

	std::string nodes_p;
	stream >> nodes_p;
	assert(nodes_p == ".nodes");

	std::string f;
	std::getline(stream, f);
	std::getline(stream, f);
	assert(f == "1 F 0 0");

	std::string t;
	std::getline(stream, t);
	assert(t == "2 T 0 0");

	// Construct BDD
	temp = std::shared_ptr<bdd>(this);

	// Fill with nodes
	uint32_t id, old_id;
	uint32_t old_layer = 0xFFFFFFFF;

	std::vector<node_product> values;
	std::vector<uint32_t> layer_starts = {1};
	std::vector<uint32_t> skipped_layers;
	bool first_layer = true;

	old_id = 2;
	while (stream >> id) {
		uint32_t layer, high, low;
		stream >> layer >> high >> low;
		if (old_layer != layer) {
			// Move parsed products into new layer
			if (!first_layer){
				layers.emplace(layers.begin(), administrator);
				layers[0].children = values;
				values = std::vector<node_product>();

				// Check if layers are being skipped. If so, we can add removable layers later.
				for (int i = layer + 1; i < old_layer; i++) {
					skipped_layers.push_back(i);
				}
			}
			first_layer = false;

			// Set the starting point of the layer
			layer_starts.insert(layer_starts.begin(), id);
			old_layer = layer;
		}

		values.push_back({low, high});
		assert(id == old_id + 1); // Make sure that the ids are consecutive.
		old_id = id;
	}

	printf("Layer size: %lu\n", layers.size());

	// Create final layer
	layers.emplace(layers.begin(), administrator);
	
	// Move parsed products to layer
	layers[0].children = values;

	printf("Layer size: %lu\n", layers.size());

	// Propagate down to replace with layer-local indices
	std::map<uint32_t, uint32_t> removable_node_to_idx;
	std::vector<uint32_t> removable_nodes = rootids;

	// Initialize removable nodes with rootids not in the root layer
	auto result = std::remove_if(
		removable_nodes.begin(), removable_nodes.end(),
		[&](uint32_t in) {
			return in >= layer_starts[0];
		});
	removable_nodes.resize(result - removable_nodes.begin());

	for (int i = 0; i < layers.size(); i++) {
		uint32_t boundary = layer_starts[i + 1];
		
		for (int j = 0; j < layers[i].children.size(); j++) {
			node_product ref = layers[i].children[j];
			for (uint32_t& edge : {std::ref(ref.high), std::ref(ref.low)}) {
				if (edge >= boundary) {
					edge -= boundary;
				}
				else {
					if (!removable_node_to_idx.contains(edge)) {
						removable_node_to_idx[edge] = removable_node_to_idx.size();
					}
					// Replace global index with local removable node index one layer down
					int size = layer_starts[i] - boundary;
					edge = removable_node_to_idx[edge] + size; // Append nodes to the end
				}
			}
			layers[i].children[j] = ref;
		}

		// Add removable nodes to layer
		uint32_t children_size = i == layers.size() - 1 ? 2 : layers[i + 1].children.size();
		for (int j = 0; j < removable_nodes.size(); j++) {
			uint32_t target;
			if (removable_nodes[j] >= boundary) {
				target = removable_nodes[j] - boundary;
			}
			else {
				target = removable_node_to_idx[removable_nodes[j]] + children_size; // Add 
			}
			node_product p = {target, target};
			layers[i].children.push_back(p);
		}

		// Manage removable nodes
		removable_nodes.resize(removable_node_to_idx.size());
		for (const auto& [key, value] : removable_node_to_idx) {
			removable_nodes[value] = key;
		}

		std::vector<uint32_t> filtered_removable_nodes(removable_nodes);
		auto result = std::remove_if(
			filtered_removable_nodes.begin(), filtered_removable_nodes.end(), 
			[&](uint32_t in) {
				return in >= boundary;
			});
		filtered_removable_nodes.resize(result - filtered_removable_nodes.begin());
		
		removable_node_to_idx = std::map<uint32_t, uint32_t>();
		for (int i = 0; i < filtered_removable_nodes.size(); i++) {
			removable_node_to_idx[filtered_removable_nodes[i]] = i;
		}
	}
	assert(removable_node_to_idx.size() == 0);

	// Propagate node order sorted on high
	data_manager<uint32_t> map(HOST, 2, administrator); // Number of elements is equal to number of terminal nodes
	map[0] = 0;
	map[1] = 1;

	for (uint16_t i = layers.size(); i-- > 0;) {
		map = layers[i].apply_map(map);
	}

	// Add root references
	uint32_t first_layer_boundary = rootids.size();
	for (int i = 0; i < rootids.size(); i++) {
		if (rootids[i] < layer_starts[0]) {
			first_layer_boundary = i;
			break;
		}
	}

	// Elements in first layer have been added backwards. Undo this.
	map.with_vector([&](auto policy, auto& v){
		thrust::reverse(v.begin(), v.begin() + first_layer_boundary);
	});

	std::vector<node_ref> refs;
	for (uint32_t m : map) {
		refs.push_back(node_ref(add_root(m), weak_from_this()));
	}
	
	// Uncomment this for testing
	// std::ofstream outFile("2var3.dddmp");
	// output_dddmp(outFile, refs);
}

/**
 * Construct a BDD from an input stream.
 * 
 * @param stream The input stream containing the BDD in dddmp format.
 * 
 * @returns std::vector< sro::node_ref > containing the root nodes of the imported BDDs.
 */
std::vector<node_ref> bdd::input_dddmp(std::istream& stream) {
	std::shared_ptr<bdd> result = (new bdd(stream))->extract_shared_pointer();
	
	printf("Layer sizes: %lu %lu\n", layers.size(), result->layers.size());
	
	// Merge BDD with self.
	join_with(result);
	
	return std::vector<node_ref>(); 
}

struct count_irreducible{
	__host__ __device__
	bool operator() (node_product p, bool b) {
		return (!p.removable()) && b;
	}
};

struct add_n{
	uint32_t n;
	add_n(uint32_t _n) : n(_n) {}

	__host__ __device__
	uint32_t operator() (uint32_t in) {
		return in + n;
	} 
};

/**
 * Output a BDD in dddmp format to an output stream. Only includes the nodes referenced by `refs`.
 * 
 * @param stream The output stream to output dddmp to.
 * 
 * @param refs List of sro::node_ref objects referring to roots to be exported.
 */
void bdd::output_dddmp(std::ostream& stream, std::vector<node_ref> refs) {
	uint32_t number_of_roots = layers[0].size();

	// Masks for relevant nodes
	data_manager<bool> stencil(layers[0].get_data_location(), layers[0].size(), false, administrator);
	for (node_ref ref : refs) {
		stencil[ref.get_id()] = true;
	}

	std::vector<data_manager<bool>> stencils;
	std::vector<uint32_t> stencil_counts;

	stencils.push_back(stencil); // Stencil for relevant nodes at each layer

	for (int i = 0; i < layers.size(); i++) {
		// Count the number of relevant nodes
		uint32_t stencil_count = stencil.with_vector([](auto policy, auto& v, auto& children){
			auto it = thrust::make_zip_iterator(children.begin(), v.begin());
			return thrust::count_if(
				policy,
				it, it + v.size(),
				thrust::make_zip_function(count_irreducible())
			);
		}, layers[i].children);
		stencil_counts.push_back(stencil_count);

		// Propagate stencil downwards
		if (i < layers.size() - 1) {
			stencil = layers[i].propagate_encoding(stencil, layers[i + 1]);
			stencil.to_location(layers[i + 1].children.get_data_location());
			stencils.push_back(stencil);
		}
	}

	// Cumulative counts
	std::vector<uint32_t> cumulative_stencil_counts;
	cumulative_stencil_counts.push_back(3); // The first layer is transient and has two elements, true and false
	for (auto count : stencil_counts | std::views::reverse) {
		cumulative_stencil_counts.push_back(cumulative_stencil_counts.back() + count);
	}

	// Prefix sum of mask for layer sizes
	// This results in real indices of the dddmp file
	std::vector<data_manager<uint32_t>> real_indices;
	for (int i = 0; i < layers.size(); i++) {
		real_indices.emplace_back(layers[i].get_data_location(), layers[i].size(), administrator);

		real_indices.back().with_vector([&](auto policy, auto& v, auto& s, auto& children){
			auto it = thrust::make_zip_iterator(children.begin(), s.begin());
			auto irreducible_stencil_it = thrust::make_transform_iterator(it, thrust::make_zip_function(count_irreducible()));
			thrust::exclusive_scan(
				policy,
				irreducible_stencil_it, irreducible_stencil_it + s.size(),
				v.begin(),
				0,
				thrust::plus<uint32_t>()
			);

			thrust::transform(
				policy,
				v.begin(), v.end(),
				v.begin(),
				add_n(cumulative_stencil_counts[layers.size() - i - 1])
			);
		}, stencils[i], layers[i].children);
	}


	// Terminal layer
	real_indices.emplace_back(data_location::HOST, 2, administrator);
	real_indices.back()[0] = 1;
	real_indices.back()[1] = 2;

	// Root ids
	std::vector<uint32_t> root_ids;
	for (node_ref ref : refs) {
		uint32_t id = ref.get_id();
		uint16_t layer = 0;
		node_product n = layers[layer].children[id];
		while (n.removable()){
			id = n.high;
			layer++;
			if (layer < layers.size()){
				n = layers[layer].children[id];
			}
			else {
				break;
			}
		}
		root_ids.push_back(real_indices[layer][id]);
	}

	// Version
	stream << ".ver DDDMP-2.0\n";

	// Mode
	stream << ".mode A\n";

	// Varinfo
	stream << ".varinfo 4\n";

	// Number of nodes
	uint32_t node_count = cumulative_stencil_counts.back() - 1;
	stream << ".nnodes " << node_count << "\n";

	// Number of variables
	stream << ".nvars " << layers.size() << "\n";
	stream << ".nsuppvars " << layers.size() << "\n";

	// Ids of variables
	stream << ".ids";
	for (int i = 0; i < layers.size(); i++) {
		stream << " " << i;
	}
	stream << "\n";

	stream << ".permids";
	for (int i = 0; i < layers.size(); i++) {
		stream << " " << i;
	}
	stream << "\n";

	// Number of roots
	stream << ".nroots " << refs.size() << "\n";

	// Number of nodes (we add 2 because the final nodes are not instantiated)
	stream << ".rootids";
	for (uint32_t root_id : root_ids)
		stream << " " << root_id;
	stream << "\n";

	// True and False terminal nodes
	stream << ".nodes\n";
	stream << "1 F 0 0\n";
	stream << "2 T 0 0\n";

	int idx = 3;

	// non-trivial nodes
	for (uint16_t i = layers.size(); i-- > 0;) {
		// // Sort nodes on low
		// layers[i].children.with_vector([](auto policy, auto& v, auto& ri, auto& s){
		// 	auto it = thrust::make_zip_iterator(ri.begin(), s.begin());
		// 	thrust::sort_by_key(
		// 		policy,
		// 		v.begin(), v.end(),
		// 		it,
		// 		sort_node_product_low_high()
		// 	);
		// }, real_indices[i], stencils[i]);

		// Append node to stream
		for (int j = 0; j < real_indices[i].size(); j++) {
			if (stencils[i][j]) {
				node_product n = layers[i].children[j];
				if (n.removable()) {
					real_indices[i][j] = real_indices[i+1][n.high];
				}
				else {
					uint32_t high = real_indices[i+1][n.high];
					uint32_t low = real_indices[i+1][n.low];
					assert(n.high != n.low);
					if (i + 1 < stencils.size()){
						assert(stencils[i+1][n.high]);
						assert(stencils[i+1][n.low]);
					}
					assert(high != low);
					stream << idx++ << " " << i << " " << high << " " << low << "\n";
				}
			}
		}
		
		// // Sort nodes on high
		// layers[i].children.with_vector([](auto policy, auto& v, auto& ri, auto& s){
		// 	auto it = thrust::make_zip_iterator(ri.begin(), s.begin());
		// 	thrust::sort_by_key(
		// 		policy,
		// 		v.begin(), v.end(),
		// 		it,
		// 		sort_node_product_high_low()
		// 	);
		// }, real_indices[i], stencils[i]);
	}
	
	stream << ".end\n";
}