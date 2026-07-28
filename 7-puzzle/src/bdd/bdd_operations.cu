#include "bdd.h"
#include "../product_layer/product_layer.h"
#include "../semi_arc_layer/semi_arc_layer.h"

#if USE_STORAGE
#include "../spill/spill_policy.h" 
#endif

#include "../profiling/mem_profiler.h"
using namespace sro;

struct apply_and{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return a && b;
	}
};

struct apply_nand{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return !(a && b);
	}
};

struct apply_or{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return a || b;
	}
};

struct apply_nor{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return !(a || b);
	}
};

struct apply_xor{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return a != b;
	}
};

struct apply_eq{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return a == b;
	}
};

struct apply_difference{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return a && !b;
	}
};

struct apply_implication{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return !a || b;
	}
};

struct apply_not{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return !a;
	}
};

struct apply_true{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return true;
	}
};

struct apply_false{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return true;
	}
};

template<typename binary_operation>
node_ref bdd::quantifier(node_ref a, std::vector<uint32_t> variables, binary_operation op) {
	std::sort(variables.rbegin(), variables.rend());
	// Encode relevant node
	std::vector<data_manager<bool>> relevant_nodes;
	relevant_nodes.emplace_back(HOST, layers[0].size(), false, administrator);
	relevant_nodes[0][a.get_id()] = true;

	// Propagate stencil up to the first quantified variable
	for (int i = 0; i < variables[0]; i++) {
		relevant_nodes.push_back(layers[i].propagate_encoding(relevant_nodes.back(), layers[i + 1]));
	}

	data_manager<node_product> map(layers[variables[0]].get_data_location(), administrator);

	// Propagate down to the quantified layer
	for (uint32_t v_idx = 0; v_idx < variables.size(); v_idx++) {
		uint32_t v = variables[v_idx];

		// First we propagate down to the terminal nodes
		std::vector<arc_layer> arc_layers;
		arc_layers.reserve(layers.size());      // no reallocation during the sweep
		{
			product_layer l(layers[v], relevant_nodes.back()); // Product layer contains all products that need to be explored
			arc_layers.push_back(arc_layer(l, layers[v], true));  // Arc layer amends this information with the low/high child products
		}
		
		#if USE_STORAGE
		SpillPolicy policy;
		policy.track_arc_layer(arc_layers.back().size());
		#endif

		for (uint16_t i = v + 1; i < layers.size(); i++) {
			product_layer generated_products(arc_layers.back());

			#if USE_STORAGE
			policy.place_persistent(layers[i]);
			#else
			layers[i].set_data_location(administrator->get_data_location({layers[i].size(), layers[i].get_data_location()}));
			#endif

			arc_layers.push_back(arc_layer(generated_products, layers[i]));

			#if USE_STORAGE
			policy.track_arc_layer(arc_layers.back().size());

			policy.maybe_spill_persistent(layers[i]); // spill dormant persistent layer if over budget
			if (arc_layers.size() >= 2)
				policy.maybe_spill_arc(arc_layers[arc_layers.size() - 2]); // spill completed arc layer if over budget
			#endif
		}

		{
			// At the terminal nodes, we apply the logical operation. 
			// This collapses the products to singlar terminal nodes 0 and 1.
			semi_arc_layer semi_arcs(arc_layers.back(), op);
			
			// After this, we propagate back up to the node where we came from
			for (uint16_t i = layers.size() - 1; i > v; i--) {
				arc_layers.pop_back();
				// Semi arcs have the product of the parent, and the low and high edges/indices of the children.
				// Included in the constructor of the semi arcs is the insertion of the new nodes and updating of indices of parent nodes.
				semi_arcs = semi_arc_layer(
					arc_layers.back(), // Arcs that are to be converted to semi arcs.
					semi_arcs, // Parent semi arcs. Used to map the low and high products of the arcs to indices.
					layers[i], // Child layer. New children will be inserted here.
					layers[i - 1] // Parent layer.
				);

				#if USE_STORAGE
				policy.maybe_respill(layers[i]);
				#endif
			}

			relevant_nodes.pop_back();

			if (v > 0) {
				map = semi_arcs.merge_semi_arcs_with_quantifier_layer(layers[v], layers[v - 1], relevant_nodes.back());
			}
			else {
				semi_arcs.merge_semi_arcs_with_quantifier_layer(layers[v], *this);
				auto result = node_ref(lifetime_root_count - 1, weak_from_this());
				return result;
			}
		}
		// TODO: Propagate upwards to next variable, or to 0.
		uint32_t target = (v_idx == variables.size() - 1) ? 0 : variables[v_idx + 1];
		for (int i = v - 1; i > target; i--) {
			assert(i < relevant_nodes.size());
			map = layers[i].propagate_quantifier_up(layers[i - 1], map, relevant_nodes[i - 1], relevant_nodes[i]);
			relevant_nodes.pop_back();
		}
	}

	thrust::host_vector<uint32_t> inserted_map = layers[0].insert_elements(map);

	apply_root_layer_mapping(inserted_map);

	uint32_t id = layers[0].get_index_of(map[0]);
	// return node_ref(lifetime_root_count - 1, weak_from_this());
	return node_ref(add_root(id), weak_from_this());
}

template<typename binary_operation>
node_ref bdd::quantifier(node_ref a, uint32_t v, binary_operation op) {
	std::vector<uint32_t> vec = {v};
	return quantifier(a, vec, op);
}

template<typename binary_operation>
node_ref bdd::logical_operation(node_ref a, node_ref b, binary_operation op) {
	false_cache_up_to_date = false;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Record the start event
	cudaEventRecord(start, 0);
	sro::PeakMemSampler mem_sampler; 

	std::vector<arc_layer> arc_layers;
	arc_layers.reserve(layers.size());

	node_product p = {roots[a.root_id], roots[b.root_id]};
	{
		product_layer products(0, p, administrator); // Product layer contains all products that need to be explored
		arc_layers.push_back(arc_layer(products, layers[0])); // Arc layer amends this information with the low/high child products
	}

	#if USE_STORAGE
	SpillPolicy policy;
    policy.track_arc_layer(arc_layers.back().size());
	#endif

	for (uint16_t i = 1; i < layers.size(); i++) {
		product_layer generated_products(arc_layers.back());

		#if USE_STORAGE
		policy.place_persistent(layers[i]);
		#else
		layers[i].set_data_location(administrator->get_data_location({layers[i].size(), layers[i].get_data_location()}));
		#endif
		
		arc_layers.push_back(arc_layer(generated_products, layers[i]));

		#if USE_STORAGE
		policy.track_arc_layer(arc_layers.back().size());

		policy.maybe_spill_persistent(layers[i]);
        if (arc_layers.size() >= 2)
            policy.maybe_spill_arc(arc_layers[arc_layers.size() - 2]);
		#endif
	}

	#if USE_STORAGE
	policy.report(arc_layers);
	#endif

	// Monitoring the total amount of arc_nodes after propagating down
	size_t free_bytes = 0;
	size_t total_bytes = 0;
	cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
	uint64_t arc_nodes = 0;
	for (arc_layer& l : arc_layers) {
		arc_nodes += l.size();
	}

	{
		// At the terminal nodes, we apply the logical operation. 
		// This collapses the products to singlar terminal nodes 0 and 1.
		semi_arc_layer semi_arcs(arc_layers.back(), op);
		// After this, we propagate back up to the root nodes
		for (uint16_t i = layers.size() - 1; i > 0; i--) {
			arc_layers.pop_back();
			semi_arcs = semi_arc_layer(
				arc_layers.back(),
				semi_arcs,
				layers[i],
				layers[i - 1]
			);

			#if USE_STORAGE
			policy.maybe_respill(layers[i]);
			#endif
		}
		semi_arc_layer(semi_arcs, layers[0], *this);
	}
	// Record the stop event
	cudaEventRecord(stop, 0);

	if (status == cudaSuccess && arc_nodes > 10000) {
		// Wait for the stop event to complete
		cudaEventSynchronize(stop);
		float elapsed_ms = 0;
		cudaEventElapsedTime(&elapsed_ms, start, stop);
		print_op_stats(mem_sampler, free_bytes, total_bytes, arc_nodes, elapsed_ms,
		               count_nodes(), [&]{ return count_irreducible_nodes(); });
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return node_ref(lifetime_root_count - 1, weak_from_this());
}

node_ref bdd::exists(node_ref a, uint32_t v) {
	return quantifier(a, v, apply_or());
}

node_ref bdd::exists(node_ref a, std::vector<uint32_t> variables) {
	validate_root(a);
	for (const auto variable : normalize_variables(std::move(variables))) a = exists_reference(a, variable);
	return a;
}

node_ref bdd::for_all(node_ref a, uint32_t v) {
	return quantifier(a, v, apply_and());
}

node_ref bdd::for_all(node_ref a, std::vector<uint32_t> variables) {
	validate_root(a);
	variables = normalize_variables(std::move(variables));
	if (variables.empty()) return a;
	return logical_not(exists(logical_not(a), std::move(variables)));
}

#include <set>

node_ref bdd::logical_and(node_ref a, node_ref b) {
	node_ref result = logical_operation(a, b, apply_and());
	if ((roots[result.root_id] > roots[a.root_id]) || (roots[result.root_id] > roots[b.root_id])) {
	    throw std::runtime_error("Order of BDD elements not consistent with operation.");
	}
	// assert(result.get_id() <= a.get_id());
	// assert(result.get_id() <= b.get_id());
	return result;
}

node_ref bdd::logical_nand(node_ref a, node_ref b) {
	return logical_operation(a, b, apply_nand());
}

node_ref bdd::logical_or(node_ref a, node_ref b) {
	return logical_operation(a, b, apply_or());
}

node_ref bdd::logical_nor(node_ref a, node_ref b) {
	return logical_operation(a, b, apply_nor());
}

node_ref bdd::logical_xor(node_ref a, node_ref b) {
	return logical_operation(a, b, apply_xor());
}

node_ref bdd::logical_eq(node_ref a, node_ref b) {
	return logical_operation(a, b, apply_eq());
}

node_ref bdd::logical_difference(node_ref a, node_ref b) {
	return logical_operation(a, b, apply_difference());
}

node_ref bdd::logical_implication(node_ref a, node_ref b) {
	return logical_operation(a, b, apply_implication());
}

node_ref bdd::logical_not(node_ref a) {
	return logical_operation(a, a, apply_not());
}

node_ref bdd::logical_ite(node_ref i, node_ref t, node_ref e) {
	node_ref if_then = logical_and(i, t);
	node_ref if_not_then = logical_difference(e, i);
	return logical_or(if_then, if_not_then);
}

node_ref bdd::logical_true() {
	return logical_operation(variables[0], variables[0], apply_true());
}

node_ref bdd::logical_false() {
	return logical_operation(variables[0], variables[0], apply_false());
}

node_ref bdd::get_variable(uint16_t label) {
	return variables[label];
}
