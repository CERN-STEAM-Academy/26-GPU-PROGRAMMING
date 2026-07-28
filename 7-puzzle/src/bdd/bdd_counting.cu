// Functions to facilitate counting BDD states, nodes, and printing information about the BDD.

#include "bdd.h"

#if USE_STORAGE
#include "../spill/spill_policy.h" 
#endif

using namespace sro;

/**
 * Count the number of satisfying assignments. A satisfying assignment is a sequence of inputs for which the output of the
 * BDD is true. If the number of satisfying assignments is 0, the BDD is unsatisfiable. If it is equal to `2^n` with n the
 * number of variables, the BDD is valid. 
 * 
 * @returns std::vector< uint64_t > containing the respective satisfying assignment count for each root.
 */
std::vector<uint64_t> bdd::count_satisfying_assignments() {
    data_manager<uint64_t> buf(layers.back().get_data_location(), 2, administrator);
    buf.with_vector([&](auto policy, auto& v){ v = {0, 1}; });

    uint16_t l = layers.size();
    while (true) {
        l--;

		#if USE_STORAGE
        bool was_spilled = layers[l].is_spilled();         // check before it gets restored
        #endif
		
		buf = layers[l].satisfying_assignments(buf);       // restores layer l
        
		#if USE_STORAGE
		if (was_spilled) layers[l].set_data_location(STORAGE_GDS);  // re-spill it
        #endif

		if (l == 0) break;
    }
    thrust::host_vector<uint64_t> result = buf;
    return std::vector<uint64_t>(result.begin(), result.end());
}

/**
 * Returns a list of sro::node_ref objects referring to the roots in this bdd. Can be used in conjunction with 
 * bdd::count_satisfying_assignments to get the satisfying assignment count of the correct node.
 * 
 * @returns std::vector< node_ref > of all nodes in the BDD.
 */
std::vector<node_ref> bdd::get_all_node_refs() {
	std::vector<node_ref> result = std::vector<node_ref>();
	for (auto [key, value] : roots) {
		result.push_back(node_ref(key, this->weak_from_this()));
	}
	return result;
}

/**
 * Returns the total number of nodes in the BDD. This number may differ from the one reported by conventional BDD libraries
 * as GPUdecide uses Quasi-Reduced Ordered Binary Decision Diagrams (QROBDD).
 * 
 * @returns The total number of nodes in the BDD.
 */
uint64_t bdd::count_nodes() {
	uint64_t result = 0;

	for (layer& l : layers) {
		result += l.node_count();
	}
	return result;
}

/**
 * Returns the total number of irreducible nodes in the BDD. A node is irreducible if the low child is different from the 
 * high child. The result should match the one reported by conventional BDD libraries operating on ROBDDs, if the BDD only 
 * contains one root.
 * 
 * @returns The total number of irreducible nodes.
 */
uint64_t bdd::count_irreducible_nodes() {
	#if USE_STORAGE
	if (spilling_enabled()) return 0;
	#endif
	
	uint64_t result = 0;

	for (layer& l : layers) {
		result += l.irreducible_node_count();
	}
	return result;
}

/**
 * Prints the contents of the layers of the binary decision diagram. 
 */
void bdd::print() {
	for (layer& l : layers) {
		l.print();
		printf("\n");
	}
}

/**
 * Prints the roots of this BDD.
 */
void bdd::print_roots() {
	printf("Key:\t\t");
	for (auto [key, value] : roots) {
		printf("%u\t", key);
	}
	printf("\n");

	printf("Index:\t\t");
	for (auto [key, value] : roots) {
		printf("%u\t", value);
	}
	printf("\n");

	printf("Count:\t\t");
	for (auto [key, value] : root_refcounts) {
		printf("%u\t", value);
	}
	printf("\n");
}
