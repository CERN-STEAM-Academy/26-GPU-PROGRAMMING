#include "product_layer.h"
#include "../arc_layer/arc_layer.h"

#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <iostream>

using namespace sro;

/**
 * Constructs an empty product layer.
 * 
 * @param _label The label of this layer, i.e. how deep this layer is in the BDD. 0 is the root layer, and higher labels 
 * are for layers towards the terminals.
 * 
 * @param admin The data_administrator which arranges the data location and policy for the products in this layer.
 */
product_layer::product_layer(uint16_t _label, std::shared_ptr<data_administrator> admin) : label(_label), administrator(admin), products(admin) {}

/**
 * Constructs a product layer that contains one product, given by parameter `product`.
 * 
 * @param _label The label of this layer, i.e. how deep this layer is in the BDD. 0 is the root layer, and higher labels 
 * are for layers towards the terminals.
 * 
 * @param product The node_product to populate this product layer with.
 * 
 * @param admin The data_administrator which arranges the data location and policy for the products in this layer.
 */
product_layer::product_layer(uint16_t _label, node_product product, std::shared_ptr<data_administrator> admin) : label(_label), administrator(admin), products(admin) {
	products.administrator = administrator;
	products.push_back(product);
}

struct divide_by_two{
	__host__ __device__
	uint32_t operator() (uint32_t i) {
		return i / 2;
	}
};

struct modulo_two{
	__host__ __device__
	uint32_t operator() (uint32_t i) {
		return i % 2;
	}
};

struct retrieve_low_high_depending_on_even{
	__host__ __device__
	node_product operator() (arc a, uint32_t is_even) {
		return is_even ? a.low : a.high;
	}
};

struct sort_product_low_high{
	__host__ __device__
	bool operator() (node_product a, node_product b) {
		return thrust::make_tuple(a.low, a.high) < thrust::make_tuple(b.low, b.high);
	}
};

/**
 * Construct a product layer from an arc layer. Each arc generates two products.
 * 
 * @param in The input arc layer. Each arc in the arc layer gets converted to two products. The number of elements in the 
 * product layer is at most twice the number of elements of the arc layer. After generation, the products are sorted, and
 * duplicates are removed.
 * 
 * @post administrator == in.administrator
 * 
 * @post ∀ sro::arc a ∈ in.arcs : ∃ sro::node_product p ∈ products : p = a.low
 * 
 * @post ∀ sro::arc a ∈ in.arcs : ∃ sro::node_product p ∈ products : p = a.high
 * 
 * @post ∀ sro::node_product p ∈ products : ∃ sro::arc a ∈ in.arcs : (p = a.low || p == a.high)
 * 
 * @post sorted(products, sort_product_low_high)
 * 
 * @post unique(products)
 */
product_layer::product_layer(arc_layer& in) : label(in.label), administrator(in.administrator), products(in.administrator) {
	products.to_location(in.get_data_location());
	products.resize(2 * in.arcs.size());

	products.with_vector([](auto policy, auto& v, auto& arcs_v) {
		// Compiler computes types
		using Vec = std::remove_reference_t<decltype(v)>;
		using ArcVec = rebind_vector_t<Vec, arc>;
		
		// Define iterators
		auto count = thrust::make_counting_iterator<uint32_t>(0);
		auto map = thrust::make_transform_iterator(count, divide_by_two());
		auto arc_iterator = arcs_v.begin();
		auto perm = thrust::make_permutation_iterator(arc_iterator, map);
		auto mod = thrust::make_transform_iterator(count, modulo_two());

		thrust::transform(
			policy,
			perm, perm + v.size(),
			mod,
			v.begin(),
			retrieve_low_high_depending_on_even()
		);
	}, in.arcs);

	// Ensure uniqueness
	products.sort(sort_product_low_high());
	products.unique();
}

struct to_removable_node_product{
	__host__ __device__
	node_product operator() (uint32_t in) {
		return {in, in};
	}
};

/**
 * Constructs a product layer that contains removable (i.e. same low and high child) products for all indices i where 
 * participants[i] is true. Used in quantifiers, where a logical operation is initiated at a lower level of the BDD.
 * 
 * @param in The base layer. Only used to get the label and the administrator.
 * 
 * @param participants The data_manager containing the participants. False means the index is ignored, True means it is 
 * included.
 */
product_layer::product_layer(layer& in, data_manager<bool> participants) : label(in.get_label()), administrator(in.administrator), products(in.administrator) {
	products.to_location(participants.get_data_location());
	products.resize(participants.size());
	products.administrator = administrator;
	products.with_vector([](auto policy, auto& products_v, auto& participants_v){
		auto counting_iterator = thrust::counting_iterator<uint32_t>();
		auto node_product_iterator = thrust::make_transform_iterator(counting_iterator, to_removable_node_product());
		auto it = thrust::copy_if(
			policy,
			node_product_iterator, node_product_iterator + participants_v.size(),
			participants_v.begin(),
			products_v.begin(),
			::cuda::std::identity()
		);
		products_v.resize(it - products_v.begin());

	}, participants);
}

/**
 * Get the products of this product_layer as an std::vector< node_product >.
 * 
 * @returns node_products in this product_layer.
 */
std::vector<node_product> product_layer::get_data() {
	thrust::host_vector<node_product> result = products;
	return std::vector<node_product>(result.begin(), result.end());
}

/**
 * Sets the products contained in this product layer directly.
 * 
 * @param data The data to replace the resident node products with.
 */
void product_layer::set_data(std::vector<node_product> data) {
	products = data;
}

/**
 * Prints the node products in this layer. The low edges are printed first, then below that the high edges are printed. 
 * Spacing is arranged by tabs.
 */
void product_layer::print() {
	printf("Low: \t\t");
	for (node_product p : products) {
		printf("%5u ", p.low);
	}
	printf("\n");
	printf("High: \t\t");
	for (node_product p : products) {
		printf("%5u ", p.high);
	}
	printf("\n");
} 