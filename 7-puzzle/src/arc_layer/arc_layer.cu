#include "arc_layer.h"
#include "../product_layer/product_layer.h"

#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <vector>

using namespace sro;

arc_layer::arc_layer(uint16_t _label, std::shared_ptr<data_administrator> admin) : label(label), administrator(admin), arcs(admin) {}

struct sort_product_on_f{
	__host__ __device__
	bool operator() (node_product a, node_product b) {
		return thrust::make_tuple(a.low, a.high) < thrust::make_tuple(b.low, b.high); 
	}
};

struct transform_product_to_f{
	__host__ __device__
	uint32_t operator() (node_product in) {
		return in.low;
	}
};

struct transform_parents_and_f_to_arcs{
	__host__ __device__
	arc operator() (node_product parent, node_product f) {
		return {
			parent.low, 	// Parent f
			parent.high,	// Parent g
			f.low, 			// Low child f
			0,				// Low child g
			f.high, 		// High child f
			0  				// High child g
		};
	}
};


struct transform_parents_and_f_to_arcs_quantifier{
	__host__ __device__
	arc operator() (node_product parent, node_product f) {
		return {
			parent.low, 	// Parent f
			parent.high,	// Parent g
			f.low, 			// Low child f
			f.high,			// Low child g
			0,		 		// High child f
			0  				// High child g
		};
	}
};


struct sort_arc_on_g{
	__host__ __device__
	bool operator() (arc a, arc b) {
		auto a_t = thrust::make_tuple(a.parent.high, a.parent.low, a.high.high, a.high.low, a.low.high, a.low.low);
		auto b_t = thrust::make_tuple(b.parent.high, b.parent.low, b.high.high, b.high.low, b.low.high, b.low.low);
		return a_t < b_t;
	}
};

struct transform_arc_parent_to_high{
	__host__ __device__
	uint32_t operator() (arc in) {
		return in.parent.high;
	}
};

struct fill_arc_g{
	__host__ __device__
	arc operator() (node_product layer, arc a) {
		a.low.high = layer.low;
		a.high.high = layer.high;
		return a;
	}
};

struct fill_arc_g_quantifier{
	__host__ __device__
	arc operator() (node_product layer, arc a) {
		a.high = layer;
		return a;
	}
};

arc_layer::arc_layer(product_layer& in, layer& layer, bool quantifier) : label(in.label + 1), administrator(in.administrator), arcs(in.administrator) {
	in.products.to_location(layer.get_data_location());
	set_data_location(layer.get_data_location());

	// Sort products on low
	in.products.sort(sort_product_on_f());

	// TODO: make sure in and layer are in the right location
	assert(in.products.get_data_location() == layer.children.get_data_location());

	// Transform products to arcs
	arcs = data_manager<arc>(get_data_location(), in.products.size(), administrator);
	arcs.with_vector([&](auto policy, auto& v, auto& products_v, auto& layer_v) {
		using Vec = std::remove_reference_t<decltype(v)>;
		using ProductVec = rebind_vector_t<Vec, node_product>;
		
		auto low = thrust::make_transform_iterator(products_v.begin(), transform_product_to_f());
		auto perm_low = thrust::make_permutation_iterator(layer_v.begin(), low);
		assert(products_v.size() == v.size());

		if (quantifier) {
			thrust::transform(
				policy,
				products_v.begin(), products_v.end(),
				perm_low, 
				v.begin(),
				transform_parents_and_f_to_arcs_quantifier()
			);
		}
		else {
			thrust::transform(
				policy,
				products_v.begin(), products_v.end(),
				perm_low, 
				v.begin(),
				transform_parents_and_f_to_arcs()
			);
		}
	}, in.products, layer.children);

	// Sort arcs on high
	arcs.sort(sort_arc_on_g());

	// Insert high arcs
	arcs.with_vector([&](auto policy, auto& v) {
		using Vec = std::remove_reference_t<decltype(v)>;
		using ProductVec = rebind_vector_t<Vec, node_product>;
		auto high = thrust::make_transform_iterator(v.begin(), transform_arc_parent_to_high());
		auto perm_high = thrust::make_permutation_iterator(layer.children.vec<ProductVec>().begin(), high);
		if (quantifier) {
			thrust::transform(
				policy,
				perm_high, perm_high + in.products.size(),
				v.begin(),
				v.begin(),
				fill_arc_g_quantifier()
			);
		}
		else {
			thrust::transform(
				policy,
				perm_high, perm_high + in.products.size(),
				v.begin(),
				v.begin(),
				fill_arc_g()
			);
		}
	});
}

uint32_t arc_layer::size() {
	return arcs.size();
}

void arc_layer::set_data_location(data_location l) {
	arcs.to_location(l);
}

data_location arc_layer::get_data_location() {
	return arcs.get_data_location();
}

#if USE_STORAGE
bool arc_layer::is_spilled() {
	return arcs.is_spilled();
}
#endif

std::vector<arc> arc_layer::get_data() {
	thrust::host_vector<arc> result = arcs;
	return std::vector<arc>(result.begin(), result.end());
}

void arc_layer::set_data(std::vector<arc> data) {
	arcs = data;
}

void arc_layer::print() {
	printf("Parent f: \t");
	for (arc a : arcs) {
		printf("%5u ", a.parent.low);
	}
	printf("\n");
	printf("Parent g: \t");
	for (arc a : arcs) {
		printf("%5u ", a.parent.high);
	}
	printf("\n");

	
	printf("Low f: \t\t");
	for (arc a : arcs) {
		printf("%5u ", a.low.low);
	}
	printf("\n");
	printf("Low g: \t\t");
	for (arc a : arcs) {
		printf("%5u ", a.low.high);
	}
	printf("\n");
	
	printf("High f: \t");
	for (arc a : arcs) {
		printf("%5u ", a.high.low);
	}
	printf("\n");
	printf("High g: \t");
	for (arc a : arcs) {
		printf("%5u ", a.high.high);
	}
	printf("\n");
}