// Sorting functions for semi arc layer

#include "semi_arc_layer.h"

using namespace sro;

struct semi_arc_to_quantified_replacement {
	__host__ __device__
	quantified_replacement operator()(const semi_arc& arc) const {
		return {arc.a.low, arc.b};
	}
};

data_manager<quantified_replacement> semi_arc_layer::quantified_replacements() {
	data_manager<quantified_replacement> result(get_data_location(), semi_arcs.size(), administrator);
	result.with_vector([](auto policy, auto& result_v, auto& arcs_v) {
		thrust::transform(policy, arcs_v.begin(), arcs_v.end(), result_v.begin(), semi_arc_to_quantified_replacement());
	}, semi_arcs);
	result.sort(quantified_replacement_origin_order());
	return result;
}

void semi_arc_layer::set_data_location(data_location l) {
	semi_arcs.to_location(l);
}

data_location semi_arc_layer::get_data_location() {
	return semi_arcs.get_data_location();
}

std::vector<semi_arc> semi_arc_layer::get_data() {
	thrust::host_vector<semi_arc> result = semi_arcs;
	return std::vector<semi_arc>(result.begin(), result.end());
}

void semi_arc_layer::set_data(std::vector<semi_arc> data) {
	semi_arcs = data;
}

void semi_arc_layer::print() {
	if (top_heavy) {
		printf("Parent f: \t");
	} else {
		printf("Low f: \t");
	}
	for (semi_arc s : semi_arcs) {
		printf("%5u ", s.a.low);
	}
	printf("\n");
	if (top_heavy) {
		printf("Parent g: \t");
	} else {
		printf("Low g: \t");
	}
	for (semi_arc s : semi_arcs) {
		printf("%5u ", s.a.high);
	}
	printf("\n");

	if (top_heavy) {
		printf("Child low: \t");
	} else {
		printf("High f: \t");
	}
	for (semi_arc s : semi_arcs) {
		printf("%5u ", s.b.low);
	}
	printf("\n");
	if (top_heavy) {
		printf("Child high: \t");
	} else {
		printf("High g: \t");
	}
	for (semi_arc s : semi_arcs) {
		printf("%5u ", s.b.high);
	}
	printf("\n");
}
