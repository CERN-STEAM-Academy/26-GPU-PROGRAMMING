// Constructors for layers

#include "layer.h"

using namespace sro;

// Dummy layer
layer::layer(std::shared_ptr<sro::data_administrator> admin) : children(admin) {}

layer::layer(uint16_t label, uint16_t variable_count, std::shared_ptr<data_administrator> admin) : administrator(admin), children(admin) {
	_label = label;

	thrust::host_vector<node_product> h_children;
	// Layer initialized sorted: (low, variable, high)
	if (label > 0) {
		h_children.push_back({
			0u,
			0u
		});
	}


	for (uint16_t i = 1; i < variable_count - label; i++) {
		h_children.push_back({
			i, i
		});
	}

	h_children.push_back({
		0u, (uint16_t)(variable_count - label)
	});
	
	if (label > 0) {
		h_children.push_back({
			(uint16_t)(variable_count - label),
			(uint16_t)(variable_count - label)
		});
		_variable = 2;
	}
	else {
		_variable = 0;
	}

	children = h_children;
	children.administrator = admin;
}

layer::layer(const layer& other) : children(other.administrator) {
	_variable = other._variable;
	_label = other._label;
	children = data_manager<node_product>(other.children);
	administrator = other.administrator;
}
