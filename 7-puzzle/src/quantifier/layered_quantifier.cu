#include "../bdd/bdd.h"
#include "../arc_layer/arc_layer.h"
#include "../product_layer/product_layer.h"
#include "../semi_arc_layer/semi_arc_layer.h"

#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>

using namespace sro;

namespace {
	struct product_order {
		__host__ __device__
		bool operator()(node_product lhs, node_product rhs) const {
			return thrust::make_tuple(lhs.high, lhs.low) < thrust::make_tuple(rhs.high, rhs.low);
		}
	};

	struct replacement_product {
		__host__ __device__
		node_product operator()(const quantified_replacement& replacement) const {
			return replacement.replacement;
		}
	};

	struct replacement_origin {
		__host__ __device__
		uint32_t operator()(const quantified_replacement& replacement) const {
			return replacement.origin;
		}
	};

	struct map_product {
		const uint32_t* mapping;

		__host__ __device__
		node_product operator()(node_product product) const {
			return {mapping[product.low], mapping[product.high]};
		}
	};

	struct make_replacement {
		__host__ __device__
		quantified_replacement operator()(uint32_t origin, node_product replacement) const {
			return {origin, replacement};
		}
	};

	struct map_replacement_origin {
		const uint32_t* mapping;

		__host__ __device__
		void operator()(quantified_replacement& replacement) const {
			replacement.origin = mapping[replacement.origin];
		}
	};

	struct compose_replacement {
		const quantified_replacement* quantified;

		__host__ __device__
		void operator()(thrust::tuple<quantified_replacement&, uint32_t> item) const {
			thrust::get<0>(item).replacement = quantified[thrust::get<1>(item)].replacement;
		}
	};

	struct apply_or_layered {
		__host__ __device__
		uint32_t operator()(uint32_t lhs, uint32_t rhs) const { return lhs || rhs; }
	};

	struct apply_and_layered {
		__host__ __device__
		uint32_t operator()(uint32_t lhs, uint32_t rhs) const { return lhs && rhs; }
	};

	struct layer_location_guard {
		std::vector<layer>& layers;
		std::vector<data_location> locations;

		explicit layer_location_guard(std::vector<layer>& layers) : layers(layers) {
			locations.reserve(layers.size());
			for (auto& layer : layers) locations.push_back(layer.get_data_location());
		}

		~layer_location_guard() {
			for (uint32_t index = 0; index < layers.size(); ++index) {
				layers[index].set_data_location(locations[index]);
			}
		}
	};
}

template<typename binary_operation>
node_ref bdd::layered_quantifier(node_ref root, std::vector<uint32_t> variables, binary_operation op) {
	false_cache_up_to_date = false;
	layer_location_guard location_guard(layers);
	std::vector<data_manager<bool>> reachable;
	reachable.emplace_back(HOST, layers[0].size(), false, administrator);
	reachable[0][root.get_id()] = true;
	for (uint32_t layer_index = 0; layer_index < variables.front(); ++layer_index) {
		reachable.push_back(layers[layer_index].propagate_encoding(reachable.back(), layers[layer_index + 1]));
	}

	auto materialize = [&](uint32_t layer_index, data_manager<quantified_replacement>& replacements) {
		auto location = layers[layer_index].get_data_location();
		data_manager<node_product> products(location, replacements.size(), administrator);
		products.with_vector([](auto policy, auto& products_v, auto& replacements_v) {
			thrust::transform(policy, replacements_v.begin(), replacements_v.end(), products_v.begin(), replacement_product());
		}, replacements);
		products.sort(product_order());
		products.unique();

		auto structural = layers[layer_index].insert_elements(products);

		data_manager<uint32_t> ids(location, replacements.size(), administrator);
		ids.with_vector([](auto policy, auto& ids_v, auto& replacements_v, auto& children_v, auto& structural_v) {
			thrust::for_each(
				policy,
				replacements_v.begin(), replacements_v.end(),
				map_replacement_origin{thrust::raw_pointer_cast(structural_v.data())}
			);
			auto values = thrust::make_transform_iterator(replacements_v.begin(), replacement_product());
			thrust::lower_bound(
				policy,
				children_v.begin(), children_v.end(),
				values, values + replacements_v.size(),
				ids_v.begin(),
				product_order()
			);
		}, replacements, layers[layer_index].children, structural);
		if (layer_index == 0) {
			thrust::host_vector<uint32_t> host_mapping = structural;
			apply_root_layer_mapping(host_mapping);
		}
		else {
			structural.to_location(layers[layer_index - 1].get_data_location());
			layers[layer_index - 1].apply_ordered_map(structural);
		}
		return ids;
	};

	auto propagate = [&](uint32_t child_index, data_manager<quantified_replacement> replacements) {
		auto ids = materialize(child_index, replacements);
		auto location = layers[child_index - 1].get_data_location();
		data_manager<uint32_t> dense(layers[child_index].get_data_location(), layers[child_index].size(), administrator);
		dense.with_vector([](auto policy, auto& dense_v, auto& replacements_v, auto& ids_v) {
			thrust::sequence(policy, dense_v.begin(), dense_v.end());
			auto origins = thrust::make_transform_iterator(replacements_v.begin(), replacement_origin());
			thrust::scatter(policy, ids_v.begin(), ids_v.end(), origins, dense_v.begin());
		}, replacements, ids);
		dense.to_location(location);

		const auto count = reachable[child_index - 1].count_if(::cuda::std::identity());
		data_manager<uint32_t> origins(location, count, administrator);
		data_manager<node_product> values(location, count, administrator);
		origins.with_vector([&](auto policy, auto& origins_v, auto& stencil_v) {
			auto counting = thrust::make_counting_iterator<uint32_t>(0);
			thrust::copy_if(policy, counting, counting + stencil_v.size(), stencil_v.begin(), origins_v.begin(), ::cuda::std::identity());
		}, reachable[child_index - 1]);
		values.with_vector([&](auto policy, auto& values_v, auto& origins_v, auto& parents_v, auto& dense_v) {
			thrust::gather(policy, origins_v.begin(), origins_v.end(), parents_v.begin(), values_v.begin());
			thrust::transform(policy, values_v.begin(), values_v.end(), values_v.begin(), map_product{thrust::raw_pointer_cast(dense_v.data())});
		}, origins, layers[child_index - 1].children, dense);

		data_manager<quantified_replacement> result(location, count, administrator);
		result.with_vector([](auto policy, auto& result_v, auto& origins_v, auto& values_v) {
			thrust::transform(policy, origins_v.begin(), origins_v.end(), values_v.begin(), result_v.begin(), make_replacement());
		}, origins, values);
		return result;
	};

	data_manager<quantified_replacement> replacements;
	for (uint32_t variable_index = 0; variable_index < variables.size(); ++variable_index) {
		const auto variable = variables[variable_index];
		data_manager<uint32_t> participants;
		if (variable_index == 0) {
			reachable.back().to_location(layers[variable].get_data_location());
			const auto count = reachable.back().count_if(::cuda::std::identity());
			participants = data_manager<uint32_t>(layers[variable].get_data_location(), count, administrator);
			participants.with_vector([](auto policy, auto& participants_v, auto& stencil_v) {
				auto counting = thrust::make_counting_iterator<uint32_t>(0);
				thrust::copy_if(policy, counting, counting + stencil_v.size(), stencil_v.begin(), participants_v.begin(), ::cuda::std::identity());
			}, reachable.back());
		}
		else {
			participants = materialize(variable, replacements);
		}

		data_manager<bool> participant_stencil(layers[variable].get_data_location(), layers[variable].size(), false, administrator);
		participant_stencil.with_vector([](auto policy, auto& stencil_v, auto& participants_v) {
			thrust::scatter(
				policy,
				thrust::make_constant_iterator(true),
				thrust::make_constant_iterator(true) + participants_v.size(),
				participants_v.begin(),
				stencil_v.begin()
			);
		}, participants);

		std::vector<arc_layer> arcs;
		product_layer initial(layers[variable], participant_stencil);
		arcs.emplace_back(initial, layers[variable], true);
		for (uint32_t layer_index = variable + 1; layer_index < layers.size(); ++layer_index) {
			product_layer products(arcs.back());
			arcs.emplace_back(products, layers[layer_index]);
		}

		semi_arc_layer semi_arcs(arcs.back(), op);
		for (uint32_t layer_index = layers.size() - 1; layer_index > variable; --layer_index) {
			arcs.pop_back();
			semi_arcs = semi_arc_layer(arcs.back(), semi_arcs, layers[layer_index], layers[layer_index - 1]);
		}
		auto quantified = semi_arcs.quantified_replacements();

		if (variable_index == 0) {
			replacements = std::move(quantified);
		}
		else {
			quantified.sort(quantified_replacement_origin_order());
			data_manager<uint32_t> positions(participants.get_data_location(), participants.size(), administrator);
			positions.with_vector([](auto policy, auto& positions_v, auto& participants_v, auto& quantified_v) {
				auto origins = thrust::make_transform_iterator(quantified_v.begin(), replacement_origin());
				thrust::lower_bound(policy, origins, origins + quantified_v.size(), participants_v.begin(), participants_v.end(), positions_v.begin());
			}, participants, quantified);
			replacements.with_vector([](auto policy, auto& replacements_v, auto& positions_v, auto& quantified_v) {
				auto zipped = thrust::make_zip_iterator(replacements_v.begin(), positions_v.begin());
				thrust::for_each(
					policy,
					zipped, zipped + replacements_v.size(),
					compose_replacement{thrust::raw_pointer_cast(quantified_v.data())}
				);
			}, positions, quantified);
		}

		const auto target = variable_index + 1 < variables.size() ? variables[variable_index + 1] : 0;
		for (uint32_t layer_index = variable; layer_index > target; --layer_index) {
			replacements = propagate(layer_index, std::move(replacements));
		}
	}

	auto result_ids = materialize(0, replacements);
	thrust::host_vector<quantified_replacement> host_replacements = replacements;
	thrust::host_vector<uint32_t> host_ids = result_ids;
	const auto current_root = root.get_id();
	for (uint32_t index = 0; index < host_replacements.size(); ++index) {
		if (host_replacements[index].origin == current_root) {
			return node_ref(add_root(host_ids[index]), weak_from_this());
		}
	}
	throw std::logic_error("Layered quantifier lost the requested root mapping.");
}

node_ref bdd::exists_fused_layered(node_ref root, std::vector<uint32_t> variables) {
	validate_root(root);
	variables = normalize_variables(std::move(variables));
	if (variables.empty()) return root;
	return layered_quantifier(root, std::move(variables), apply_or_layered());
}

node_ref bdd::for_all_fused_layered(node_ref root, std::vector<uint32_t> variables) {
	validate_root(root);
	variables = normalize_variables(std::move(variables));
	if (variables.empty()) return root;
	return layered_quantifier(root, std::move(variables), apply_and_layered());
}
