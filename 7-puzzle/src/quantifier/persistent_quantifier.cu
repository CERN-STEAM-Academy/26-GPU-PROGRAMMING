#include "../bdd/bdd.h"
#include "persistent_state.h"

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <limits>

using namespace sro;

namespace {
	struct apply_or_persistent {
		__host__ __device__
		uint32_t operator()(uint32_t lhs, uint32_t rhs) const { return lhs || rhs; }
	};

	struct apply_and_persistent {
		__host__ __device__
		uint32_t operator()(uint32_t lhs, uint32_t rhs) const { return lhs && rhs; }
	};

	struct expand_state_member {
		const persistent_state_member* members;
		const node_product* children;
		bool quantified;

		__host__ __device__
		persistent_state_member operator()(uint32_t index) const {
			const auto member = members[index / 2];
			const auto high = index & 1;
			const auto child = children[member.node];
			return {
				quantified ? member.state : 2 * member.state + high,
				high ? child.high : child.low,
			};
		}
	};

	struct member_state {
		__host__ __device__
		uint32_t operator()(const persistent_state_member& member) const { return member.state; }
	};

	struct state_row_order {
		const persistent_state_member* members;
		const uint32_t* offsets;

		__host__ __device__
		bool operator()(uint32_t lhs, uint32_t rhs) const {
			const auto lhs_size = offsets[lhs + 1] - offsets[lhs];
			const auto rhs_size = offsets[rhs + 1] - offsets[rhs];
			const auto common = lhs_size < rhs_size ? lhs_size : rhs_size;
			for (uint32_t index = 0; index < common; ++index) {
				const auto lhs_node = members[offsets[lhs] + index].node;
				const auto rhs_node = members[offsets[rhs] + index].node;
				if (lhs_node != rhs_node) return lhs_node < rhs_node;
			}
			return lhs_size < rhs_size;
		}
	};

	struct state_row_equal {
		const persistent_state_member* members;
		const uint32_t* offsets;

		__host__ __device__
		bool operator()(uint32_t lhs, uint32_t rhs) const {
			const auto lhs_size = offsets[lhs + 1] - offsets[lhs];
			if (lhs_size != offsets[rhs + 1] - offsets[rhs]) return false;
			for (uint32_t index = 0; index < lhs_size; ++index) {
				if (members[offsets[lhs] + index].node != members[offsets[rhs] + index].node) return false;
			}
			return true;
		}
	};

	struct canonical_boundary {
		const uint32_t* order;
		state_row_equal equal;

		__host__ __device__
		uint32_t operator()(uint32_t index) const {
			return index == 0 || !equal(order[index - 1], order[index]);
		}
	};

	struct decrement {
		__host__ __device__
		uint32_t operator()(uint32_t value) const { return value - 1; }
	};

	struct remap_member_state {
		const uint32_t* mapping;

		__host__ __device__
		void operator()(persistent_state_member& member) const { member.state = mapping[member.state]; }
	};

	struct make_state_arc {
		const uint32_t* mapping;
		bool quantified;

		__host__ __device__
		persistent_state_arc operator()(uint32_t parent) const {
			const auto low = mapping[quantified ? parent : 2 * parent];
			const auto high = quantified ? low : mapping[2 * parent + 1];
			return {low, high};
		}
	};

	struct member_node {
		__host__ __device__
		uint32_t operator()(const persistent_state_member& member) const { return member.node; }
	};

	struct make_candidate {
		const uint32_t* node_ids;

		__host__ __device__
		node_product operator()(const persistent_state_arc& arc) const {
			return {node_ids[arc.low], node_ids[arc.high]};
		}
	};

	struct product_order {
		__host__ __device__
		bool operator()(const node_product& lhs, const node_product& rhs) const {
			return thrust::make_tuple(lhs.high, lhs.low) < thrust::make_tuple(rhs.high, rhs.low);
		}
	};

	struct canonicalized_states {
		data_manager<uint32_t> mapping;
		uint32_t count;
	};

	canonicalized_states canonicalize_states(
		data_manager<persistent_state_member>& members,
		uint32_t state_count
	) {
		members.sort(persistent_state_member_order());
		members.unique();
		const auto location = members.get_data_location();
		data_manager<uint32_t> offsets(location, state_count + 1, members.administrator);
		offsets.with_vector([&](auto policy, auto& offsets_v, auto& members_v) {
			auto states = thrust::make_transform_iterator(members_v.begin(), member_state());
			auto queries = thrust::make_counting_iterator<uint32_t>(0);
			thrust::lower_bound(
				policy,
				states, states + members_v.size(),
				queries, queries + offsets_v.size(),
				offsets_v.begin()
			);
		}, members);

		data_manager<uint32_t> order(location, state_count, members.administrator);
		order.with_vector([&](auto policy, auto& order_v, auto& members_v, auto& offsets_v) {
			thrust::sequence(policy, order_v.begin(), order_v.end());
			thrust::sort(
				policy,
				order_v.begin(), order_v.end(),
				state_row_order{
					thrust::raw_pointer_cast(members_v.data()),
					thrust::raw_pointer_cast(offsets_v.data()),
				}
			);
		}, members, offsets);

		data_manager<uint32_t> canonical_sorted(location, state_count, members.administrator);
		canonical_sorted.with_vector([&](auto policy, auto& ids_v, auto& order_v, auto& members_v, auto& offsets_v) {
			auto indices = thrust::make_counting_iterator<uint32_t>(0);
			auto boundaries = thrust::make_transform_iterator(
				indices,
				canonical_boundary{
					thrust::raw_pointer_cast(order_v.data()),
					state_row_equal{
						thrust::raw_pointer_cast(members_v.data()),
						thrust::raw_pointer_cast(offsets_v.data()),
					},
				}
			);
			thrust::inclusive_scan(policy, boundaries, boundaries + ids_v.size(), ids_v.begin());
			thrust::transform(policy, ids_v.begin(), ids_v.end(), ids_v.begin(), decrement());
		}, order, members, offsets);

		data_manager<uint32_t> mapping(location, state_count, members.administrator);
		mapping.with_vector([](auto policy, auto& mapping_v, auto& ids_v, auto& order_v) {
			thrust::scatter(policy, ids_v.begin(), ids_v.end(), order_v.begin(), mapping_v.begin());
		}, canonical_sorted, order);
		members.with_vector([](auto policy, auto& members_v, auto& mapping_v) {
			thrust::for_each(
				policy,
				members_v.begin(), members_v.end(),
				remap_member_state{thrust::raw_pointer_cast(mapping_v.data())}
			);
		}, mapping);
		members.sort(persistent_state_member_order());
		members.unique();
		const auto count = canonical_sorted[canonical_sorted.size() - 1].get_data() + 1;
		return {std::move(mapping), count};
	}
}

template<typename binary_operation>
node_ref bdd::persistent_quantifier(
	node_ref root,
	std::vector<uint32_t> variables,
	binary_operation op
) {
	false_cache_up_to_date = false;
	persistent_statistics = {};
	persistent_statistics.states_per_layer.reserve(layers.size());
	std::vector<bool> quantified(layers.size(), false);
	for (const auto variable : variables) quantified[variable] = true;

	std::vector<data_manager<persistent_state_arc>> graph;
	graph.reserve(layers.size());
	data_manager<persistent_state_member> members(layers.front().get_data_location(), 1, administrator);
	std::vector<persistent_state_member> initial{{0, root.get_id()}};
	members = initial;
	uint32_t state_count = 1;

	for (uint32_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
		const auto max_state_count = std::numeric_limits<uint32_t>::max();
		if (state_count == max_state_count || (!quantified[layer_index] && state_count > max_state_count / 2)) {
			throw std::overflow_error("Persistent quantifier state count exceeds uint32_t capacity.");
		}
		if (members.size() > std::numeric_limits<uint32_t>::max() / 2) {
			throw std::overflow_error("Persistent quantifier member count exceeds uint32_t capacity.");
		}
		const auto location = layers[layer_index].get_data_location();
		members.to_location(location);
		data_manager<persistent_state_member> expanded(location, 2 * members.size(), administrator);
		expanded.with_vector([&](auto policy, auto& expanded_v, auto& members_v, auto& children_v) {
			auto indices = thrust::make_counting_iterator<uint32_t>(0);
			thrust::transform(
				policy,
				indices, indices + expanded_v.size(),
				expanded_v.begin(),
				expand_state_member{
					thrust::raw_pointer_cast(members_v.data()),
					thrust::raw_pointer_cast(children_v.data()),
					quantified[layer_index],
				}
			);
		}, members, layers[layer_index].children);

		const auto branch_count = quantified[layer_index] ? state_count : 2 * state_count;
		auto canonical = canonicalize_states(expanded, branch_count);
		data_manager<persistent_state_arc> arcs(location, state_count, administrator);
		arcs.with_vector([&](auto policy, auto& arcs_v, auto& mapping_v) {
			auto parents = thrust::make_counting_iterator<uint32_t>(0);
			thrust::transform(
				policy,
				parents, parents + arcs_v.size(),
				arcs_v.begin(),
				make_state_arc{thrust::raw_pointer_cast(mapping_v.data()), quantified[layer_index]}
			);
		}, canonical.mapping);
		graph.push_back(std::move(arcs));
		members = std::move(expanded);
		state_count = canonical.count;
		persistent_statistics.states_per_layer.push_back(state_count);
		if (state_count > persistent_statistics.peak_state_count) {
			persistent_statistics.peak_state_count = state_count;
			persistent_statistics.peak_state_layer = layer_index;
		}
		if (members.size() > persistent_statistics.peak_layer_member_count) {
			persistent_statistics.peak_layer_member_count = members.size();
			persistent_statistics.peak_layer_member_count_layer = layer_index;
		}
	}

	data_manager<uint32_t> node_ids(members.get_data_location(), state_count, administrator);
	node_ids.with_vector([&](auto policy, auto& ids_v, auto& members_v) {
		auto keys = thrust::make_transform_iterator(members_v.begin(), member_state());
		auto values = thrust::make_transform_iterator(members_v.begin(), member_node());
		thrust::reduce_by_key(
			policy,
			keys, keys + members_v.size(),
			values,
			thrust::make_discard_iterator(), ids_v.begin(),
			thrust::equal_to<uint32_t>(), op
		);
	}, members);

	for (uint32_t layer_index = layers.size(); layer_index-- > 0;) {
		const auto location = layers[layer_index].get_data_location();
		graph[layer_index].to_location(location);
		node_ids.to_location(location);
		data_manager<node_product> candidates(location, graph[layer_index].size(), administrator);
		candidates.with_vector([](auto policy, auto& candidates_v, auto& arcs_v, auto& ids_v) {
			thrust::transform(
				policy,
				arcs_v.begin(), arcs_v.end(),
				candidates_v.begin(),
				make_candidate{thrust::raw_pointer_cast(ids_v.data())}
			);
		}, graph[layer_index], node_ids);
		auto insertions = candidates;
		insertions.sort(product_order());
		insertions.unique();
		auto structural = layers[layer_index].insert_elements(insertions);
		if (layer_index == 0) {
			thrust::host_vector<uint32_t> root_mapping = structural;
			apply_root_layer_mapping(root_mapping);
		}
		else {
			structural.to_location(layers[layer_index - 1].get_data_location());
			layers[layer_index - 1].apply_ordered_map(structural);
		}
		data_manager<uint32_t> parent_ids(location, candidates.size(), administrator);
		parent_ids.with_vector([](auto policy, auto& ids_v, auto& candidates_v, auto& children_v) {
			thrust::lower_bound(
				policy,
				children_v.begin(), children_v.end(),
				candidates_v.begin(), candidates_v.end(),
				ids_v.begin(),
				product_order()
			);
		}, candidates, layers[layer_index].children);
		node_ids = std::move(parent_ids);
	}

	return node_ref(add_root(node_ids[0]), weak_from_this());
}

const persistent_quantifier_statistics& bdd::get_persistent_quantifier_statistics() const {
	return persistent_statistics;
}

node_ref bdd::exists_fused_persistent(node_ref root, std::vector<uint32_t> variables) {
	validate_root(root);
	variables = normalize_variables(std::move(variables));
	if (variables.empty()) {
		persistent_statistics = {};
		return root;
	}
	return persistent_quantifier(root, std::move(variables), apply_or_persistent());
}

node_ref bdd::for_all_fused_persistent(node_ref root, std::vector<uint32_t> variables) {
	validate_root(root);
	variables = normalize_variables(std::move(variables));
	if (variables.empty()) {
		persistent_statistics = {};
		return root;
	}
	return persistent_quantifier(root, std::move(variables), apply_and_persistent());
}
