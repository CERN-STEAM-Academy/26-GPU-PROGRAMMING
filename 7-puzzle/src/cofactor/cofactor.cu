#include "../bdd/bdd.h"

#include <algorithm>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <utility>

namespace {
	using sro::node_product;

	struct product_order {
		__host__ __device__
		bool operator()(node_product a, node_product b) const {
			return thrust::make_tuple(a.high, a.low) < thrust::make_tuple(b.high, b.low);
		}
	};

	struct select_branch {
		bool high;

		__host__ __device__
		node_product operator()(node_product product) const {
			const auto child = high ? product.high : product.low;
			return {child, child};
		}
	};

	struct map_product {
		const uint32_t* mapping;

		__host__ __device__
		node_product operator()(node_product product) const {
			return {mapping[product.low], mapping[product.high]};
		}
	};

	struct affected_node {
		template <typename Tuple>
		__host__ __device__
		bool operator()(Tuple item) const {
			const auto product = thrust::get<1>(item);
			return thrust::get<0>(item) && product.low != product.high;
		}
	};

}

namespace sro {
	void bdd::validate_root(const node_ref& root) const {
		const auto owner = root.b.lock();
		if (!owner || owner.get() != this) throw std::invalid_argument("Root belongs to another BDD.");
	}

	std::vector<uint32_t> bdd::normalize_variables(std::vector<uint32_t> variables) const {
		if (std::any_of(variables.begin(), variables.end(), [&](const auto variable) { return variable >= layers.size(); })) {
			throw std::out_of_range("Quantifier variable is out of range.");
		}
		std::sort(variables.rbegin(), variables.rend());
		variables.erase(std::unique(variables.begin(), variables.end()), variables.end());
		return variables;
	}

	node_ref bdd::cofactor(node_ref root, uint32_t variable, bool value) {
		validate_root(root);
		if (variable >= layers.size()) throw std::out_of_range("Cofactor variable is out of range.");

		const auto root_index = root.get_id();

		std::vector<data_manager<bool>> relevant;
		relevant.emplace_back(HOST, layers.front().size(), false, administrator);
		relevant.front()[root_index] = true;
		for (uint32_t layer_index = 0; layer_index < variable; layer_index++) {
			relevant.push_back(layers[layer_index].propagate_encoding(relevant.back(), layers[layer_index + 1]));
		}

		auto make_semantic_map = [&](data_manager<uint32_t> structural, data_manager<bool>& stencil, data_manager<uint32_t>& replacements) {
			auto semantic = structural;
			data_manager<uint32_t> positions(structural.get_data_location(), replacements.size(), administrator);
			positions.with_vector([&](auto policy, auto& positions_v, auto& stencil_v) {
				auto indices = thrust::make_counting_iterator<uint32_t>(0);
				thrust::copy_if(
					policy,
					indices, indices + stencil_v.size(),
					stencil_v.begin(),
					positions_v.begin(),
					::cuda::std::identity()
				);
			}, stencil);
			semantic.with_vector([&](auto policy, auto& semantic_v, auto& positions_v, auto& replacements_v) {
				thrust::scatter(
					policy,
					replacements_v.begin(), replacements_v.end(),
					positions_v.begin(),
					semantic_v.begin()
				);
			}, positions, replacements);
			return semantic;
		};

		auto insert_replacements = [&](data_manager<node_product>& candidates, layer& target) {
			auto insertions = candidates;
			insertions.sort(product_order());
			auto structural = target.insert_elements(insertions);

			data_manager<uint32_t> result(target.get_data_location(), candidates.size(), administrator);
			result.with_vector([&](auto policy, auto& result_v, auto& candidates_v, auto& children_v) {
				thrust::lower_bound(
					policy,
					children_v.begin(), children_v.end(),
					candidates_v.begin(), candidates_v.end(),
					result_v.begin(),
					product_order()
				);
			}, candidates, target.children);
			return std::pair(std::move(structural), std::move(result));
		};

		auto& target = layers[variable];
		relevant.back().to_location(target.get_data_location());
		const auto affected = target.children.with_vector([&](auto policy, auto& children_v, auto& stencil_v) {
			auto nodes = thrust::make_zip_iterator(thrust::make_tuple(stencil_v.begin(), children_v.begin()));
			return thrust::count_if(policy, nodes, nodes + children_v.size(), affected_node());
		}, relevant.back());
		if (affected == 0) return root;

		false_cache_up_to_date = false;
		const auto replacement_count = relevant.back().count_if(::cuda::std::identity());
		data_manager<node_product> replacements(target.get_data_location(), replacement_count, administrator);
		replacements.with_vector([&](auto policy, auto& replacements_v, auto& children_v, auto& stencil_v) {
			auto selected = thrust::make_transform_iterator(children_v.begin(), select_branch{value});
			thrust::copy_if(
				policy,
				selected, selected + children_v.size(),
				stencil_v.begin(),
				replacements_v.begin(),
				::cuda::std::identity()
			);
		}, target.children, relevant.back());
		auto [structural, inserted] = insert_replacements(replacements, target);
		auto semantic = make_semantic_map(structural, relevant.back(), inserted);

		for (uint32_t layer_index = variable; layer_index-- > 0;) {
			auto& parent = layers[layer_index];
			structural.to_location(parent.get_data_location());
			semantic.to_location(parent.get_data_location());
			relevant[layer_index].to_location(parent.get_data_location());
			const auto count = relevant[layer_index].count_if(::cuda::std::identity());
			data_manager<node_product> parent_replacements(parent.get_data_location(), count, administrator);
			parent_replacements.with_vector([&](auto policy, auto& replacements_v, auto& children_v, auto& stencil_v, auto& semantic_v) {
				const auto mapping = thrust::raw_pointer_cast(semantic_v.data());
				auto mapped = thrust::make_transform_iterator(children_v.begin(), map_product{mapping});
				thrust::copy_if(
					policy,
					mapped, mapped + children_v.size(),
					stencil_v.begin(),
					replacements_v.begin(),
					::cuda::std::identity()
				);
			}, parent.children, relevant[layer_index], semantic);
			parent.apply_ordered_map(structural);
			auto insertion = insert_replacements(parent_replacements, parent);
			structural = std::move(insertion.first);
			inserted = std::move(insertion.second);
			semantic = make_semantic_map(structural, relevant[layer_index], inserted);
		}

		thrust::host_vector<uint32_t> root_mapping = structural;
		apply_root_layer_mapping(root_mapping);
		return node_ref(add_root(semantic[root_index]), weak_from_this());
	}

	node_ref bdd::projection(node_ref root, uint32_t variable, bool high) {
		return cofactor(root, variable, high);
	}

	node_ref bdd::exists_reference(node_ref root, uint32_t variable) {
		auto low = cofactor(root, variable, false);
		auto high = cofactor(root, variable, true);
		return logical_or(low, high);
	}

	node_ref bdd::exists_reference(node_ref root, std::vector<uint32_t> variables) {
		validate_root(root);
		for (const auto variable : normalize_variables(std::move(variables))) root = exists_reference(root, variable);
		return root;
	}

	node_ref bdd::for_all_reference(node_ref root, uint32_t variable) {
		auto low = cofactor(root, variable, false);
		auto high = cofactor(root, variable, true);
		return logical_and(low, high);
	}

	node_ref bdd::for_all_reference(node_ref root, std::vector<uint32_t> variables) {
		validate_root(root);
		for (const auto variable : normalize_variables(std::move(variables))) root = for_all_reference(root, variable);
		return root;
	}
}
