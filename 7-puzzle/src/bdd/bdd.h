#ifndef SROBDD
#define SROBDD

#include "../layer/layer.h"
#include "../quantifier/persistent_state.h"
#include "administrators.h"

#include <thrust/host_vector.h>
#include <iostream>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>

namespace sro{
	class bdd;
	class bdd_iterator;
	
	class node_ref{
		friend class bdd;

		uint32_t root_id;
		std::weak_ptr<bdd> b;

		void check_and_propagate_redirect();

		void increment_reference();
		void decrement_reference();
		
		node_ref(const node_ref& from, std::weak_ptr<bdd> target);

		public:
		
		node_ref(uint32_t _root_id, std::weak_ptr<bdd> _b);
		node_ref(const node_ref& from);
		node_ref(node_ref&& from);
		~node_ref();

		node_ref& operator=(const node_ref& from);
		uint32_t get_id() const;
		uint32_t get_id();
		uint64_t get_metric(std::vector<uint64_t> root_metrics);
		
		bdd_iterator begin();
		bdd_iterator end();
	};

	class bdd_iterator{
		bdd* b;
		node_ref root;
		std::vector<bool> assignments;
		
		public:

		using difference_type = std::ptrdiff_t;
		using value_type = std::vector<bool>;

		bdd_iterator(bdd* b, node_ref root, std::vector<bool> assignments);

		std::vector<bool> operator*() const;
		bdd_iterator& operator++();
		void operator++(int);
    	bool operator==(const bdd_iterator&) const;
	};

	/**
	 * The main (quasi-reduced) binary decision diagram class for GPUdecide.
	 */
	class bdd : public std::enable_shared_from_this<bdd> {
		friend class node_ref;
		friend class bdd_iterator;
		friend class semi_arc_layer;

		std::shared_ptr<bdd> temp;

		std::vector<layer> layers;

		uint32_t lifetime_root_count = 0;
		std::map<uint32_t, uint32_t> roots;
		std::map<uint32_t, uint32_t> root_refcounts;
		std::map<uint32_t, std::shared_ptr<bdd>> redirects; // These nodes have been separated
		
		std::vector<std::shared_ptr<bdd>> copies;

		std::map<std::weak_ptr<bdd>, uint32_t, std::owner_less<std::weak_ptr<bdd>>> copied_bdds;
		std::vector<node_ref> variables;

		void increment_reference(uint32_t root_id);
		void decrement_reference(uint32_t root_id);

		void update_bdd_nodes(data_manager<node_product>& children, layer& child_layer);

		std::vector<node_ref> queued_for_deletion;

		bool false_cache_up_to_date = false;
		persistent_quantifier_statistics persistent_statistics;
		uint16_t lowermost_occurrence_of_false = 0;
		void update_occurrences_of_false();
		void validate_root(const node_ref& root) const;
		std::vector<uint32_t> normalize_variables(std::vector<uint32_t> variables) const;

		template<typename binary_operation>
		node_ref quantifier(node_ref a, std::vector<uint32_t> v, binary_operation op);
		template<typename binary_operation>
		node_ref quantifier(node_ref a, uint32_t v, binary_operation op);
		template<typename binary_operation>
		node_ref layered_quantifier(node_ref root, std::vector<uint32_t> variables, binary_operation op);
		template<typename binary_operation>
		node_ref persistent_quantifier(node_ref root, std::vector<uint32_t> variables, binary_operation op);
		template<typename binary_operation>
		node_ref logical_operation(node_ref a, node_ref b, binary_operation op);

		public:

		std::shared_ptr<data_administrator> administrator;

		void remove_root_nodes(std::vector<uint32_t> nodes);
		void collect_garbage(uint16_t layer);
		void remove_variables();

		void remove_layers(std::vector<uint16_t> layers);
		void insert_removable_layers(std::vector<uint16_t> layers);

		std::shared_ptr<bdd> separate_root(node_ref root);
		std::shared_ptr<bdd> separate_roots(std::vector<node_ref> roots);
		std::pair<std::shared_ptr<bdd>, std::vector<sro::node_ref>> duplicate(std::vector<sro::node_ref> node_refs);
		void join_with(std::shared_ptr<bdd> bdd, bool destroy=false);

		void apply_root_layer_mapping(thrust::host_vector<uint32_t>& p);
		void apply_root_layer_mapping(std::vector<uint32_t>& p);
		uint32_t add_root(uint32_t id);

		bdd(uint16_t depth, std::shared_ptr<data_administrator> administrator = std::make_shared<thrust_administrator>());
		bdd(bdd& other, std::shared_ptr<data_administrator> administrator = std::make_shared<thrust_administrator>());
		// bdd(std::istream& stream);
		~bdd();
		std::shared_ptr<bdd> extract_shared_pointer();

		node_ref exists(node_ref a, uint32_t v);
		node_ref exists(node_ref a, std::vector<uint32_t> variables);
		node_ref for_all(node_ref a, uint32_t v);
		node_ref for_all(node_ref a, std::vector<uint32_t> variables);
		node_ref cofactor(node_ref root, uint32_t variable, bool value);
		node_ref projection(node_ref root, uint32_t variable, bool high);
		node_ref exists_reference(node_ref root, uint32_t variable);
		node_ref exists_reference(node_ref root, std::vector<uint32_t> variables);
		node_ref for_all_reference(node_ref root, uint32_t variable);
		node_ref for_all_reference(node_ref root, std::vector<uint32_t> variables);
		node_ref exists_fused_layered(node_ref root, std::vector<uint32_t> variables);
		node_ref for_all_fused_layered(node_ref root, std::vector<uint32_t> variables);
		node_ref exists_fused_persistent(node_ref root, std::vector<uint32_t> variables);
		node_ref for_all_fused_persistent(node_ref root, std::vector<uint32_t> variables);
		const persistent_quantifier_statistics& get_persistent_quantifier_statistics() const;

		node_ref logical_and(node_ref a, node_ref b);
		node_ref logical_nand(node_ref a, node_ref b);
		node_ref logical_or(node_ref a, node_ref b);
		node_ref logical_nor(node_ref a, node_ref b);
		node_ref logical_xor(node_ref a, node_ref b);
		node_ref logical_eq(node_ref a, node_ref b);
		node_ref logical_difference(node_ref a, node_ref b);
		node_ref logical_implication(node_ref a, node_ref b);
		node_ref logical_not(node_ref a);
		node_ref logical_ite(node_ref i, node_ref t, node_ref e);
		node_ref logical_true();
		node_ref logical_false();

		node_ref get_variable(uint16_t label);
		sro::layer& get_layer(uint16_t label);

		std::vector<uint64_t> count_satisfying_assignments();

		std::vector<node_ref> get_all_node_refs();

		bdd_iterator begin(node_ref ref);
		bdd_iterator end(node_ref ref);

		uint64_t count_nodes();
		uint64_t count_irreducible_nodes();

		void print();
		void print_roots();

		// std::vector<node_ref> input_dddmp(std::istream& stream);
		// void output_dddmp(std::ostream& stream, std::vector<node_ref> refs);
	};
};

#endif
