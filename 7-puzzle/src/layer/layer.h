#ifndef SRO_LAYER
#define SRO_LAYER

#include "../data_manager/data_manager.h"

#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

namespace sro{
	/**
	 * A node_product is a product of two indices, `low` and `high`. In the context of layers, the `low` index refers to the
	 * node representing the BDD if the respective variable is false, and the `high` index refers to the node representing
	 * the BDD if the respective variable is true. As the BDD is quasi-reduced, these indices always refer to the next layer.
	 */
	struct node_product{
		uint32_t low;
		uint32_t high;

		__host__ __device__
		bool operator==(node_product const& other) const{
			return low == other.low && high == other.high;
		}

		__host__ __device__
		void project_stencil(uint32_t* stencil);
		
		template<typename M>
		__host__ __device__
		void apply_map(M begin) {
			low = *(begin + low);
			high = *(begin + high);
		}

		__host__ __device__
		bool removable() {
			return low == high;
		}
	};

	class arc_layer;
	class semi_arc_layer;

	/**
	 * The sro::layer class contains the low and high paths for the variable in the BDD at the respective depth. This depth
	 * is stored in the `_label` variable. A sro::layer contains a list of sro::node_product objects, for which the low and
	 * high edges refer to indices of the layer below (so where _label is 1 higher). The terminal layer itself is not stored
	 * in memory.
	 */
	class layer{
		friend class product_layer;
		friend class arc_layer;
		friend class semi_arc_layer;
		friend class bdd;

		data_manager<node_product> children;

		uint32_t _variable;
		uint16_t _label;
		uint32_t _irreducible_count = 0;

		public:
		/**
		 * Administers the data policy and location of the children. depending on the size.
		 */
		std::shared_ptr<data_administrator> administrator;

		layer(std::shared_ptr<sro::data_administrator> admin);
		layer(uint16_t label, uint16_t size, std::shared_ptr<data_administrator> admin);
		layer(const layer& other);

		data_manager<uint64_t> satisfying_assignments(data_manager<uint64_t>& lower_count);
		uint32_t node_count();
		uint32_t irreducible_node_count();

		data_manager<uint32_t> apply_map(data_manager<uint32_t>& mapping);
		void apply_ordered_map(data_manager<uint32_t>& mapping);

		data_manager<uint32_t> update_child_ids(data_manager<node_product>& old_children, layer& child_layer);
		data_manager<bool> propagate_encoding(data_manager<bool>& encoding, layer& child_layer);
		data_manager<node_product> propagate_quantifier_up(layer& parent_layer, data_manager<node_product>& merge, data_manager<bool>& parent_stencil, data_manager<bool>& child_stencil);
		void propagate_separation_down(layer& separated, data_manager<bool>& stencil, data_manager<bool>& stencil_separated, data_manager<bool>& child_stencil, data_manager<bool>& child_stencil_separated);
		void propagate_duplication_down(layer& duplicated, data_manager<bool>& stencil, data_manager<bool>& child_stencil, bool bottom);
		void propagate_merge_up(layer& merge_with, layer& parent, layer& merge_with_parent);

		data_manager<uint32_t> remove_elements(std::vector<uint32_t> indices, std::vector<uint32_t> replacement_indices);
		std::vector<uint32_t> remove_elements(std::vector<uint32_t> indices);
		data_manager<uint32_t> insert_elements(data_manager<node_product> elements);
		
		bool is_removable();
		void make_removable_layer(uint32_t number_of_elements);

		void collect_garbage(layer& from);
		bool check_validity();

		uint32_t size();
		bool starts_with_false();
		uint32_t get_child(uint32_t idx, bool high);
		uint32_t get_index_of(node_product p);

		uint16_t get_label();

		void set_data_location(data_location l);
		data_location get_data_location();
		
		#if USE_STORAGE
		bool is_spilled();
		#endif
		
		void update_irreducible_count();
		uint32_t cached_irreducible_count() { return _irreducible_count; }

		std::vector<node_product> get_data();
		void set_data(std::vector<node_product> data);
		void print();
	};
};

#endif