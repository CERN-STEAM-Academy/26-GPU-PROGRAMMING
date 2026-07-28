#ifndef SRO_SEMI_ARC_LAYER
#define SRO_SEMI_ARC_LAYER

#include "../arc_layer/arc_layer.h"
#include "../layer/layer.h"
#include "../bdd/bdd.h"
#include "../quantifier/quantified_replacement.h"

#include <thrust/transform.h>

namespace sro{
	struct semi_arc{
		node_product a; // Parent or low child
		node_product b; // Children or high child
	};

	template<typename binary_operation>
	struct transform_arc_to_semi_arc{
		const binary_operation op;
		transform_arc_to_semi_arc(binary_operation _op) : op(_op) {}

		__host__ __device__
		semi_arc operator() (arc in) {
			return {
				in.parent.low,
				in.parent.high,
				op(in.low.low, in.low.high),
				op(in.high.low, in.high.high)
			};
		}
	};
	
	struct sort_on_semi_arc_parents{
		__host__ __device__
		bool operator() (semi_arc a, semi_arc b) const {
			auto a_t = thrust::make_tuple(a.a.high, a.a.low, a.b.high, a.b.low);
			auto b_t = thrust::make_tuple(b.a.high, b.a.low, b.b.high, b.b.low);
			return a_t < b_t;
		}
	};

	class semi_arc_layer{
		friend class layer;

		data_manager<semi_arc> semi_arcs;

		// TODO: make this const
		bool top_heavy; // True if a contains parents. False if a contains low child.

		void sort_on_children();

		template<typename list>
		void sort_on_children(list values);

		void sort_on_parents();

		template<typename list>
		void sort_on_parents(list values) {
			semi_arcs.sort_by_key(values, sort_on_semi_arc_parents());
		}

		data_manager<node_product> merge_semi_arcs_with_layer(layer& l);
		data_manager<uint32_t> fill_semi_arcs(arc_layer& parent_arcs, semi_arc_layer& child_arcs, layer& merge_with);

		public:
		std::shared_ptr<data_administrator> administrator;

		template<typename binary_operation>
		semi_arc_layer(arc_layer& in, binary_operation op) : top_heavy(true), administrator(in.administrator), semi_arcs(in.administrator) {
			semi_arcs.to_location(in.get_data_location());
			semi_arcs.resize(in.arcs.size());
			semi_arcs.with_vector([&](auto policy, auto& semi_arcs_v, auto& arcs_v){
				thrust::transform(
					policy, 
					arcs_v.begin(), arcs_v.end(), 
					semi_arcs_v.begin(), 
					transform_arc_to_semi_arc(op)
				);
			}, in.arcs);

			sort_on_children();

			assert(semi_arcs.size() > 0);
		}

		semi_arc_layer(arc_layer& parent_arcs, semi_arc_layer& child_arcs, layer& merge_with, layer& parent_layer, bool monotonic=false);

		semi_arc_layer(semi_arc_layer& child_arcs, layer& merge_with, bdd& b);
		uint32_t merge_semi_arcs_with_quantifier_layer(layer& child_layer, bdd& b);
		data_manager<node_product> merge_semi_arcs_with_quantifier_layer(layer& child_layer, layer& parent_layer, data_manager<bool> stencil);
		data_manager<quantified_replacement> quantified_replacements();

		void set_data_location(data_location l);
		data_location get_data_location();
		std::vector<semi_arc> get_data();
		void set_data(std::vector<semi_arc> data);
		void print();
	};
};

#endif
