#ifndef SRO_ARC_LAYER
#define SRO_ARC_LAYER

#include "../layer/layer.h"
#include "../data_manager/data_manager.h"

namespace sro{
	struct arc{
		node_product parent;
		node_product low;
		node_product high;
	};

	class product_layer;
	class semi_arc_layer;

	class arc_layer{
		friend class product_layer;
		friend class semi_arc_layer;

		data_manager<arc> arcs;
		
		const uint16_t label;

		public:
		std::shared_ptr<data_administrator> administrator;

		arc_layer(uint16_t label, std::shared_ptr<data_administrator> admin);
		arc_layer(product_layer& in, layer& layer, bool quantifier = false);

		uint32_t size();
		
		void set_data_location(data_location l);
		data_location get_data_location();

		#if USE_STORAGE
		bool is_spilled();
		#endif
		
		std::vector<arc> get_data();
		void set_data(std::vector<arc> data);
		void print();
	};
};

#endif