#ifndef SRO_PRODUCT_LAYER
#define SRO_PRODUCT_LAYER

#include "../layer/layer.h"

namespace sro{
	class arc_layer;

	class product_layer{
		friend class arc_layer;

		data_manager<node_product> products;

		const uint16_t label;

		public:
		std::shared_ptr<data_administrator> administrator;

		product_layer(uint16_t label, std::shared_ptr<data_administrator> admin);

		product_layer(uint16_t label, node_product product, std::shared_ptr<data_administrator> admin);

		product_layer(arc_layer& in);

		product_layer(layer& in, data_manager<bool> participants);

		std::vector<node_product> get_data();
		void set_data(std::vector<node_product> data);
		void print();
	};
};

#endif