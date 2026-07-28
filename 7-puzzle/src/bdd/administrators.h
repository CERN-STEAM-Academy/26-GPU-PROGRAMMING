#ifndef ADMINISTRATORS
#define ADMINISTRATORS

#include <cstdint>
#include "../data_manager/data_enums.h"

namespace sro{
	struct data_info{
		data_info(unsigned int size, data_location location);
		unsigned int size;
		data_location location;
	};

	struct data_administrator{
		virtual ~data_administrator() = default;
		
		virtual data_location get_data_location(data_info info);
		virtual data_policy get_data_policy(data_info info);
		virtual bool is_default_administrator();
	};

	struct host_administrator : public data_administrator{
		uint32_t switch_to_multi_cpu_at = 5000;

		data_location get_data_location(data_info info) override;
		data_policy get_data_policy(data_info info) override;
		bool is_default_administrator() override;
	};

	struct thrust_administrator : public data_administrator{
		uint32_t switch_to_multi_cpu_at = 5000;
		uint32_t switch_to_gpu_at = 10000;

		data_location get_data_location(data_info info) override;
		data_policy get_data_policy(data_info info) override;
		bool is_default_administrator() override;
	};
};

#endif