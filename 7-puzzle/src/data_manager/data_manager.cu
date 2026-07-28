#include "data_manager.h"

data_manager_base::data_manager_base(std::shared_ptr<sro::data_administrator> admin) : administrator(admin) {}

size_t data_manager_base::size() {
	throw std::runtime_error("Size requested on base class of data_manager.");
}

// size_t data_manager_base::size_bytes() {
// 	throw std::runtime_error("Size in bytes requested on base class of data_manager.");
// }