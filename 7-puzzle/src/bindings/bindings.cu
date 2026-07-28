#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <malloc.h>
#include "python/pystreambuf.h"

#include "../bdd/bdd.h"
#include "../layer/layer.h"
#include "../product_layer/product_layer.h"
#include "../arc_layer/arc_layer.h"
#include "../semi_arc_layer/semi_arc_layer.h"

namespace py = pybind11;

// (Re)definition of operators
struct apply_and{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return a && b;
	}
};

struct apply_nand{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return !(a && b);
	}
};

struct apply_or{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return a || b;
	}
};

struct apply_nor{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return !(a || b);
	}
};

struct apply_xor{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return a != b;
	}
};

struct apply_eq{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return a == b;
	}
};

struct apply_difference{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return a && !b;
	}
};

struct apply_not{
	__host__ __device__
	uint32_t operator() (uint32_t a, uint32_t b) const {
		return !a;
	}
};

static size_t gpu_free_mb() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f / (1024 * 1024);
}
static size_t gpu_used_mb() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return (t - f) / (1024 * 1024);
}

namespace pybind11 { namespace detail {

template <typename T>
struct type_caster<data_manager<T>> {
public:
    PYBIND11_TYPE_CASTER(data_manager<T>, _("data_manager"));

    // Python -> C++
    bool load(handle src, bool) {
        py::array_t<T> array = py::cast<py::array_t<T>>(src);
        py::buffer_info info = array.request();
        T* ptr = static_cast<T*>(info.ptr);
        thrust::host_vector<T> h_vec(ptr, ptr + info.size);
        value = data_manager<T>(h_vec);
        return true;
    }

    // C++ -> Python
    static handle cast(const data_manager<T>& manager, return_value_policy, handle) {
		thrust::host_vector<T> h_vec = manager;
        return py::array_t<T>(h_vec.size(), h_vec.data()).release();
    }
};

}} // namespace pybind11::detail

PYBIND11_MODULE(gpudecide, m, py::mod_gil_not_used()) {
	m.doc() = "GPU-accelerated BDD library";
	// Make the default CUDA memory pool release freed memory eagerly,
	// so spilled buffers actually return to the OS instead of being
	// retained at the pool's high-water mark.
	{
		cudaMemPool_t mp;
		if (cudaDeviceGetDefaultMemPool(&mp, 0) == cudaSuccess) {
			uint64_t zero = 0;
			cudaMemPoolSetAttribute(mp, cudaMemPoolAttrReleaseThreshold, &zero);
		}
	}
	
	m.def("gpu_free_mb", &gpu_free_mb, "Free GPU memory (MB) via cudaMemGetInfo");
	m.def("gpu_used_mb", &gpu_used_mb, "Used GPU memory (MB) via cudaMemGetInfo");

	py::enum_<data_location>(m, "DataLocation")
        .value("NONE", NONE)
        .value("HOST", HOST)

#if USE_UNIVERSAL_VECTORS
        .value("UNIVERSAL", UNIVERSAL)
#endif

        .value("DEVICE", DEVICE)
        .value("STORAGE", STORAGE)
        .value("STORAGE_GDS", STORAGE_GDS)
        .export_values();

	py::class_<thrust::device_vector<uint32_t>>(m, "DeviceVectorInt32");

	// Bindings for BDD:
	py::class_<sro::bdd, std::shared_ptr<sro::bdd>>(m, "BDD")
		.def(py::init([](uint16_t var, std::shared_ptr<sro::data_administrator> admin){
			sro::bdd* result = new sro::bdd(var, admin);
			return result->extract_shared_pointer();
		}))
		.def(py::init([](sro::bdd& b){
			sro::bdd* result = new sro::bdd(b);
			return result->extract_shared_pointer();
		}))
		.def("remove_variables", &sro::bdd::remove_variables)
		.def("remove_layers", &sro::bdd::remove_layers)
		.def("insert_removable_layers", &sro::bdd::insert_removable_layers)
		.def("separate_root", &sro::bdd::separate_root)
		.def("separate_roots", &sro::bdd::separate_roots)
		.def("duplicate", &sro::bdd::duplicate)
		.def("join_with", &sro::bdd::join_with)
		.def("exists", py::overload_cast<sro::node_ref, uint32_t>(&sro::bdd::exists))
		.def("exists", py::overload_cast<sro::node_ref, std::vector<uint32_t>>(&sro::bdd::exists))
		.def("for_all", py::overload_cast<sro::node_ref, uint32_t>(&sro::bdd::for_all))
		.def("for_all", py::overload_cast<sro::node_ref, std::vector<uint32_t>>(&sro::bdd::for_all))
		.def("cofactor", &sro::bdd::cofactor)
		.def("projection", &sro::bdd::projection)
		.def("exists_reference", py::overload_cast<sro::node_ref, uint32_t>(&sro::bdd::exists_reference))
		.def("exists_reference", py::overload_cast<sro::node_ref, std::vector<uint32_t>>(&sro::bdd::exists_reference))
		.def("for_all_reference", py::overload_cast<sro::node_ref, uint32_t>(&sro::bdd::for_all_reference))
		.def("for_all_reference", py::overload_cast<sro::node_ref, std::vector<uint32_t>>(&sro::bdd::for_all_reference))
		.def("exists_fused_layered", &sro::bdd::exists_fused_layered)
		.def("for_all_fused_layered", &sro::bdd::for_all_fused_layered)
		.def("exists_fused_persistent", &sro::bdd::exists_fused_persistent)
		.def("for_all_fused_persistent", &sro::bdd::for_all_fused_persistent)
		.def("get_persistent_quantifier_statistics", [](const sro::bdd& bdd) {
			const auto& statistics = bdd.get_persistent_quantifier_statistics();
			py::dict result;
			result["states_per_layer"] = statistics.states_per_layer;
			result["peak_state_count"] = statistics.peak_state_count;
			result["peak_state_layer"] = statistics.peak_state_layer;
			result["peak_layer_member_count"] = statistics.peak_layer_member_count;
			result["peak_layer_member_count_layer"] = statistics.peak_layer_member_count_layer;
			return result;
		})
		.def("logical_and", &sro::bdd::logical_and)
		.def("logical_nand", &sro::bdd::logical_nand)
		.def("logical_or", &sro::bdd::logical_or)
		.def("logical_nor", &sro::bdd::logical_nor)
		.def("logical_xor", &sro::bdd::logical_xor)
		.def("logical_eq", &sro::bdd::logical_eq)
		.def("logical_difference", &sro::bdd::logical_difference)
		.def("logical_implication", &sro::bdd::logical_implication)
		.def("logical_not", &sro::bdd::logical_not)
		.def("logical_ite", &sro::bdd::logical_ite)
		.def("logical_true", &sro::bdd::logical_true)
		.def("logical_false", &sro::bdd::logical_false)
		.def("__getitem__", &sro::bdd::get_variable)
		.def("get_layer", &sro::bdd::get_layer, py::return_value_policy::reference_internal)
		.def("count_nodes", &sro::bdd::count_nodes)
		.def("count_irreducible_nodes", &sro::bdd::count_irreducible_nodes)
		.def("count_satisfying_assignments", &sro::bdd::count_satisfying_assignments)
		.def("get_all_node_refs", &sro::bdd::get_all_node_refs)
		.def("print", &sro::bdd::print)
		.def("print_roots", &sro::bdd::print_roots)
		// .def("input_dddmp", &sro::bdd::input_dddmp)
		// .def("output_dddmp", &sro::bdd::output_dddmp)
    	.def_readwrite("administrator", &sro::bdd::administrator);

	py::class_<sro::data_administrator, std::shared_ptr<sro::data_administrator>>(m, "data_administrator");

	py::class_<sro::host_administrator, sro::data_administrator, std::shared_ptr<sro::host_administrator>>(m, "host_administrator")
		.def(py::init())
		.def("__repr__",
        [](const sro::host_administrator &a) {
            return "<sro::host_administrator>";
        })
    	.def_readwrite("switch_to_multi_cpu_at", &sro::host_administrator::switch_to_multi_cpu_at);

	py::class_<sro::thrust_administrator, sro::data_administrator, std::shared_ptr<sro::thrust_administrator>>(m, "thrust_administrator")
		.def(py::init())
		.def("__repr__",
        [](const sro::thrust_administrator &a) {
            return "<sro::thrust_administrator>";
        })
    	.def_readwrite("switch_to_multi_cpu_at", &sro::thrust_administrator::switch_to_multi_cpu_at)
    	.def_readwrite("switch_to_gpu_at", &sro::thrust_administrator::switch_to_gpu_at);

	py::class_<sro::node_ref>(m, "NodeRef")
		.def(py::init<uint32_t, std::weak_ptr<sro::bdd>>())
		.def(py::init([](uint32_t id, std::shared_ptr<sro::bdd> b){
			std::weak_ptr<sro::bdd> bdd_ptr = b;
			return sro::node_ref(id, bdd_ptr);
		}))
		.def(py::init([](sro::node_ref ref, std::shared_ptr<sro::bdd> b){
			std::weak_ptr<sro::bdd> bdd_ptr = b;
			return sro::node_ref(ref.get_id(), bdd_ptr);
		}))
		.def("get_id", py::overload_cast<>(&sro::node_ref::get_id))
		.def("__iter__",
			[](sro::node_ref &c) {
				return py::make_iterator(c.begin(), c.end());
			},
			py::keep_alive<0,1>()
		)
		.def("get_metric", &sro::node_ref::get_metric);
	
	// Bindings for layer
	py::class_<sro::layer>(m, "Layer")
		.def("apply_map", py::overload_cast<data_manager<uint32_t>&>(&sro::layer::apply_map))
		.def("propagate_encoding", &sro::layer::propagate_encoding)
		.def("remove_elements", py::overload_cast<std::vector<uint32_t>, std::vector<uint32_t>>(&sro::layer::remove_elements))
		.def("remove_elements", py::overload_cast<std::vector<uint32_t>>(&sro::layer::remove_elements))
		.def("is_removable", &sro::layer::is_removable)
		.def("make_removable_layer", &sro::layer::make_removable_layer)
		.def("check_validity", &sro::layer::check_validity)
		.def("size", &sro::layer::size)
		.def("set_data_location", &sro::layer::set_data_location)
    	.def("get_data_location", &sro::layer::get_data_location)
		.def("get_data", &sro::layer::get_data)
		.def("set_data", &sro::layer::set_data)
		.def("print", &sro::layer::print);
	py::class_<sro::node_product>(m, "NodeProduct")
		.def_readwrite("low", &sro::node_product::low)
		.def_readwrite("high", &sro::node_product::high);
	
	// Bindings for arc layer
	py::class_<sro::arc_layer>(m, "ArcLayer")
		.def(py::init<sro::product_layer&, sro::layer&>())
		.def("get_data", &sro::arc_layer::get_data)
		.def("set_data", &sro::arc_layer::set_data)
		.def("print", &sro::arc_layer::print);
	py::class_<sro::arc>(m, "Arc")
		.def_readwrite("parent", &sro::arc::parent)
		.def_readwrite("low", &sro::arc::low)
		.def_readwrite("high", &sro::arc::high);

	// Bindings for product layer
	py::class_<sro::product_layer>(m, "ProductLayer")
		.def(py::init<uint16_t, sro::node_product, std::shared_ptr<sro::data_administrator>>())
		.def(py::init<sro::arc_layer&>())
		.def("get_data", &sro::product_layer::get_data)
		.def("set_data", &sro::product_layer::set_data)
		.def("print", &sro::product_layer::print);
	
	// Bindings for logical operators
	py::class_<apply_and>(m, "And")
		.def(py::init());
	py::class_<apply_nand>(m, "Nand")
		.def(py::init());
	py::class_<apply_or>(m, "Or")
		.def(py::init());
	py::class_<apply_nor>(m, "Nor")
		.def(py::init());
	py::class_<apply_xor>(m, "Xor")
		.def(py::init());
	py::class_<apply_eq>(m, "Eq")
		.def(py::init());
	py::class_<apply_difference>(m, "Difference")
		.def(py::init());
	py::class_<apply_not>(m, "Not")
		.def(py::init());

	// Bindings for semi arc layer
	py::class_<sro::semi_arc_layer>(m, "SemiArcLayer")
		.def(py::init<sro::arc_layer&, apply_and>())
		.def(py::init<sro::arc_layer&, apply_nand>())
		.def(py::init<sro::arc_layer&, apply_or>())
		.def(py::init<sro::arc_layer&, apply_xor>())
		.def(py::init<sro::arc_layer&, apply_eq>())
		.def(py::init<sro::arc_layer&, apply_difference>())
		.def(py::init<sro::arc_layer&, apply_not>())
		.def(py::init<sro::arc_layer&, sro::semi_arc_layer&, sro::layer&, sro::layer&>())
		.def(py::init<sro::semi_arc_layer&, sro::layer&, sro::bdd&>())
		.def("get_data", &sro::semi_arc_layer::get_data)
		.def("set_data", &sro::semi_arc_layer::set_data)
		.def("print", &sro::semi_arc_layer::print);
	py::class_<sro::semi_arc>(m, "SemiArc")
		.def_readwrite("a", &sro::semi_arc::a)
		.def_readwrite("b", &sro::semi_arc::b);
}
