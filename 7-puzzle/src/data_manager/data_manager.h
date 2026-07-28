#ifndef DATA_MANAGER
#define DATA_MANAGER

#include <cstdio>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/omp/vector.h>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/unique.h>

#include <thrust/tuple.h>
#include <tuple>
#include <utility>
#include <iostream>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <cstdlib>
#include <malloc.h>

#include "../spill/spill_io.h"  

// #include "system/benchmark/execution_policy.h"
#include "../system/multicore_cpu/detail/execution_policy.h"
#include "../timer_policy.h"
#include "../multi_gpu/multi_gpu_policy.h"
#include "../multi_gpu/multi_gpu_vector.h"

#include "../bdd/administrators.h"
#include "data_enums.h"

// Alternative for std::apply that works on device without warnings
template <typename F, typename Tuple, std::size_t... I>
__device__ __host__ auto device_apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>)
{
	return f(thrust::get<I>(t)...);  // Use thrust::get for tuples in device code
}

template <typename F, typename Tuple>
__device__ __host__ auto device_apply(F&& f, Tuple&& t)
{
	constexpr auto N = std::tuple_size<std::remove_reference_t<Tuple>>::value;
	return device_apply_impl(
		std::forward<F>(f),
		std::forward<Tuple>(t),
		std::make_index_sequence<N>{}
	);
}

// Template type replacer
// TODO: replace this with generic type find and replace
template<typename OldVector, typename NewT>
struct rebind_vector;

// IGNORE

// host_vector
template<typename OldT, typename Allocator, typename NewT>
struct rebind_vector<thrust::host_vector<OldT, Allocator>, NewT> {
	using type = thrust::host_vector<NewT, typename std::allocator_traits<Allocator>::template rebind_alloc<NewT>>;
};

// device_vector
template<typename OldT, typename Allocator, typename NewT>
struct rebind_vector<thrust::device_vector<OldT, Allocator>, NewT> {
	using type = thrust::device_vector<NewT, typename std::allocator_traits<Allocator>::template rebind_alloc<NewT>>;
};

// universal_vector
template<typename OldT, typename Allocator, typename NewT>
struct rebind_vector<thrust::universal_vector<OldT, Allocator>, NewT> {
	using type = thrust::universal_vector<NewT, typename std::allocator_traits<Allocator>::template rebind_alloc<NewT>>;
};

// multi_gpu_vector
template<typename OldT, typename NewT>
struct rebind_vector<multi_gpu_vector<OldT>, NewT> {
	using type = multi_gpu_vector<NewT>;
};

// helper alias
template<typename OldVector, typename NewT>
using rebind_vector_t = typename rebind_vector<OldVector, NewT>::type;

// Device code helpers
struct minus{
	__host__ __device__
	uint32_t operator() (thrust::tuple<uint32_t, uint32_t> in) {
		return thrust::get<0>(in) - thrust::get<1>(in);
	}
};

// Template type extractor
template<typename Vector>
struct vector_value_type;

// host_vector
template<typename T, typename Allocator>
struct vector_value_type<thrust::host_vector<T, Allocator>> {
	using type = T;
};

// device_vector
template<typename T, typename Allocator>
struct vector_value_type<thrust::device_vector<T, Allocator>> {
	using type = T;
};

// universal_vector
template<typename T, typename Allocator>
struct vector_value_type<thrust::universal_vector<T, Allocator>> {
	using type = T;
};

// multi_gpu_vector
template<typename T> 
struct vector_value_type<multi_gpu_vector<T>> {
	using type = T;
};

template<typename Vector>
using vector_value_type_t = typename vector_value_type<Vector>::type;

// Generic class function wrapper
template <auto Func, typename... BoundArgs>
struct Wrapper {
	thrust::tuple<BoundArgs...> args;

	Wrapper(BoundArgs... bound_args) : args(bound_args...) {}

	template <typename T>
	__host__ __device__
	auto operator()(T&& obj) const {
		return device_apply([&](auto&&... a) { return (obj.*Func)(a...); }, args);
	}
};

inline data_location most_general_data_compatibility(data_location l) {
	return l;
}

template<typename... Args> // IGNORE
inline data_location most_general_data_compatibility(data_location a, data_location b, Args... args) {
	data_location other = most_general_data_compatibility(b, args...);
	switch(a) {
	case HOST:
		switch (other) {
		case HOST:
		#if USE_UNIVERSAL_VECTORS
		case UNIVERSAL:
		#endif
			return HOST;
		default:
			return NONE;
		}
	
	#if USE_UNIVERSAL_VECTORS
	case UNIVERSAL:
		switch (other) {
		case HOST:
		case UNIVERSAL:
		case DEVICE:
			return other;
		default:
			return NONE;
		}
	#endif

	case DEVICE:
		switch (other) {

		#if USE_UNIVERSAL_VECTORS
		case UNIVERSAL:
		#endif

		case DEVICE:
			return DEVICE;
		default:
			return NONE;
		}
	default:
		return NONE;
	}
}

// Data manager defined later
template<typename T>
class data_manager;

// Reference to data manager
template<typename T>
class data_ref{
	data_manager<T>& m;
	const size_t idx;
	public:
	data_ref(data_manager<T>& _m, size_t _idx) : m(_m), idx(_idx) {}

	data_ref& operator=(const data_ref& other) {
		if (this != &other) {
			T eval = other.m.with_vector([&](auto policy, auto const& v) -> T {
				return v[other.idx];
			});

			*this = eval;
		}
		return *this;
	}

	void operator=(T value) {
		m.with_vector([&](auto policy, auto& v){
			v[idx] = value;
		});
	}

	operator T() {
		return get_data();
	}

	T get_data() {
		return m.with_vector([&](auto policy, auto const& v) -> T {
			return v[idx];
		});
	}
};


// Iterator on data manager
template<typename T>
class data_iterator{
	data_manager<T>& m;
	size_t idx;

	public:
	data_iterator(data_manager<T>& _m, size_t _idx) : m(_m), idx(_idx) {}

	data_ref<T> operator*() {
		return m[idx];
	}

	void operator++() {
		idx++;
	}

	bool operator==(data_iterator<T>& other) {
		return (idx == other.idx);
	}
};

class data_manager_base{
public:

	std::shared_ptr<sro::data_administrator> administrator;
	virtual size_t size();

	// virtual size_t size_bytes(); // Size of the array in bytes

	explicit data_manager_base(std::shared_ptr<sro::data_administrator> admin = std::make_shared<sro::thrust_administrator>());
};

// Data manager
template<typename T>
class data_manager : public data_manager_base{
	thrust::omp::vector<T> omp; 
	thrust::host_vector<T> h;
	
	#if USE_UNIVERSAL_VECTORS
	thrust::universal_vector<T> u;
	#endif

	thrust::device_vector<T> d;
	
	#if USE_MULTI_GPU
	multi_gpu_vector<T> mg;
	#endif

	// Disk-backed storage state (used when l == STORAGE_GDS or STORAGE)
	struct storage_handle {
		std::string path;                       // file path on disk
		size_t element_count = 0;               // number of T elements stored
		int fd = -1;                            // POSIX file descriptor
		CUfileHandle_t cufile_handle = nullptr; // cuFile handle (STORAGE_GDS only)
		data_location original_location = HOST; // where to restore to
	};
	storage_handle s;

	template<typename>
	friend class data_manager;

	data_location l;

	template<typename C> 
	data_location get_data_location_from(const C& vec) {
		using Vec = std::remove_reference_t<C>;
		using BoolVec = rebind_vector_t<Vec, bool>;
		if constexpr (std::is_same_v<BoolVec, data_manager<bool>>) {
			printf("get_data_location_from: HOST\n");
			return vec.l;
		}
		printf("get_data_location_from: DEVICE\n");
		return DEVICE; 
	}

	#if USE_STORAGE
	// Release disk-backed storage resources (cuFile handle, fd, file).
	// No operation if not spilled. Does NOT reset s/l the caller handles that.
	void release_storage() {
		if (l != STORAGE_GDS && l != STORAGE) return;
		if (l == STORAGE_GDS && s.cufile_handle) cuFileHandleDeregister(s.cufile_handle);
		if (s.fd >= 0)       close(s.fd);
		if (!s.path.empty()) unlink(s.path.c_str());
	}
	#endif

	public:
	enum class policy{
		host,
		device
	};

	template <typename F>
	decltype(auto) with_vector_no_policy(F&& f) {
		#if USE_STORAGE
		// Auto-restore from disk if necessary
		if (l == STORAGE_GDS || l == STORAGE) {
			restore_to_original();
		}
		#endif

		switch (l) {
		case HOST:
			return f(h);
		
		#if USE_UNIVERSAL_VECTORS
		case UNIVERSAL:
			return f(u);
		#endif
		
		#if USE_MULTI_GPU
		case MULTI_GPU:
			printf("with_vector_no_policy(F&& f)");
			assert(false);
		// 	return f(mg);
		#endif

		case DEVICE:
			return f(d);
		}
		assert(false);
		throw std::logic_error("Data locality not yet implemented.");
	}

	template <data_manager<T>::policy Mode, typename F>
	decltype(auto) with_vector(F&& f) {
		#if USE_STORAGE
		// Auto-restore from disk if necessary
		if (l == STORAGE_GDS || l == STORAGE) {
			restore_to_original();
		}
		#endif

		switch (l) {
		case HOST:
			if constexpr(Mode == data_manager<T>::policy::host) {
				sro::data_info info(size(), l);
				data_policy p = administrator->get_data_policy(info);

				if (p == SINGLE_CORE_CPU) {
					// return f(thrust::measured_host, h);
					return f(thrust::host, h);
				}
				else if (p == MULTI_CORE_CPU) {
					return f(thrust::omp::par, h);
				}
				assert(false);
				throw std::logic_error("Data is on CPU, but policy is not.");
			}
			printf("with_vector: bad HOST\n");
			assert(false);
			throw std::logic_error("Conflicting data locality.");

		#if USE_UNIVERSAL_VECTORS
		case UNIVERSAL:
			return f(thrust::device, u);
		#endif

		#if USE_MULTI_GPU
		case MULTI_GPU:
			printf("with_vector(F&& f)->MULTI_GPU\n");
			assert(false);
		// 	return f(thrust::multi_gpu_policy, mg);
		#endif

		case DEVICE:
			// if constexpr(Mode == data_manager<T>::policy::device) return f(thrust::measured_device, d);
			if constexpr(Mode == data_manager<T>::policy::device) return f(thrust::device, d);
			printf("with_vector: bad DEVICE\n");
			assert(false);
			throw std::logic_error("Conflicting data locality.");
		}
		assert(false);
		throw std::logic_error("Data locality not yet implemented.");
	}

	template <data_manager<T>::policy Mode, typename F, typename D, typename... Args>
	decltype(auto) with_vector(F&& f, D& data, Args&... args) {
		#if USE_STORAGE
		if (data.l == STORAGE_GDS || data.l == STORAGE) {
			data.restore_to_original();
		}
		#endif

		switch (data.l) {
		case HOST:
			if constexpr(Mode == data_manager<T>::policy::host) {
				return with_vector<Mode>([&](auto p, auto& self_vec, auto&... l_args){
					return f(p, self_vec, data.h, l_args...);
				}, args...);
			}
			assert(false);
			throw std::logic_error("Conflicting data locality.");
		
		#if USE_UNIVERSAL_VECTORS
		case UNIVERSAL:
			return with_vector<Mode>([&](auto p, auto& self_vec, auto&... l_args){
				return f(p, self_vec, data.u, l_args...);
			}, args...);
		#endif
		
		#if USE_MULTI_GPU
		case MULTI_GPU:
			printf("with_vector(...)->MULTI_GPU\n");
			assert(false);
			// return with_vector<Mode>([&](auto p, auto& self_vec, auto&... l_args){
			// 			return f(p, self_vec, data.mg, l_args...);
			// 		}, args...);
		#endif

		case DEVICE:
			if constexpr(Mode == data_manager<T>::policy::device) {
				return with_vector<Mode>([&](auto p, auto& self_vec, auto&... l_args){
					return f(p, self_vec, data.d, l_args...);
				}, args...);
			}
			assert(false);
			throw std::logic_error("Conflicting data locality.");
		}
		assert(false);
		throw std::logic_error("Data locality not yet implemented.");
	}

	public:	
	template<typename V>
	V& vec() { // Can't fix, does this cause problems?
		if constexpr (std::is_same_v<V, thrust::host_vector<T>>) return h;
		else if constexpr (std::is_same_v<V, thrust::device_vector<T>>) return d;

		#if USE_UNIVERSAL_VECTORS
		else if constexpr (std::is_same_v<V, thrust::universal_vector<T>>) return u;
		#endif

		// else if constexpr (std::is_same_v<V, multi_gpu_vector<T>>) return mg;
		// else if constexpr (std::is_same_v<V, thrust::device_vector<T>>) return mg;
		else assert(false);
	}

	template <typename F>
	decltype(auto) with_vector(F&& f) {
		#if USE_STORAGE
		// Auto-restore from disk if necessary
		if (l == STORAGE_GDS || l == STORAGE) {
			restore_to_original();
		}
		#endif

		switch (l) {
		case HOST:
			{
				data_policy policy = administrator->get_data_policy(sro::data_info(size(), HOST));

				if (policy == SINGLE_CORE_CPU) {
					return f(thrust::host, h);
				}
				else if (policy == MULTI_CORE_CPU) {
					// return f(thrust::measured_host, h);
					return f(thrust::omp::par, h);
				}
			}
		
		#if USE_UNIVERSAL_VECTORS
		case UNIVERSAL:
			return f(thrust::device, u);
		#endif
		
		#if USE_MULTI_GPU
		case MULTI_GPU:
			printf("with_vector(...)->MULTI_GPU");
			assert(false);
			// return f(multi_gpu_policy, mg);
		#endif

		case DEVICE:
			// return f(thrust::measured_device, d);
			return f(thrust::device, d);
		}
		assert(false);
		throw std::logic_error("Incorrect data locality.");
	}
	
	template <typename F, typename D, typename... Args>
	decltype(auto) with_vector(F&& f, D& data, Args&... args) {
		#if USE_STORAGE
		// Auto-restore the other data manager if needed too
		if (data.l == STORAGE_GDS || data.l == STORAGE) {
			data.restore_to_original();
		}
		#endif

		switch (data.l) {
		case HOST:
			return with_vector<data_manager<T>::policy::host>([&](auto p, auto& self_vec, auto&... l_args){
				return f(p, self_vec, data.h, l_args...);
			}, args...);
		
		#if USE_UNIVERSAL_VECTORS
		case UNIVERSAL:
			return with_vector([&](auto p, auto& self_vec, auto&... l_args){
				return f(p, self_vec, data.u, l_args...);
			}, args...);
		#endif
		
		#if USE_MULTI_GPU
		case MULTI_GPU:
			printf("with_vector(...)->MULTI-GPU\n");
			assert(false);
			// return with_vector<data_manager<T>::policy::device>([&](auto p, auto& self_vec, auto&... l_args){
			// 	return f(p, self_vec, data.mg, l_args...);
			// }, args...);
		#endif

		case DEVICE:
			return with_vector<data_manager<T>::policy::device>([&](auto p, auto& self_vec, auto&... l_args){
				return f(p, self_vec, data.d, l_args...);
			}, args...);
		}
		assert(false);
		throw std::logic_error("Incorrect data locality.");
	}

	data_manager(std::shared_ptr<sro::data_administrator> admin=std::make_shared<sro::thrust_administrator>())
		: data_manager_base(admin)
		, l(HOST) {}

	data_manager(
		data_manager&& other, 
		std::shared_ptr<sro::data_administrator> admin
	) noexcept
		: data_manager_base(admin)
		, omp(std::move(other.omp))
		, h(std::move(other.h))
#if USE_UNIVERSAL_VECTORS
		, u(std::move(other.u))
#endif
		, d(std::move(other.d))
		, s(std::move(other.s))
		, l(other.l)
	{
		other.l = NONE;
		other.s.fd = -1;
		other.s.cufile_handle = nullptr;
		other.s.element_count = 0;
		other.s.path.clear();
	}

	data_manager& operator=(data_manager&& other) noexcept {
		if (this != &other) {
			#if USE_STORAGE
			// Clean up own state if needed
			release_storage();
			#endif
			
			omp = std::move(other.omp);
			h = std::move(other.h);

			#if USE_UNIVERSAL_VECTORS
			u = std::move(other.u);
			#endif

			d = std::move(other.d);
			s = std::move(other.s);
			l = other.l;
			
			other.l = NONE;
			other.s.fd = -1;
			other.s.cufile_handle = nullptr;
			other.s.element_count = 0;
			other.s.path.clear();

			administrator = other.administrator;
		}
		return *this;
	}

	// Copy constructor — restores a spilled source before copying
	data_manager(const data_manager& other) : data_manager_base(other.administrator) {
		#if USE_STORAGE
		if (other.l == STORAGE_GDS || other.l == STORAGE) {
			const_cast<data_manager&>(other).restore_to_original();
		}
		#endif

		omp = other.omp;
		h = other.h;

		#if USE_UNIVERSAL_VECTORS
		u = other.u;
		#endif

		d = other.d;
		l = other.l;
	}

	data_manager& operator=(const data_manager& other) {
		if (this == &other) return *this;

		#if USE_STORAGE
		if (other.l == STORAGE_GDS || other.l == STORAGE) {
			const_cast<data_manager&>(other).restore_to_original();
		}

		// Clean up own state if it's in storage
		release_storage();
		#endif

		omp = other.omp;
		h = other.h;

		#if USE_UNIVERSAL_VECTORS
		u = other.u;
		#endif

		d = other.d;
		l = other.l;
		s = storage_handle{};

		administrator = other.administrator;

		return *this;
	}

	template <typename C>
	requires (
		!std::is_same_v<std::decay_t<C>, data_manager> &&
		!std::is_integral_v<std::decay_t<C>>
	)
	data_manager(const C& copy_from) {
		throw std::runtime_error("copy.");

		l = get_data_location_from(copy_from);
		with_vector_no_policy([&](auto& v){
			v = copy_from;
		});
	}

	explicit data_manager(
		data_location _l, 
		size_t size,
		std::shared_ptr<sro::data_administrator> admin
	)
		: data_manager_base(admin)
	{
		l = _l;
		with_vector([&](auto policy, auto& v){
			v.resize(size);
		});
	}

	explicit data_manager(
		data_location _l, 
		size_t size, 
		T initial_value,
		std::shared_ptr<sro::data_administrator> admin
	)
		: data_manager_base(admin)
	{
		l = _l;
		with_vector([&](auto policy, auto& v){
			using Vec = std::remove_reference_t<decltype(v)>;
			v = Vec(size, initial_value);
		});
	}

	data_manager(
		data_location _l,
		std::shared_ptr<sro::data_administrator> admin
	)
		: data_manager_base(admin)
	{
		l = _l;
	}

	/**
	 * Creates a data manager of the same size in the same location, and fils it with overwrite
	 */
	data_manager(data_manager& from, T overwrite) : data_manager_base(from.administrator) {
		assert(false);
	}

	~data_manager() {
		// Only STORAGE locations need explicit cleanup.
		// HOST/UNIVERSAL/DEVICE are managed by thrust's RAII destructors.
		#if USE_STORAGE
		release_storage();
		#endif
	}

	// Functions for changing location
	void to_host() {
		switch(l) {
		case HOST:
			return;
		
		#if USE_UNIVERSAL_VECTORS
		case UNIVERSAL:
			h = u;
			u.clear();
			return;
		#endif
		
		#if USE_MULTI_GPU
		case MULTI_GPU:
			h = mg;
			mg.clear();
			l = HOST;
			return;
		#endif

		case DEVICE:
			h = d;
			d.clear();
			return;
		case STORAGE:
			assert(false); // TODO

		#if USE_STORAGE
		case STORAGE_GDS:
			to_device();           // brings data from disk into d
			h = d;                 // copy device → host
			d.clear();
			d.shrink_to_fit();
			l = HOST;
			return;
		#endif
		}
	}

	#if USE_UNIVERSAL_VECTORS
	void to_universal() {
		switch(l) {
		case HOST:
			d = h;
			h.clear();
			h.shrink_to_fit();
			return;
		case UNIVERSAL:
			return;
		case DEVICE:
			h = d;
			d.clear();
			d.shrink_to_fit();
			return;
		
		#if USE_MULTI_GPU
		case MULTI_GPU:
			printf("to_universal()->MULTI_GPU\n");
			assert(false);
		#endif

		#if USE_STORAGE
		case STORAGE:
			assert(false); // TODO
		case STORAGE_GDS:
			to_device();
			u = d;
			d.clear();
			d.shrink_to_fit();
			l = UNIVERSAL;
		return;
		#endif
		}
	}
	#endif

	#if USE_MULTI_GPU
	void to_multi_gpu() {
		printf("to_multi_gpu()\n");
		switch(l) {
		case HOST:
			mg = h;
			h.clear();
			break;
		case MULTI_GPU:
			break;
		case DEVICE:
			mg = d;
			d.clear();
			break;

		#if USE_UNIVERSAL_VECTORS
		case UNIVERSAL:
			assert(false);
			break;
		#endif

		#if USE_STORAGE
		case STORAGE:
			assert(false);
			break;
		case STORAGE_GDS:
			assert(false);
			break;
		#endif

		default:
			assert(false);
			return;
		}
		l = MULTI_GPU;
	}
	#endif

	void to_device() {
		switch(l) {
		case HOST:
			d = h;
			h.clear();
			l = DEVICE;
			h.shrink_to_fit();
			return;
		
		#if USE_UNIVERSAL_VECTORS
		case UNIVERSAL:
			d = u;
			u.clear();
			u.shrink_to_fit();
			return;
		#endif
		
		case DEVICE:
			l = DEVICE;
			return;

		#if USE_MULTI_GPU
		case MULTI_GPU:
			d = mg;
			mg.clear();
			l = DEVICE;
			// printf("MULT -> DEVICE\n");
			return;
		#endif

		#if USE_STORAGE
		case STORAGE:
			assert(false); // TODO
		case STORAGE_GDS:
			{
				// Handle empty case — no file to read from
				if (s.element_count == 0) {
					d.clear();
					l = DEVICE;
					return;
				}

				// Re-open + re-register the file (handle/fd were released at spill time)
				s.fd = open(s.path.c_str(), O_RDWR | O_DIRECT);
				if (s.fd < 0) {
					throw std::runtime_error("Failed to reopen spill file: " + s.path);
				}
				CUfileDescr_t desc = {};
				desc.handle.fd = s.fd;
				desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
				CUfileError_t reg_err = cuFileHandleRegister(&s.cufile_handle, &desc);
				if (reg_err.err != CU_FILE_SUCCESS) {
					close(s.fd);
					s.fd = -1;
					throw std::runtime_error("cuFileHandleRegister failed on restore");
				}

				// Allocate the device buffer to read into
				d.resize(s.element_count);
				void* device_ptr = thrust::raw_pointer_cast(d.data());
				size_t byte_count = s.element_count * sizeof(T);

				// Register the device buffer explicitly (same reason as the write path)
				bool buf_registered =
					(cuFileBufRegister(device_ptr, byte_count, 0).err == CU_FILE_SUCCESS);

				try {
					cuFile_read_all(s.cufile_handle, device_ptr, byte_count);
				} catch (...) {
					if (buf_registered) cuFileBufDeregister(device_ptr);
					cuFileHandleDeregister(s.cufile_handle);
					s.cufile_handle = nullptr;
					close(s.fd);
					s.fd = -1;
					throw;
				}

				// Release buffer registration, handle, fd; delete the file
				if (buf_registered) cuFileBufDeregister(device_ptr);
				cuFileHandleDeregister(s.cufile_handle);
				s.cufile_handle = nullptr;
				close(s.fd);
				s.fd = -1;
				unlink(s.path.c_str());
				s.path.clear();
				s.element_count = 0;

				l = DEVICE;
				return;
			}
		#endif
		}
	}

	#if USE_STORAGE
	void to_storage_gds() { // Future for Multi GPU
		if (l == STORAGE_GDS) return;

		// capture original location before to_device() changes it
		data_location original = l;

		// ensure data is on device
		if (l != DEVICE) to_device();

		// handle empty data without creating a file
		if (d.size() == 0) {
			s.element_count = 0;
			s.path.clear();
			s.fd = -1;
			s.cufile_handle = nullptr;
			s.original_location = original;
			l = STORAGE_GDS;
			return;
		}

		// unique spill path
		s.path = spill_directory() + "/gpudecide_spill_"
			+ std::to_string(getpid()) + "_"
			+ std::to_string(reinterpret_cast<uintptr_t>(this)) + ".bin";

		// open file
		s.fd = open(s.path.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0664);
		if (s.fd < 0) {
			throw std::runtime_error("Failed to open spill file: " + s.path);
		}

		// register file with cuFile
		CUfileDescr_t desc = {};
		desc.handle.fd = s.fd;
		desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

		CUfileError_t reg_err = cuFileHandleRegister(&s.cufile_handle, &desc);
		if (reg_err.err != CU_FILE_SUCCESS) {
			close(s.fd);
			s.fd = -1;
			throw std::runtime_error("cuFileHandleRegister failed");
		}

		void* device_ptr = thrust::raw_pointer_cast(d.data());
		size_t byte_count = d.size() * sizeof(T);

		bool buf_registered =
			(cuFileBufRegister(device_ptr, byte_count, 0).err == CU_FILE_SUCCESS);

		bool dbg = std::getenv("GPUDECIDE_SPILL_DEBUG");

		auto fmb = []() {
			size_t f = 0, t = 0;
			cudaMemGetInfo(&f, &t);
			return (long)(f / (1024 * 1024)); // MB free
		};

		long f_before_write = dbg ? fmb() : 0;

		// write GPU → NVMe
		try {
			cuFile_write_all(s.cufile_handle, device_ptr, byte_count);
		} catch (...) {
			if (buf_registered) cuFileBufDeregister(device_ptr);
			cuFileHandleDeregister(s.cufile_handle);
			s.cufile_handle = nullptr;
			close(s.fd);
			s.fd = -1;
			throw;
		}

		long f_after_write = dbg ? fmb() : 0;

		// record metadata
		s.element_count = d.size();

		// teardown cuFile handles BEFORE freeing memory
		if (buf_registered) cuFileBufDeregister(device_ptr);
		cuFileHandleDeregister(s.cufile_handle);
		s.cufile_handle = nullptr;
		close(s.fd);
		s.fd = -1;

		// free GPU memory
		d.clear();
		d.shrink_to_fit();

		reclaim_device_memory();

		long f_after_free = dbg ? fmb() : 0;

		if (dbg) {
			std::cerr
				<< "[gds] tgt=" << (byte_count / (1024 * 1024)) << "MB "
				<< "write_cost=" << (f_before_write - f_after_write) << "MB "
				<< "free_gain=" << (f_after_free - f_after_write) << "MB"
				<< std::endl;
		}

		// finalize state
		s.original_location = original;
		l = STORAGE_GDS;
	}
	#endif

	// Restore from storage back to whatever location the data was at before spilling
	void restore_to_original() {
		if (l != STORAGE_GDS && l != STORAGE) return;
		
		switch (s.original_location) {
			case HOST:
				to_host();
				h.shrink_to_fit(); 
				break;
			
			#if USE_UNIVERSAL_VECTORS
			case UNIVERSAL:
				to_universal();
				break;
			#endif

			case DEVICE:
			default:
				to_device();
				break;
		}
		reclaim_device_memory();
	}

	void to_location(data_location _l) {
		if (_l == l) return;

		#if USE_STORAGE
		if (l == STORAGE_GDS || l == STORAGE) {
			restore_to_original();
		}
		
		if (_l == STORAGE_GDS) {
			to_storage_gds();
			return;
		}
		if (_l == STORAGE) {
			throw std::runtime_error("STORAGE backend not yet implemented");
		}
		#endif

		with_vector_no_policy([&](auto& v){
			switch(_l) {
			case HOST:
				h = v;
				return;


			#if USE_UNIVERSAL_VECTORS
			case UNIVERSAL:
				u = v;
				return;
			#endif
			
			#if USE_MULTI_GPU
			case MULTI_GPU:
				printf("to_location->MULTI_GPU\n");
				mg = v;
				l = MULTI_GPU;
				// assert(false);
				return;
			#endif

			case DEVICE:
				d = v;
				return;
			default:
				throw std::runtime_error("Location conversion not yet supported.");
			}
		});
		l = _l;
	}

	data_location get_data_location() {
		#if USE_STORAGE
		if (l == STORAGE_GDS || l == STORAGE) {
			return s.original_location;
		}
		#endif

		return l;
	}

	#if USE_STORAGE
	// True spill predicate. get_data_location() deliberately hides spill state
	bool is_spilled() const {
		return l == STORAGE_GDS || l == STORAGE;
	}
	#endif

	// Iterators
	auto begin() {
		return data_iterator(*this, 0);
	}
	
	auto end() {
		return data_iterator(*this, size());
	}

	// Generic vector functions
	size_t size() override {
		#if USE_STORAGE
		if (l == STORAGE_GDS || l == STORAGE) {
			return s.element_count;
		}
		#endif

		return with_vector_no_policy([](auto const& v) {
			return v.size();
		});
	}

	void resize(size_t new_size) {
		with_vector_no_policy([&](auto& v) {
			v.resize(new_size);
		});
	}

	void push_back(const T& value) {
		with_vector_no_policy([&](auto& v) {
			v.push_back(value);
		});
	}

	void push_back(T&& value) {
		with_vector_no_policy([&](auto& v) {
			v.push_back(value);
		});
	}
	
	data_ref<T> operator[](size_t idx) {
		return data_ref(*this, idx);
	}

	void operator=(std::vector<T>& from) {
		if(l == DEVICE) {
			d = from;
		}

		#if USE_UNIVERSAL_VECTORS
		else if (l == UNIVERSAL) {
			u = from;
		}
		#endif

		else if (l == HOST) {
			h = from;
		}

		#if USE_MULTI_GPU
		else if (l == MULTI_GPU) {
			printf("operator=(std::vector<T>& from)->MULTI_GPU\n");
			assert(false);
			// mg = from;
		}
		#endif

		else {
			assert(false); // We cant assign to other devices directly
		}
	}

	template<typename L>
	void operator=(L& from) {
		with_vector_no_policy([&](auto& v) {
			v = from;
		});
	}

	void clear() {
		with_vector_no_policy([](auto& v){
			v.clear();
		});
	}

	void swap(data_manager<T>& other) {
		assert(l == other.l);
		assert(l == HOST 
#if USE_UNIVERSAL_VECTORS
			|| l == UNIVERSAL 
#endif
			|| l == DEVICE
#if USE_MULTI_GPU
			|| l == MULTI_GPU
#endif
		);
		// if (l == MULTI_GPU) {
		// 	printf("swap(data_manager<T>& other)->MULTI_GPU\n");
		// 	assert(false);
		// }
		
		with_vector_no_policy([&](auto& v) {
			using Vec = std::remove_reference_t<decltype(v)>;
			v.swap(other.vec<Vec>());
		});
	}

	void swap(thrust::device_vector<T>& other) {
		to_device();
		d.swap(other);
	}

	void swap(thrust::host_vector<T>& other) {
		to_host();
		h.swap(other);
	}

	#if USE_MULTI_GPU
	void swap(multi_gpu_vector<T>& other) {
		printf("swap(multi_gpu_vector<T>& other)->MULTI_GPU\n");
		assert(false);
		to_multi_gpu();
		mg.swap(other);
	}
	#endif

	// Third party thrust functions
	template<typename SWO> // Strict weak ordering
	void sort(SWO swo) {
		with_vector([&](auto policy, auto& v) {
			thrust::sort(
				policy, 
				v.begin(), v.end(),
				swo
			);
		});
	}

	template<typename List, typename SWO>
	void sort_by_key(List payload, SWO swo) {
		assert(payload.size() == size());
		with_vector([&](auto policy, auto& v, auto& payload_v) {
			thrust::sort_by_key(
				policy, 
				v.begin(), v.end(),
				payload_v.begin(),
				swo
			);
		}, payload);
	}

	template<typename SWO>
	void stable_sort(SWO swo) {
		with_vector([&](auto policy, auto& v) {
			thrust::stable_sort(
				policy, 
				v.begin(), v.end(),
				swo
			);
		});
	}

	template<typename List, typename SWO>
	void stable_sort_by_key(List payload, SWO swo) {
		assert(payload.size() == size());
		with_vector([&](auto policy, auto& v, auto& payload_v) {
			thrust::stable_sort_by_key(
				policy, 
				v.begin(), v.end(),
				payload_v.begin(),
				swo
			);
		}, payload);
	}

	template<typename SWO>
	bool is_sorted(SWO swo) {
		return with_vector([&](auto policy, auto& v){
			return thrust::is_sorted(
				policy, 
				v.begin(), v.end(), 
				swo
			);
		});
	}

	void unique() {
		with_vector([&](auto policy, auto& v) {
			auto it = thrust::unique(policy, v.begin(), v.end());
			resize(it - v.begin());
		});
	}

	operator thrust::device_vector<T>() const {
		thrust::device_vector<T> result;
		switch(l) {
		case HOST:
			result = h;
			break;

		#if USE_UNIVERSAL_VECTORS
		case UNIVERSAL:
			result = u;
			break;
		#endif

		case DEVICE:
			result = d;
			break;

		#if USE_MULTI_GPU
		case MULTI_GPU:
			printf("thrust::device_vector<T>()->MULTI_GPU\n");
			assert(false);
		#endif
		}
		return result;
	}

	operator thrust::universal_vector<T>() const {
		thrust::universal_vector<T> result;
		switch(l) {
		case HOST:
			result = h;
			break;

		#if USE_UNIVERSAL_VECTORS
		case UNIVERSAL:
			result = u;
			break;
		#endif

		case DEVICE:
			result = d;
			break;

		#if USE_MULTI_GPU
		case MULTI_GPU:
			printf("thrust::universal_vector<T>()->MULTI_GPU\n");
			assert(false);
		#endif
		}
		return result;
	}

	operator thrust::host_vector<T>() const {
		thrust::host_vector<T> result;
		switch(l) {
		case HOST:
			result = h;
			break;

		#if USE_UNIVERSAL_VECTORS
		case UNIVERSAL:
			result = u;
			break;
		#endif

		case DEVICE:
			result = d;
			break;

		#if USE_MULTI_GPU
		case MULTI_GPU:
			printf("thrust::host_vector<T>()->MULTI_GPU\n");
			assert(false);
			break;
		#endif
		}
		return result;
	}

	template <typename M>
	struct apply_map_func{
		M begin;
		
		apply_map_func(M _begin) : begin(_begin) {}

		__host__ __device__
		void operator() (T& in) {
			in.apply_map(begin);
		}
	};

	template <typename M>
	void apply_map(M map) {
		with_vector([&](auto policy, auto& v) {
			thrust::for_each(
				policy, 
				v.begin(), v.end(),
				apply_map_func(map)
			);
		});
	}

	template <typename F>
	uint32_t count_if(F filter) {
		return with_vector([&](auto policy, auto const& v) {
			return thrust::count_if(
				policy,
				v.begin(), v.end(),
				filter
			);
		});
	}
};


#endif
