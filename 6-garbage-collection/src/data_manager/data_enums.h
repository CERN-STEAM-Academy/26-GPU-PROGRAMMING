#ifndef DATA_ENUMS
#define DATA_ENUMS

// Enum for data location
enum data_location{
	NONE,
	HOST,

#if USE_UNIVERSAL_VECTORS
	UNIVERSAL,
#endif

	DEVICE,
	STORAGE,
	STORAGE_GDS,

#if USE_MULTI_GPU
	MULTI_GPU
#endif
};

enum data_policy{
	SINGLE_CORE_CPU,
	MULTI_CORE_CPU,
	THRUST_GPU
};

#endif
