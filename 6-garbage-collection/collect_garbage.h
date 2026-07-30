/**
 * A node_product has two member variables:
 * uint32_t low; // <-- the index of the node pointed to by the low edge
 * uint32_t high; // <-- the index of the node pointed to by the high edge
 */

 /**
  * policy is the execution policy with which the thrust functions will be executed.
  * 
  * v is the vector of node_product objects of this layer. They define the connections to the layer below.
  * Types could be:
  * thrust::host_vector<node_product>
  * thrust::device_vector<node_product>
  * 
  * parent_v is the vector of node_products of the parent layer. They define the connections to the current layer.
  * Types could be:
  * thrust::host_vector<node_product>
  * thrust::device_vector<node_product>
  * 
  * In this parent layer, some nodes have already been removed. It is the job of this function to inspect the current 
  * layer to see which nodes are orphaned and can be removed.
  */
void garbage_collection(auto policy, auto& v, auto& parent_v) {
	// Templating logic to deduce type of vector (host/device)
	using Vec = std::remove_reference_t<decltype(v)>;

	// Templating logic to construct the right type of vector (host/device) with our own type (uint32_t)
	// StencilVec is thrust::device_vector<uint32_t> or thrust::host_vector<uint32_t>
	using StencilVec = rebind_vector_t<Vec, uint32_t>; // Why use uint32_t and not bool?

	
	// Generate list of referenced nodes
	// This initializes a thrust::(host|device)_vector called stencil with v.size() elements, all initialized to `1`
	StencilVec stencil(v.size(), 1);


	// TODO: implement garbage collection

	/*
	Example:
	Adding 1 to each element of stencil
	thrust::transform(
		policy, // This is either thrust::host or thrust::device
		stencil.begin(), stencil.end(), // Input list
		stencil.begin(),	// Output list. Can be same as input list, but must not overlap with offset.
		[]__host__ __device__(uint32_t in) { return in + 1; }
	);
	*/ 
};