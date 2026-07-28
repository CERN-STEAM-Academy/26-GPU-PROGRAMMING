// Node ref used in BDDs

#include "bdd.h"

using namespace sro;

void node_ref::check_and_propagate_redirect() {
	if (b.expired()) {
		throw std::runtime_error("Reference constructed with expired pointer.");
	}

	std::shared_ptr<bdd> locked_bdd = b.lock();
	if (locked_bdd->redirects.contains(root_id)) {
		b = locked_bdd->redirects[root_id];
	}
}

void node_ref::increment_reference() {
	if (b.expired()) return;
	b.lock()->increment_reference(root_id);
}

void node_ref::decrement_reference() {
	if (b.expired()) return;

	b.lock()->decrement_reference(root_id);
}

node_ref::node_ref(const node_ref& from, std::weak_ptr<bdd> _b) : b(_b) {
	if (b.expired()) {
		throw std::runtime_error("Reference constructed with expired pointer.");
	}

	root_id = b.lock()->add_root(from.get_id());
	increment_reference();
}

node_ref::node_ref(uint32_t _root_id, std::weak_ptr<bdd> _b) : b(_b) {
	if (b.expired()) {
		throw std::runtime_error("Reference constructed with expired pointer.");
	}
	root_id = _root_id;
	increment_reference();
}

node_ref::node_ref(const node_ref& from){
	root_id = from.root_id;
	b = from.b;
	if (b.expired()) {
		throw std::runtime_error("Reference copied from reference with expired pointer.");
	}
	increment_reference();
}

node_ref::node_ref(node_ref&& from){
	root_id = from.root_id;
	b = from.b;
	if (b.expired()) {
		throw std::runtime_error("Reference moved from reference with expired pointer.");
	}
	increment_reference();
}

node_ref::~node_ref() {
	decrement_reference();
}

node_ref& node_ref::operator=(const node_ref& from) {
	if (from.b.expired()) assert(false);
	if (b.expired()) assert(false);
	assert(from.b.lock() == b.lock());
	decrement_reference();

	root_id = from.root_id;
	b = from.b;
	
	increment_reference();

	return *this;
}

uint32_t node_ref::get_id() const {
	if (b.expired()) assert(false);
	std::shared_ptr<bdd> raw_b = b.lock();
	return raw_b->roots[root_id];
}

uint32_t node_ref::get_id() {
	if (b.expired()) assert(false);
	check_and_propagate_redirect();
	// printf("Get id\n");
	std::shared_ptr<bdd> raw_b = b.lock();
	return raw_b->roots[root_id];
}

uint64_t node_ref::get_metric(std::vector<uint64_t> root_metrics) {
	check_and_propagate_redirect();
	// printf("Get metric\n");
	return root_metrics[get_id()];
}

bdd_iterator node_ref::begin() {
	check_and_propagate_redirect();
	if (b.expired()) assert(false);
	std::shared_ptr<bdd> raw_b = b.lock();
	// printf("Begin\n");
	return raw_b->begin(*this);
}

bdd_iterator node_ref::end() {
	check_and_propagate_redirect();
	if (b.expired()) assert(false);
	std::shared_ptr<bdd> raw_b = b.lock();
	// printf("End\n");
	return raw_b->end(*this);
}

void bdd::increment_reference(uint32_t root_id) {
	root_refcounts[root_id]++;
}

void bdd::decrement_reference(uint32_t root_id) {
	assert(root_refcounts[root_id] > 0);
	root_refcounts[root_id]--;

	if (root_refcounts[root_id] == 0) {
		// The reference to this node is dead. We first delete the root
		uint32_t id = roots[root_id];
		roots.erase(root_id);
		root_refcounts.erase(root_id);

		// Then we check if the same node is also referenced by any other roots
		bool safe_to_remove = true;
		for (auto [key, value] : roots) {
			if (value == id) safe_to_remove = false;
		}
		if (safe_to_remove) remove_root_nodes({id});

		for (const auto& [key, value] : copied_bdds) {
			if (root_id < value) {
				key.lock()->decrement_reference(id);	
			}
		}
	}
}