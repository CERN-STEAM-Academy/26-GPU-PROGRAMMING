// Iterator to range over satisfying assignments of BDD

#include "bdd.h"

using namespace sro;

bdd_iterator::bdd_iterator(bdd* _b, node_ref _root, std::vector<bool> _assignments) : root(_root) {
	b = _b;
	assignments = _assignments;
}

std::vector<bool> bdd_iterator::operator*() const {
	return assignments;
}

bdd_iterator& bdd_iterator::operator++() {
	std::vector<uint32_t> path;
	path.push_back(root.get_id());

	while (path.size() < b->layers.size()) {
		uint16_t l = path.size() - 1;
		path.push_back(b->layers[l].get_child(path.back(), assignments[l]));
	}

	// Remove tail of path until we can build a new tail
	while (
		assignments.back() || // We can't update a variable that's already true
		(b->lowermost_occurrence_of_false <= assignments.size() && b->layers[assignments.size() - 1].get_child(path.back(), true) == 0)
	) {
		path.pop_back();
		assignments.pop_back();

		if (assignments.size() == 0) {
			return *this;
		}
	}

	// Replace low assignment with high assignment
	assert(!assignments.back());
	assignments.back() = true;

	// Fill missing tail back up
	while (path.size() < b->layers.size()) {
		uint16_t l = path.size() - 1;
		path.push_back(b->layers[l].get_child(path.back(), assignments[l]));
		bool assignment = b->lowermost_occurrence_of_false < path.size() && b->layers[l + 1].get_child(path.back(), false) == 0;
		assignments.push_back(assignment);
	}

	return *this;
}

void bdd_iterator::operator++(int) {++*this;}

bool bdd_iterator::operator==(const bdd_iterator& comp) const {
	if (assignments.size() != comp.assignments.size()) return false;

	for (int i = 0; i < assignments.size(); i++) {
		if (assignments[i] != comp.assignments[i]) return false;
	}

	return true;
}

bdd_iterator bdd::begin(node_ref ref) {
	if (!false_cache_up_to_date) update_occurrences_of_false();

	uint32_t root = ref.get_id();

	// Check if root is false
	if (root == 0 && lowermost_occurrence_of_false == 0) return end(ref);

	std::vector<bool> assignments;
	uint32_t working_node = root;
	for (uint16_t l = 0; l < layers.size(); l++) {
		bool v = false;
		if (l + 1 >= lowermost_occurrence_of_false && layers[l].get_child(working_node, 0) == 0) v = true;

		working_node = layers[l].get_child(working_node, v);
		assignments.push_back(v);
	}
	
	return bdd_iterator(this, ref, assignments);
}

bdd_iterator bdd::end(node_ref ref) {
	if (!false_cache_up_to_date) update_occurrences_of_false();
	return bdd_iterator(this, ref, std::vector<bool>());
}