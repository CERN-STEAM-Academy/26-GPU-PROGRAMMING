#ifndef SRO_SPILL_POLICY
#define SRO_SPILL_POLICY

#include "../layer/layer.h"
#include "../arc_layer/arc_layer.h"
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

namespace sro {

// Spilling is enabled only when GPUDECIDE_MEM_BUDGET_GB is set.
// No budget -> original (no-spill) behavior everywhere.
inline bool spilling_enabled() {
    static const bool enabled = std::getenv("GPUDECIDE_MEM_BUDGET_GB") != nullptr;
    return enabled;
}

// Per-operation spill decisions. When disabled, every method is a no-op.
class SpillPolicy {
    bool   _enabled;
    size_t _budget_bytes    = 0;
    size_t _spill_min_elems = 4096;
    size_t _resident_bytes  = 0;
    bool   _spilled_this_op = false;

public:
    SpillPolicy() {
        static const bool   en      = spilling_enabled();
        static const size_t budget  = en
            ? std::stoull(std::getenv("GPUDECIDE_MEM_BUDGET_GB")) * (size_t)1024*1024*1024
            : 0;
        static const size_t minelem = std::getenv("GPUDECIDE_SPILL_MIN_ELEMS")
            ? std::stoull(std::getenv("GPUDECIDE_SPILL_MIN_ELEMS")) : 4096;
        _enabled = en; _budget_bytes = budget; _spill_min_elems = minelem;
    }

    bool enabled() const { return _enabled; }

    // Downward sweep: record an arc layer's size in the resident set.
    void track_arc_layer(size_t arc_count) {
        _resident_bytes += arc_count * sizeof(sro::arc);
    }

    // Downward sweep: spill the dormant persistent layer if over budget.
    void maybe_spill_persistent(layer& l) {
        if (!_enabled) return;
        if (_resident_bytes > _budget_bytes &&
            l.get_data_location() == DEVICE && l.size() >= 5000) {
            l.set_data_location(STORAGE_GDS);
            _spilled_this_op = true;
        }
    }

    // Downward sweep: spill the most-recently-completed arc layer if over budget.
    void maybe_spill_arc(arc_layer& target) {
        if (!_enabled) return;
        size_t tgt = target.size();
        if (_resident_bytes > _budget_bytes && !target.is_spilled()
            && tgt > _spill_min_elems) {
            target.set_data_location(STORAGE_GDS);
            _resident_bytes -= tgt * sizeof(sro::arc);
            _spilled_this_op = true;
        }
    }

    // Upward sweep: re-spill the persistent layer after its last use,
    // only if this op spilled on the way down.
    void maybe_respill(layer& l) {
        if (!_enabled) return;
        if (_spilled_this_op &&
            l.get_data_location() == DEVICE && l.size() >= 5000) {
            l.set_data_location(STORAGE_GDS);
        }
    }

    //Downward sweep: where a freshly-touched persistent layer lives.
    //   spilling ON  : large layers -> DEVICE (so they can be spilled)
    //   spilling OFF : original behavior -> everything to HOST
    void place_persistent(layer& l) {
        l.set_data_location(l.size() < 5000 ? HOST : DEVICE); 
    }

    // Diagnostics: print resident/spilled arc state (gated on GPUDECIDE_SPILL_DEBUG)
    void report(std::vector<arc_layer>& arc_layers) {
        static const bool dbg = std::getenv("GPUDECIDE_SPILL_DEBUG") != nullptr;
        if (!dbg) return;

        size_t n_spilled = 0, resident_arcs = 0, spilled_arcs = 0;
        for (arc_layer& l : arc_layers) {
            if (l.is_spilled()) { n_spilled++; spilled_arcs += l.size(); }
            else                {              resident_arcs += l.size(); }
        }
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        std::cerr << "[peak] layers=" << arc_layers.size()
                  << " spilled=" << n_spilled
                  << " resident_arcs=" << resident_arcs
                  << " resident_MB=" << (resident_arcs * sizeof(sro::arc) / (1024*1024))
                  << " spilled_MB="  << (spilled_arcs  * sizeof(sro::arc) / (1024*1024))
                  << " used_MB=" << ((tb - fb) / (1024*1024)) << "\n";
    }
};

} // namespace sro
#endif