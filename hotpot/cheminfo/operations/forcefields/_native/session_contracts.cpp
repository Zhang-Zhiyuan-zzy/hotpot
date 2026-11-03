#include "session_contracts.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace hotpot::forcefields {


std::size_t PerturbationOffsetBatch::frame_count() const noexcept {
    return frames.size();
}


void PerturbationOffsetBatch::validate() const {
    for (const auto& frame : frames) {
        if (frame.size() != atom_count) {
            throw std::invalid_argument(
                "each perturbation frame must match atom_count"
            );
        }
        for (const auto& offset : frame) {
            if (!std::all_of(
                    offset.begin(),
                    offset.end(),
                    [](double value) { return std::isfinite(value); }
                )) {
                throw std::invalid_argument(
                    "perturbation offsets must be finite"
                );
            }
        }
    }
}


}  // namespace hotpot::forcefields
