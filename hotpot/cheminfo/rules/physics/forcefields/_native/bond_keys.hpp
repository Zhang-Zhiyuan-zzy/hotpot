#pragma once

#include "../../../../structure/_native/forcefield_input.hpp"

#include <utility>


namespace hotpot::forcefields::internal {


inline BondIndex canonical_bond_key(BondIndex endpoints) noexcept {
    if (endpoints[1] < endpoints[0]) {
        std::swap(endpoints[0], endpoints[1]);
    }
    return endpoints;
}


}  // namespace hotpot::forcefields::internal
