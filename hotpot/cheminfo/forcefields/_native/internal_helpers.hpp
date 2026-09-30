#pragma once

#include "contracts.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>


namespace hotpot::forcefields::internal {


inline BondIndex canonical_bond_key(BondIndex endpoints) noexcept {
    if (endpoints[1] < endpoints[0]) {
        std::swap(endpoints[0], endpoints[1]);
    }
    return endpoints;
}


inline void append_unique(
    std::vector<std::string>& values,
    const std::string& value
) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}


}  // namespace hotpot::forcefields::internal
