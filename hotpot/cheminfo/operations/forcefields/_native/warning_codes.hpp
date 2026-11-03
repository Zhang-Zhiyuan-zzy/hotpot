#pragma once

#include <algorithm>
#include <string>
#include <vector>


namespace hotpot::forcefields::internal {


inline void append_unique(
    std::vector<std::string>& values,
    const std::string& value
) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}


}  // namespace hotpot::forcefields::internal
