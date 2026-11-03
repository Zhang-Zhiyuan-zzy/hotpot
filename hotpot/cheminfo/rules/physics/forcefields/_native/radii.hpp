#pragma once

#include <cstdint>


namespace hotpot::forcefields {


enum class RadiusSource : std::uint8_t {
    OPENBABEL_COVALENT = 0,
    DEFAULT_COVALENT = 1,
};


struct AtomicRadius {
    double angstrom;
    RadiusSource source;
};


AtomicRadius covalent_radius(std::int32_t atomic_number);


}  // namespace hotpot::forcefields
