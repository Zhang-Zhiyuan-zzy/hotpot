#include "radii.hpp"

#include "../../../../reference/_native/openbabel_radii.hpp"

#include <cmath>


namespace hotpot::forcefields {


AtomicRadius covalent_radius(std::int32_t atomic_number) {
    const double radius = hotpot::reference::openbabel_covalent_radius(
        static_cast<unsigned int>(atomic_number)
    );
    if (std::isfinite(radius) && radius > 0.0) {
        return {radius, RadiusSource::OPENBABEL_COVALENT};
    }
    return {0.77, RadiusSource::DEFAULT_COVALENT};
}


}  // namespace hotpot::forcefields
