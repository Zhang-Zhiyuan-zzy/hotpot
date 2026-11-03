#include "openbabel_radii.hpp"

#include <openbabel/elements.h>


namespace hotpot::reference {


double openbabel_covalent_radius(unsigned int atomic_number) {
    return OpenBabel::OBElements::GetCovalentRad(atomic_number);
}


}  // namespace hotpot::reference
