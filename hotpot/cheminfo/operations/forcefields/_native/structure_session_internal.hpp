#pragma once

#include "structure_session.hpp"

#include <openbabel/mol.h>


namespace hotpot::forcefields {


class StructureSessionAccess final {
public:
    static OpenBabel::OBMol& obmol(StructureSession& session) noexcept {
        return *session.obmol_;
    }

    static const OpenBabel::OBMol& obmol(
        const StructureSession& session
    ) noexcept {
        return *session.obmol_;
    }

    static void record_coordinate_change(
        StructureSession& session
    ) noexcept {
        ++session.coordinate_revision_;
    }
};


}  // namespace hotpot::forcefields
