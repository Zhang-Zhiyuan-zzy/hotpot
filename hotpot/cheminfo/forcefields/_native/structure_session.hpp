#pragma once

#include "contracts.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>


namespace OpenBabel {


class OBMol;


}  // namespace OpenBabel


namespace hotpot::forcefields {


class StructureSessionAccess;


class StructureSession final {
public:
    StructureSession(const StructureSession&) = delete;
    StructureSession& operator=(const StructureSession&) = delete;
    StructureSession(StructureSession&&) = delete;
    StructureSession& operator=(StructureSession&&) = delete;
    ~StructureSession();

    std::size_t atom_count() const noexcept;
    std::size_t ligand_bond_count() const noexcept;
    std::size_t intended_coordination_bond_count() const noexcept;
    const ComplexSessionInput& input() const noexcept;
    const std::vector<std::int32_t>& component_ids() const noexcept;
    const std::vector<std::uint8_t>& active_ligand_bond_mask() const noexcept;
    const std::vector<std::uint8_t>& active_coordination_mask() const noexcept;

private:
    friend class StructureSessionAccess;
    friend std::unique_ptr<StructureSession> create_coordination_session(
        const ComplexSessionInput& input
    );
    friend std::unique_ptr<StructureSession> create_optimization_session(
        const ComplexSessionInput& input
    );
    friend StructureSnapshot snapshot_structure(
        const StructureSession& session
    );
    friend void update_structure_coordinates(
        StructureSession& session,
        const std::vector<Coordinate>& coordinates
    );
    friend void set_coordination_active_mask(
        StructureSession& session,
        const std::vector<std::uint8_t>& active_mask
    );
    friend void set_ligand_bond_active_mask(
        StructureSession& session,
        const std::vector<std::uint8_t>& active_mask
    );

    StructureSession(const ComplexSessionInput& input, bool coordination_active);
    void apply_ligand_bond_active_mask(
        const std::vector<std::uint8_t>& active_mask,
        bool update_revision
    );
    void apply_coordination_active_mask(
        const std::vector<std::uint8_t>& active_mask,
        bool update_revision
    );

    ComplexSessionInput input_;
    std::unique_ptr<OpenBabel::OBMol> obmol_;
    std::vector<std::int32_t> component_ids_;
    std::vector<std::uint8_t> active_ligand_bond_mask_;
    std::vector<std::uint8_t> active_coordination_mask_;
    std::uint64_t coordinate_revision_ = 0;
    std::uint64_t topology_revision_ = 0;
};


std::unique_ptr<StructureSession> create_coordination_session(
    const ComplexSessionInput& input
);


std::unique_ptr<StructureSession> create_optimization_session(
    const ComplexSessionInput& input
);


StructureSnapshot snapshot_structure(const StructureSession& session);


void update_structure_coordinates(
    StructureSession& session,
    const std::vector<Coordinate>& coordinates
);


void set_ligand_bond_active_mask(
    StructureSession& session,
    const std::vector<std::uint8_t>& active_mask
);


void set_coordination_active_mask(
    StructureSession& session,
    const std::vector<std::uint8_t>& active_mask
);


}  // namespace hotpot::forcefields
