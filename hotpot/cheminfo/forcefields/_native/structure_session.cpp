#include "structure_session.hpp"
#include "structure_session_internal.hpp"

#include "../../obWrappers/_native/native_engine.hpp"
#include "../../obWrappers/_native/openbabel_adapter.hpp"

#include <openbabel/bond.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <utility>


namespace hotpot::forcefields {


namespace {


std::vector<std::int32_t> derive_component_ids(
    const ComplexSessionInput& input
) {
    std::vector<std::vector<std::size_t>> adjacency(input.atom_count());
    for (const auto& endpoints : input.ligand_bond_indices) {
        const auto first = static_cast<std::size_t>(endpoints[0]);
        const auto second = static_cast<std::size_t>(endpoints[1]);
        adjacency[first].push_back(second);
        adjacency[second].push_back(first);
    }

    std::vector<std::int32_t> component_ids(input.atom_count(), -1);
    std::int32_t next_component = 0;
    for (std::size_t start = 0; start < input.atom_count(); ++start) {
        if (component_ids[start] != -1) {
            continue;
        }
        std::deque<std::size_t> pending{start};
        component_ids[start] = next_component;
        while (!pending.empty()) {
            const auto atom = pending.front();
            pending.pop_front();
            for (const auto neighbour : adjacency[atom]) {
                if (component_ids[neighbour] == -1) {
                    component_ids[neighbour] = next_component;
                    pending.push_back(neighbour);
                }
            }
        }
        ++next_component;
    }
    return component_ids;
}


int openbabel_ligand_bond_order(double order) {
    return static_cast<int>(order);
}


int openbabel_coordination_bond_order(double order) {
    return std::max(1, static_cast<int>(order));
}


OpenBabel::OBMol make_session_obmol(const ComplexSessionInput& input) {
    input.validate();
    return hotpot::obwrappers::make_obmol(ligand_molecule_data(input));
}


void validate_active_mask(
    const ComplexSessionInput& input,
    const std::vector<std::uint8_t>& active_mask
) {
    if (active_mask.size() != input.intended_coordination_bond_count()) {
        throw std::invalid_argument(
            "active coordination mask must match intended bonds"
        );
    }
    if (std::any_of(
            active_mask.begin(),
            active_mask.end(),
            [](std::uint8_t active) { return active > 1; }
        )) {
        throw std::invalid_argument(
            "active coordination mask values must be zero or one"
        );
    }
}


}  // namespace


StructureSession::StructureSession(
    const ComplexSessionInput& input,
    bool coordination_active
) :
    input_(input),
    obmol_(std::make_unique<OpenBabel::OBMol>(make_session_obmol(input_))),
    component_ids_(derive_component_ids(input_)),
    active_ligand_bond_mask_(input_.ligand_bond_count(), 1),
    active_coordination_mask_(input_.intended_coordination_bond_count(), 0) {
    if (coordination_active) {
        apply_coordination_active_mask(
            std::vector<std::uint8_t>(
                input_.intended_coordination_bond_count(),
                1
            ),
            false
        );
    }
}


StructureSession::~StructureSession() {
    std::lock_guard<std::recursive_mutex> lock(
        hotpot::obwrappers::openbabel_runtime_mutex()
    );
    obmol_.reset();
}


std::size_t StructureSession::atom_count() const noexcept {
    return input_.atom_count();
}


std::size_t StructureSession::ligand_bond_count() const noexcept {
    return input_.ligand_bond_count();
}


std::size_t StructureSession::intended_coordination_bond_count()
    const noexcept {
    return input_.intended_coordination_bond_count();
}


const ComplexSessionInput& StructureSession::input() const noexcept {
    return input_;
}


const std::vector<std::int32_t>& StructureSession::component_ids()
    const noexcept {
    return component_ids_;
}


const std::vector<std::uint8_t>&
StructureSession::active_ligand_bond_mask() const noexcept {
    return active_ligand_bond_mask_;
}


const std::vector<std::uint8_t>&
StructureSession::active_coordination_mask() const noexcept {
    return active_coordination_mask_;
}


void StructureSession::apply_ligand_bond_active_mask(
    const std::vector<std::uint8_t>& active_mask,
    bool update_revision
) {
    if (active_mask.size() != input_.ligand_bond_count()) {
        throw std::invalid_argument(
            "active ligand-bond mask must match ligand bonds"
        );
    }
    if (std::any_of(
            active_mask.begin(),
            active_mask.end(),
            [](std::uint8_t active) { return active > 1; }
        )) {
        throw std::invalid_argument(
            "active ligand-bond mask values must be zero or one"
        );
    }
    if (active_mask == active_ligand_bond_mask_) {
        return;
    }

    auto& obmol = StructureSessionAccess::obmol(*this);
    obmol.BeginModify();
    for (std::size_t index = 0; index < active_mask.size(); ++index) {
        if (active_mask[index] == active_ligand_bond_mask_[index]) {
            continue;
        }
        const auto endpoints = input_.ligand_bond_indices[index];
        if (active_mask[index] != 0) {
            if (!obmol.AddBond(
                    static_cast<unsigned long>(endpoints[0] + 1),
                    static_cast<unsigned long>(endpoints[1] + 1),
                    openbabel_ligand_bond_order(
                        input_.ligand_bond_orders[index]
                    )
                )) {
                throw std::runtime_error(
                    "Open Babel rejected a ligand bond"
                );
            }
        } else {
            auto* bond = obmol.GetBond(
                static_cast<int>(endpoints[0] + 1),
                static_cast<int>(endpoints[1] + 1)
            );
            if (bond == nullptr || !obmol.DeleteBond(bond)) {
                throw std::runtime_error(
                    "Open Babel could not deactivate a ligand bond"
                );
            }
        }
    }
    obmol.EndModify();
    for (std::size_t index = 0; index < active_mask.size(); ++index) {
        if (active_mask[index] == 0) {
            continue;
        }
        const auto endpoints = input_.ligand_bond_indices[index];
        auto* bond = obmol.GetBond(
            static_cast<int>(endpoints[0] + 1),
            static_cast<int>(endpoints[1] + 1)
        );
        bond->SetAromatic(
            input_.ligand_bond_kinds[index] == BondKind::AROMATIC
            || input_.ligand_bond_aromatic[index] != 0
        );
    }
    active_ligand_bond_mask_ = active_mask;
    if (update_revision) {
        ++topology_revision_;
    }
}


void StructureSession::apply_coordination_active_mask(
    const std::vector<std::uint8_t>& active_mask,
    bool update_revision
) {
    validate_active_mask(input_, active_mask);
    if (active_mask == active_coordination_mask_) {
        return;
    }

    auto& obmol = StructureSessionAccess::obmol(*this);
    obmol.BeginModify();
    for (std::size_t index = 0; index < active_mask.size(); ++index) {
        if (active_mask[index] == active_coordination_mask_[index]) {
            continue;
        }
        const auto endpoints = input_.intended_coordination_bonds[index];
        if (active_mask[index] != 0) {
            if (!obmol.AddBond(
                    static_cast<unsigned long>(endpoints[0] + 1),
                    static_cast<unsigned long>(endpoints[1] + 1),
                    openbabel_coordination_bond_order(
                        input_.intended_coordination_orders[index]
                    )
                )) {
                throw std::runtime_error(
                    "Open Babel rejected an intended coordination bond"
                );
            }
        } else {
            auto* bond = obmol.GetBond(
                static_cast<int>(endpoints[0] + 1),
                static_cast<int>(endpoints[1] + 1)
            );
            if (bond == nullptr || !obmol.DeleteBond(bond)) {
                throw std::runtime_error(
                    "Open Babel could not deactivate a coordination bond"
                );
            }
        }
    }
    obmol.EndModify();
    active_coordination_mask_ = active_mask;
    if (update_revision) {
        ++topology_revision_;
    }
}


std::unique_ptr<StructureSession> create_coordination_session(
    const ComplexSessionInput& input
) {
    std::lock_guard<std::recursive_mutex> lock(
        hotpot::obwrappers::openbabel_runtime_mutex()
    );
    return std::unique_ptr<StructureSession>(
        new StructureSession(input, false)
    );
}


std::unique_ptr<StructureSession> create_optimization_session(
    const ComplexSessionInput& input
) {
    std::lock_guard<std::recursive_mutex> lock(
        hotpot::obwrappers::openbabel_runtime_mutex()
    );
    return std::unique_ptr<StructureSession>(
        new StructureSession(input, true)
    );
}


StructureSnapshot snapshot_structure(const StructureSession& session) {
    std::lock_guard<std::recursive_mutex> lock(
        hotpot::obwrappers::openbabel_runtime_mutex()
    );
    const auto& obmol = StructureSessionAccess::obmol(session);
    return StructureSnapshot{
        hotpot::obwrappers::extract_coordinates(obmol),
        session.active_ligand_bond_mask_,
        session.active_coordination_mask_,
        session.component_ids_,
        session.input_.ligand_bond_count(),
        obmol.NumBonds(),
        session.coordinate_revision_,
        session.topology_revision_,
    };
}


void update_structure_coordinates(
    StructureSession& session,
    const std::vector<Coordinate>& coordinates
) {
    std::lock_guard<std::recursive_mutex> lock(
        hotpot::obwrappers::openbabel_runtime_mutex()
    );
    auto& obmol = StructureSessionAccess::obmol(session);
    const auto current = hotpot::obwrappers::extract_coordinates(
        obmol
    );
    if (current == coordinates) {
        return;
    }
    hotpot::obwrappers::set_coordinates(obmol, coordinates);
    ++session.coordinate_revision_;
}


void set_ligand_bond_active_mask(
    StructureSession& session,
    const std::vector<std::uint8_t>& active_mask
) {
    std::lock_guard<std::recursive_mutex> lock(
        hotpot::obwrappers::openbabel_runtime_mutex()
    );
    session.apply_ligand_bond_active_mask(active_mask, true);
}


void set_coordination_active_mask(
    StructureSession& session,
    const std::vector<std::uint8_t>& active_mask
) {
    std::lock_guard<std::recursive_mutex> lock(
        hotpot::obwrappers::openbabel_runtime_mutex()
    );
    session.apply_coordination_active_mask(active_mask, true);
}


}  // namespace hotpot::forcefields
