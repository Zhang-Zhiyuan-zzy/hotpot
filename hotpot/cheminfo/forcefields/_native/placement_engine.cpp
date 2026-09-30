#include "placement_engine.hpp"

#include "placement_candidates.hpp"

#include "../../obWrappers/_native/native_engine.hpp"

#include <algorithm>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <utility>


namespace hotpot::forcefields {
namespace {


void append_status_warning(
    PlacementStatus status,
    std::vector<std::string>& warning_codes
) {
    if (status == PlacementStatus::PARTIAL) {
        warning_codes.push_back("metal_placement_partial");
    } else if (status == PlacementStatus::INFEASIBLE) {
        warning_codes.push_back("metal_placement_infeasible");
    }
}


void synchronize_status_warning(MetalPlacementResult& result) {
    result.warning_codes.erase(std::remove_if(
        result.warning_codes.begin(),
        result.warning_codes.end(),
        [](const std::string& warning) {
            return warning == "metal_placement_partial"
                || warning == "metal_placement_infeasible";
        }
    ), result.warning_codes.end());
    append_status_warning(result.status, result.warning_codes);
}


MetalPlacementResult finish_result(
    StructureSession& session,
    const StructureSnapshot& snapshot,
    const detail::PlacementEvaluationWorkspace& workspace,
    PlacementCandidateEvidence selected,
    std::vector<PlacementCandidateEvidence> evaluated,
    const MetalPlacementOptions& options
) {
    const auto metal = static_cast<std::size_t>(workspace.target.metal_index);
    const Coordinate original = snapshot.coordinates[metal];
    const bool usable = selected.status != PlacementStatus::INFEASIBLE;
    const Coordinate committed = usable ? selected.coordinates : original;
    if (usable && committed != original) {
        auto coordinates = snapshot.coordinates;
        coordinates[metal] = committed;
        update_structure_coordinates(session, coordinates);
    }
    std::vector<std::string> warnings;
    append_status_warning(selected.status, warnings);
    if (workspace.cycles->topology.excluded_large_cycle_count != 0) {
        warnings.push_back("metal_placement_large_cycles_excluded");
    }
    const std::size_t candidate_count = evaluated.size();
    if (!options.retain_candidate_evidence) {
        evaluated.clear();
    }
    return {
        workspace.target.metal_index,
        selected.status,
        original,
        committed,
        committed != original,
        candidate_count,
        std::move(selected),
        std::move(evaluated),
        workspace.cycles->topology.excluded_large_cycle_count,
        std::move(warnings),
    };
}


detail::PlacementEvaluationWorkspace make_workspace(
    const StructureSession& session,
    const StructureSnapshot& snapshot,
    std::int32_t metal_index,
    const MetalPlacementOptions& options
) {
    return detail::prepare_placement_evaluation(
        session.input(),
        snapshot.coordinates,
        snapshot.active_ligand_bond_mask,
        snapshot.component_ids,
        metal_index,
        options
    );
}


void require_unbound_metal(
    const ComplexSessionInput& input,
    const StructureSnapshot& snapshot,
    std::int32_t metal_index
) {
    for (
        std::size_t index = 0;
        index < input.intended_coordination_bonds.size();
        ++index
    ) {
        const auto& bond = input.intended_coordination_bonds[index];
        if (bond[0] == metal_index
            && snapshot.active_coordination_mask[index] != 0) {
            throw std::logic_error(
                "metal placement requires all intended coordination bonds "
                "for the target metal to be inactive"
            );
        }
    }
}


}  // namespace


PlacementCandidateEvidence assess_metal_position(
    const StructureSession& session,
    std::int32_t metal_index,
    const Coordinate& candidate,
    const MetalPlacementOptions& options
) {
    std::lock_guard<std::recursive_mutex> lock(
        hotpot::obwrappers::openbabel_runtime_mutex()
    );
    const auto snapshot = snapshot_structure(session);
    const auto workspace = make_workspace(
        session,
        snapshot,
        metal_index,
        options
    );
    return detail::evaluate_metal_position(
        workspace,
        candidate,
        PlacementProposalKind::CURRENT,
        0,
        options
    );
}


MetalPlacementResult place_metal(
    StructureSession& session,
    std::int32_t metal_index,
    const MetalPlacementOptions& options
) {
    std::lock_guard<std::recursive_mutex> lock(
        hotpot::obwrappers::openbabel_runtime_mutex()
    );
    const auto snapshot = snapshot_structure(session);
    require_unbound_metal(session.input(), snapshot, metal_index);
    const auto workspace = make_workspace(
        session,
        snapshot,
        metal_index,
        options
    );
    const auto metal = static_cast<std::size_t>(metal_index);
    auto current = detail::evaluate_metal_position(
        workspace,
        snapshot.coordinates[metal],
        PlacementProposalKind::CURRENT,
        0,
        options
    );
    if (current.status == PlacementStatus::FULLY_FEASIBLE) {
        return finish_result(
            session,
            snapshot,
            workspace,
            current,
            {current},
            options
        );
    }

    const auto proposals = detail::generate_metal_placement_candidates(
        workspace,
        options
    );
    std::vector<PlacementCandidateEvidence> evaluated;
    evaluated.reserve(proposals.size());
    evaluated.push_back(std::move(current));
    if (proposals.size() > 1) {
        const std::vector<detail::PlacementProposal> relocation_proposals(
            proposals.begin() + 1,
            proposals.end()
        );
        auto relocation_evidence = detail::evaluate_metal_positions(
            workspace,
            relocation_proposals,
            options
        );
        evaluated.insert(
            evaluated.end(),
            std::make_move_iterator(relocation_evidence.begin()),
            std::make_move_iterator(relocation_evidence.end())
        );
    }
    auto selected = evaluated.front();
    for (std::size_t index = 1; index < evaluated.size(); ++index) {
        if (detail::prefer_placement_candidate(evaluated[index], selected)) {
            selected = evaluated[index];
        }
    }
    return finish_result(
        session,
        snapshot,
        workspace,
        std::move(selected),
        std::move(evaluated),
        options
    );
}


MetalPlacementReport place_metals(
    StructureSession& session,
    const MetalPlacementOptions& options
) {
    std::lock_guard<std::recursive_mutex> lock(
        hotpot::obwrappers::openbabel_runtime_mutex()
    );
    MetalPlacementReport report;
    report.metals.reserve(session.input().metal_indices.size());
    for (const auto metal : session.input().metal_indices) {
        const bool has_donor = std::any_of(
            session.input().intended_coordination_bonds.begin(),
            session.input().intended_coordination_bonds.end(),
            [metal](const BondIndex& bond) { return bond[0] == metal; }
        );
        if (!has_donor) {
            continue;
        }
        auto result = place_metal(session, metal, options);
        report.metals.push_back(std::move(result));
    }
    const auto final_snapshot = snapshot_structure(session);
    for (auto& result : report.metals) {
        const auto workspace = make_workspace(
            session,
            final_snapshot,
            result.metal_index,
            options
        );
        const auto metal = static_cast<std::size_t>(result.metal_index);
        auto final_evidence = detail::evaluate_metal_position(
            workspace,
            final_snapshot.coordinates[metal],
            result.selected_evidence.proposal_kind,
            result.selected_evidence.proposal_ordinal,
            options
        );
        final_evidence.displacement_angstrom =
            result.selected_evidence.displacement_angstrom;
        if (final_evidence.status != result.status) {
            result.warning_codes.push_back(
                "metal_placement_post_batch_recheck_changed"
            );
        }
        result.status = final_evidence.status;
        result.selected_evidence = std::move(final_evidence);
        synchronize_status_warning(result);
        report.warning_codes.insert(
            report.warning_codes.end(),
            result.warning_codes.begin(),
            result.warning_codes.end()
        );
    }
    report.selected_coordinates = final_snapshot.coordinates;
    return report;
}


}  // namespace hotpot::forcefields
