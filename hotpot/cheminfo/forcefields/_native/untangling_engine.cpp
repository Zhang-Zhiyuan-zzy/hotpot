#include "untangling_engine.hpp"

#include "internal_helpers.hpp"

#include "../../geometry/_native/prepared_cycle.hpp"
#include "../../geometry/_native/primitives.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>


namespace hotpot::forcefields {
namespace {


using internal::append_unique;
using internal::canonical_bond_key;


bool declared_metal(
    const ComplexSessionInput& input,
    std::size_t atom_index
) {
    return std::find(
        input.metal_indices.begin(),
        input.metal_indices.end(),
        static_cast<std::int32_t>(atom_index)
    ) != input.metal_indices.end();
}


bool active_bond(
    const ComplexSessionInput& input,
    const StructureSnapshot& snapshot,
    const BondIndex& key
) {
    for (std::size_t index = 0; index < input.ligand_bond_count(); ++index) {
        if (
            snapshot.active_ligand_bond_mask[index] != 0
            && canonical_bond_key(input.ligand_bond_indices[index]) == key
        ) {
            return true;
        }
    }
    for (
        std::size_t index = 0;
        index < input.intended_coordination_bond_count();
        ++index
    ) {
        if (
            snapshot.active_coordination_mask[index] != 0
            && canonical_bond_key(input.intended_coordination_bonds[index])
                == key
        ) {
            return true;
        }
    }
    return false;
}


struct WatchedPiercing {
    std::vector<std::size_t> ring_atom_indices;
    BondIndex target_bond_key;
    std::vector<BondIndex> opening_bond_keys;
};


struct WatchObservation {
    bool valid = true;
    hotpot::geometry::PiercingState state =
        hotpot::geometry::PiercingState::DOES_NOT_PIERCE;
    std::vector<WatchedPiercing> confirmed;
    std::size_t cycle_preparation_count = 0;
};


std::vector<BondIndex> eligible_opening_bonds(
    const ComplexSessionInput& input,
    const StructureSnapshot& snapshot,
    const detail::PreparedRingWorkspace& workspace,
    std::size_t ring_index
) {
    const auto& ring_atoms = workspace.topology.atom_indices[ring_index];
    const bool contains_metal = std::any_of(
        ring_atoms.begin(),
        ring_atoms.end(),
        [&input](std::size_t atom) { return declared_metal(input, atom); }
    );
    std::vector<BondIndex> eligible;
    for (const BondIndex& edge : workspace.topology.edge_keys[ring_index]) {
        const BondIndex key = canonical_bond_key(edge);
        if (!active_bond(input, snapshot, key)) {
            continue;
        }
        if (contains_metal) {
            const bool first_metal = declared_metal(
                input, static_cast<std::size_t>(key[0])
            );
            const bool second_metal = declared_metal(
                input, static_cast<std::size_t>(key[1])
            );
            if (first_metal == second_metal) {
                continue;
            }
            for (
                std::size_t index = 0;
                index < input.intended_coordination_bond_count();
                ++index
            ) {
                if (
                    snapshot.active_coordination_mask[index] != 0
                    && canonical_bond_key(
                        input.intended_coordination_bonds[index]
                    ) == key
                ) {
                    eligible.push_back(key);
                    break;
                }
            }
            continue;
        }
        const auto membership = workspace.topology.edge_memberships.find(key);
        if (
            membership == workspace.topology.edge_memberships.end()
            || membership->second != 1
        ) {
            continue;
        }
        for (std::size_t index = 0; index < input.ligand_bond_count(); ++index) {
            if (
                snapshot.active_ligand_bond_mask[index] != 0
                && input.ligand_bond_orders[index] == 1.0
                && canonical_bond_key(input.ligand_bond_indices[index]) == key
            ) {
                eligible.push_back(key);
                break;
            }
        }
    }
    std::sort(eligible.begin(), eligible.end());
    eligible.erase(std::unique(eligible.begin(), eligible.end()), eligible.end());
    return eligible;
}


std::vector<WatchedPiercing> make_watch(
    const ComplexSessionInput& input,
    const StructureSnapshot& snapshot,
    const detail::PreparedRingWorkspace& workspace,
    const detail::BondRingCheckpoint& checkpoint
) {
    std::vector<WatchedPiercing> watch;
    std::set<std::pair<std::vector<std::size_t>, BondIndex>> seen;
    for (const auto& finding : checkpoint.actionable_findings) {
        if (finding.relation.state != hotpot::geometry::PiercingState::PIERCES) {
            continue;
        }
        const auto key = std::make_pair(
            finding.ring_atom_indices,
            canonical_bond_key(finding.bond_key)
        );
        if (!seen.insert(key).second) {
            continue;
        }
        watch.push_back({
            finding.ring_atom_indices,
            key.second,
            eligible_opening_bonds(
                input, snapshot, workspace, finding.ring_index
            ),
        });
    }
    return watch;
}


hotpot::geometry::Segment3 segment(
    const StructureSnapshot& snapshot,
    const BondIndex& key
) {
    return {
        snapshot.coordinates[static_cast<std::size_t>(key[0])],
        snapshot.coordinates[static_cast<std::size_t>(key[1])],
    };
}


std::optional<BondIndex> select_opening_bond(
    const StructureSnapshot& snapshot,
    const std::vector<WatchedPiercing>& confirmed,
    const hotpot::geometry::NumericTolerances& tolerances
) {
    for (const auto& piercing : confirmed) {
        if (piercing.opening_bond_keys.empty()) {
            continue;
        }
        const auto target = segment(snapshot, piercing.target_bond_key);
        return *std::min_element(
            piercing.opening_bond_keys.begin(),
            piercing.opening_bond_keys.end(),
            [&snapshot, &target, &tolerances](
                const BondIndex& first,
                const BondIndex& second
            ) {
                const double first_distance =
                    hotpot::geometry::segment_segment_distance(
                        segment(snapshot, first), target, tolerances
                    );
                const double second_distance =
                    hotpot::geometry::segment_segment_distance(
                        segment(snapshot, second), target, tolerances
                    );
                return first_distance != second_distance
                    ? first_distance < second_distance
                    : first < second;
            }
        );
    }
    return std::nullopt;
}


bool ring_edges_are_active(
    const ComplexSessionInput& input,
    const StructureSnapshot& snapshot,
    const std::vector<std::size_t>& ring_atoms
) {
    for (std::size_t index = 0; index < ring_atoms.size(); ++index) {
        if (!active_bond(input, snapshot, canonical_bond_key({
                static_cast<std::int32_t>(ring_atoms[index]),
                static_cast<std::int32_t>(
                    ring_atoms[(index + 1) % ring_atoms.size()]
                ),
            }))) {
            return false;
        }
    }
    return true;
}


WatchObservation scan_watch(
    const ComplexSessionInput& input,
    const StructureSnapshot& snapshot,
    const std::vector<WatchedPiercing>& watch,
    const RingUntanglingOptions& options
) {
    WatchObservation observation;
    std::map<std::vector<std::size_t>, std::size_t> cycle_positions;
    std::vector<hotpot::geometry::PreparedCycle> prepared_cycles;
    for (const auto& piercing : watch) {
        if (cycle_positions.find(piercing.ring_atom_indices)
            != cycle_positions.end()) {
            continue;
        }
        if (
            piercing.ring_atom_indices.size() < 3
            || !ring_edges_are_active(
                input, snapshot, piercing.ring_atom_indices
            )
        ) {
            observation.valid = false;
            return observation;
        }
        std::vector<hotpot::geometry::Point3> ring_coordinates;
        ring_coordinates.reserve(piercing.ring_atom_indices.size());
        for (const std::size_t atom : piercing.ring_atom_indices) {
            if (atom >= snapshot.coordinates.size()) {
                observation.valid = false;
                return observation;
            }
            ring_coordinates.push_back(snapshot.coordinates[atom]);
        }
        cycle_positions.emplace(
            piercing.ring_atom_indices,
            prepared_cycles.size()
        );
        prepared_cycles.push_back(hotpot::geometry::prepare_cycle(
            hotpot::geometry::ArrayView<hotpot::geometry::Point3>(
                ring_coordinates
            ),
            options.ring_workspace.geometry_tolerances,
            options.ring_workspace.surface_limits
        ));
        ++observation.cycle_preparation_count;
    }
    for (const auto& piercing : watch) {
        if (!active_bond(input, snapshot, piercing.target_bond_key)) {
            observation.valid = false;
            return observation;
        }
        const auto& cycle = prepared_cycles[
            cycle_positions.at(piercing.ring_atom_indices)
        ];
        const auto screening = hotpot::geometry::screen_segment_cycle(
            segment(snapshot, piercing.target_bond_key), cycle, false
        );
        if (screening.state == hotpot::geometry::PiercingState::PIERCES) {
            observation.state = hotpot::geometry::PiercingState::PIERCES;
            observation.confirmed.push_back(piercing);
        } else if (
            screening.state == hotpot::geometry::PiercingState::UNDETERMINED
            && observation.state
                == hotpot::geometry::PiercingState::DOES_NOT_PIERCE
        ) {
            observation.state = hotpot::geometry::PiercingState::UNDETERMINED;
        }
    }
    return observation;
}


void apply_offsets(
    StructureSession& session,
    const std::vector<Coordinate>& offsets
) {
    auto snapshot = snapshot_structure(session);
    for (std::size_t atom = 0; atom < snapshot.coordinates.size(); ++atom) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            snapshot.coordinates[atom][axis] += offsets[atom][axis];
        }
    }
    update_structure_coordinates(session, snapshot.coordinates);
}


bool finite_coordinates(const std::vector<Coordinate>& coordinates) {
    return std::all_of(
        coordinates.begin(),
        coordinates.end(),
        [](const Coordinate& coordinate) {
            return std::all_of(
                coordinate.begin(),
                coordinate.end(),
                [](double value) { return std::isfinite(value); }
            );
        }
    );
}


std::optional<std::size_t> ligand_bond_index(
    const ComplexSessionInput& input,
    const BondIndex& key
) {
    for (std::size_t index = 0; index < input.ligand_bond_count(); ++index) {
        if (canonical_bond_key(input.ligand_bond_indices[index]) == key) {
            return index;
        }
    }
    return std::nullopt;
}


std::optional<std::size_t> coordination_bond_index(
    const ComplexSessionInput& input,
    const BondIndex& key
) {
    for (
        std::size_t index = 0;
        index < input.intended_coordination_bond_count();
        ++index
    ) {
        if (canonical_bond_key(input.intended_coordination_bonds[index]) == key) {
            return index;
        }
    }
    return std::nullopt;
}


class OpenBondGuard final {
public:
    OpenBondGuard(StructureSession& session, BondIndex bond_key) :
        session_(session),
        ligand_mask_(session.active_ligand_bond_mask()),
        coordination_mask_(session.active_coordination_mask()) {
        if (const auto index = ligand_bond_index(session.input(), bond_key)) {
            auto mask = ligand_mask_;
            mask[*index] = 0;
            set_ligand_bond_active_mask(session_, mask);
            kind_ = Kind::LIGAND;
        } else if (const auto index = coordination_bond_index(
                session.input(), bond_key
            )) {
            auto mask = coordination_mask_;
            mask[*index] = 0;
            set_coordination_active_mask(session_, mask);
            kind_ = Kind::COORDINATION;
        } else {
            throw std::invalid_argument(
                "ring-opening bond is absent from the session topology"
            );
        }
    }

    OpenBondGuard(const OpenBondGuard&) = delete;
    OpenBondGuard& operator=(const OpenBondGuard&) = delete;

    ~OpenBondGuard() {
        if (open_) {
            try {
                restore();
            } catch (...) {
                std::terminate();
            }
        }
    }

    void restore() {
        if (!open_) {
            return;
        }
        if (kind_ == Kind::LIGAND) {
            set_ligand_bond_active_mask(session_, ligand_mask_);
        } else {
            set_coordination_active_mask(session_, coordination_mask_);
        }
        open_ = false;
    }

private:
    enum class Kind : std::uint8_t {
        LIGAND,
        COORDINATION,
    };

    StructureSession& session_;
    std::vector<std::uint8_t> ligand_mask_;
    std::vector<std::uint8_t> coordination_mask_;
    Kind kind_ = Kind::LIGAND;
    bool open_ = true;
};


void append_step(
    RingUntanglingResult& result,
    StructureSession& session,
    RingUntanglingEvent event,
    std::size_t attempt,
    std::optional<BondIndex> opening_bond_key = std::nullopt,
    std::optional<double> energy_kj_mol = std::nullopt,
    std::optional<hotpot::geometry::PiercingState> state = std::nullopt,
    std::optional<std::size_t> piercing_count = std::nullopt,
    std::optional<detail::BondRingCheckpoint> checkpoint_evidence =
        std::nullopt
) {
    result.steps.push_back({
        event,
        attempt,
        snapshot_structure(session),
        std::move(opening_bond_key),
        energy_kj_mol,
        state,
        piercing_count,
        std::move(checkpoint_evidence),
    });
}


detail::BondRingCheckpoint full_checkpoint(
    StructureSession& session,
    const RingUntanglingOptions& options,
    detail::RingWorkspaceCache& cache
) {
    const auto snapshot = snapshot_structure(session);
    return detail::scan_bond_ring_checkpoint(
        cache.prepare(session.input(), snapshot, options.ring_workspace),
        options.ring_workspace
    );
}


}  // namespace


void RingUntanglingOptions::validate() const {
    if (forcefield.empty()) {
        throw std::invalid_argument("forcefield must not be empty");
    }
    if (short_optimization_steps == 0) {
        throw std::invalid_argument(
            "short_optimization_steps must be positive"
        );
    }
    if (
        !std::isfinite(torsion_singularity_threshold)
        || torsion_singularity_threshold < 0.0
    ) {
        throw std::invalid_argument(
            "torsion_singularity_threshold must be finite and non-negative"
        );
    }
    if (
        !std::isfinite(torsion_repair_angle_radians)
        || torsion_repair_angle_radians <= 0.0
    ) {
        throw std::invalid_argument(
            "torsion_repair_angle_radians must be finite and positive"
        );
    }
    ring_workspace.validate();
}


std::size_t RingUntanglingAttemptCursor::remaining() const noexcept {
    return attempt_limit - attempts_completed;
}


bool RingUntanglingAttemptCursor::exhausted() const noexcept {
    return attempts_completed == attempt_limit;
}


void RingUntanglingAttemptCursor::validate() const {
    if (attempts_completed > attempt_limit) {
        throw std::invalid_argument(
            "untangling attempts_completed exceeds attempt_limit"
        );
    }
}


RingUntanglingResult untangle_ring_piercings(
    StructureSession& session,
    const RingUntanglingOptions& options,
    const PerturbationOffsetBatch& perturbation_offsets,
    RingUntanglingAttemptCursor& attempt_cursor,
    const detail::BondRingCheckpoint& entry_checkpoint
) {
    options.validate();
    attempt_cursor.validate();
    perturbation_offsets.validate();
    if (perturbation_offsets.atom_count != session.atom_count()) {
        throw std::invalid_argument(
            "untangling perturbation atom_count must match the session"
        );
    }
    if (perturbation_offsets.frame_count() != attempt_cursor.attempt_limit) {
        throw std::invalid_argument(
            "untangling perturbation count must match the shared attempt limit"
        );
    }
    if (entry_checkpoint.scope != options.ring_workspace.scope) {
        throw std::invalid_argument(
            "entry checkpoint ring scope must match untangling options"
        );
    }

    RingUntanglingResult result;
    result.initial_piercing_count = entry_checkpoint.piercing_pair_count;
    result.minimum_piercing_count = entry_checkpoint.piercing_pair_count;
    result.final_checkpoint = entry_checkpoint;
    append_step(
        result,
        session,
        RingUntanglingEvent::TOPOLOGY_CHECKPOINT,
        attempt_cursor.attempts_completed,
        std::nullopt,
        std::nullopt,
        entry_checkpoint.state,
        entry_checkpoint.piercing_pair_count,
        entry_checkpoint
    );
    if (entry_checkpoint.piercing_pair_count == 0) {
        result.resolved = true;
        result.final_piercing_count = 0;
        if (entry_checkpoint.state
            == hotpot::geometry::PiercingState::UNDETERMINED) {
            append_unique(result.warning_codes, "ring_relation_undetermined");
        }
        return result;
    }

    detail::RingWorkspaceCache cache;
    const auto scan_full = [&session, &options, &cache, &result]() {
        ++result.full_checkpoint_count;
        return full_checkpoint(session, options, cache);
    };
    auto current_snapshot = snapshot_structure(session);
    const auto& initial_workspace = cache.prepare(
        session.input(), current_snapshot, options.ring_workspace
    );
    auto watch = make_watch(
        session.input(), current_snapshot, initial_workspace, entry_checkpoint
    );
    auto current_confirmed = watch;
    auto global_best_coordinates = current_snapshot.coordinates;
    std::size_t global_minimum = entry_checkpoint.piercing_pair_count;
    auto watch_best_coordinates = current_snapshot.coordinates;
    std::size_t watch_minimum = watch.size();
    bool stopped_without_edge = watch.empty();

    while (!current_confirmed.empty() && !attempt_cursor.exhausted()) {
        current_snapshot = snapshot_structure(session);
        const auto opening_bond = select_opening_bond(
            current_snapshot,
            current_confirmed,
            options.ring_workspace.geometry_tolerances
        );
        if (!opening_bond.has_value()) {
            stopped_without_edge = true;
            break;
        }

        const std::size_t attempt = attempt_cursor.attempts_completed + 1;
        OpenBondGuard open_bond(session, *opening_bond);
        append_step(
            result, session, RingUntanglingEvent::RING_OPENED,
            attempt, opening_bond
        );
        try {
            apply_offsets(
                session,
                perturbation_offsets.frames[attempt_cursor.attempts_completed]
            );
            append_step(
                result, session, RingUntanglingEvent::PERTURBED,
                attempt, opening_bond
            );
            const auto optimized = single_optimize_session(
                session,
                options.forcefield,
                options.short_optimization_steps,
                options.torsion_singularity_threshold,
                options.torsion_repair_angle_radians
            );
            append_step(
                result,
                session,
                RingUntanglingEvent::OPEN_TOPOLOGY_OPTIMIZED,
                attempt,
                opening_bond,
                std::isfinite(optimized.energy_kj_mol)
                    ? std::optional<double>(optimized.energy_kj_mol)
                    : std::nullopt
            );
        } catch (...) {
            open_bond.restore();
            throw;
        }
        open_bond.restore();
        if (!finite_coordinates(snapshot_structure(session).coordinates)) {
            update_structure_coordinates(session, current_snapshot.coordinates);
            append_unique(
                result.warning_codes, "ring_untangling_nonfinite_frame"
            );
        }
        ++attempt_cursor.attempts_completed;
        ++result.attempts_used;
        append_step(
            result, session, RingUntanglingEvent::RING_CLOSED,
            attempt, opening_bond
        );

        current_snapshot = snapshot_structure(session);
        const auto observation = scan_watch(
            session.input(), current_snapshot, watch, options
        );
        result.watch_cycle_preparation_count +=
            observation.cycle_preparation_count;
        if (
            observation.valid
            && observation.state
                == hotpot::geometry::PiercingState::PIERCES
        ) {
            current_confirmed = observation.confirmed;
            if (current_confirmed.size() <= watch_minimum) {
                watch_minimum = current_confirmed.size();
                watch_best_coordinates = current_snapshot.coordinates;
            }
            continue;
        }

        auto checkpoint = scan_full();
        append_step(
            result,
            session,
            RingUntanglingEvent::TOPOLOGY_CHECKPOINT,
            attempt_cursor.attempts_completed,
            std::nullopt,
            std::nullopt,
            checkpoint.state,
            checkpoint.piercing_pair_count,
            checkpoint
        );
        if (checkpoint.piercing_pair_count <= global_minimum) {
            global_minimum = checkpoint.piercing_pair_count;
            global_best_coordinates = snapshot_structure(session).coordinates;
        }
        result.minimum_piercing_count = std::min(
            result.minimum_piercing_count, checkpoint.piercing_pair_count
        );
        result.final_checkpoint = checkpoint;
        if (checkpoint.piercing_pair_count == 0) {
            if (checkpoint.state
                == hotpot::geometry::PiercingState::UNDETERMINED) {
                append_unique(
                    result.warning_codes, "ring_relation_undetermined"
                );
            }
            result.resolved = true;
            result.final_piercing_count = 0;
            return result;
        }
        const auto snapshot = snapshot_structure(session);
        const auto& workspace = cache.prepare(
            session.input(), snapshot, options.ring_workspace
        );
        watch = make_watch(
            session.input(), snapshot, workspace, checkpoint
        );
        current_confirmed = watch;
        watch_best_coordinates = snapshot.coordinates;
        watch_minimum = watch.size();
        if (watch.empty()) {
            stopped_without_edge = true;
            break;
        }
    }

    if (result.final_checkpoint.piercing_pair_count != 0) {
        auto checkpoint = entry_checkpoint;
        if (result.attempts_used != 0) {
            update_structure_coordinates(session, watch_best_coordinates);
            append_step(
                result,
                session,
                RingUntanglingEvent::ROLLED_BACK,
                attempt_cursor.attempts_completed
            );
            checkpoint = scan_full();
            append_step(
                result,
                session,
                RingUntanglingEvent::TOPOLOGY_CHECKPOINT,
                attempt_cursor.attempts_completed,
                std::nullopt,
                std::nullopt,
                checkpoint.state,
                checkpoint.piercing_pair_count,
                checkpoint
            );
            if (checkpoint.piercing_pair_count <= global_minimum) {
                global_minimum = checkpoint.piercing_pair_count;
                global_best_coordinates = snapshot_structure(
                    session
                ).coordinates;
            } else {
                update_structure_coordinates(
                    session, global_best_coordinates
                );
                append_step(
                    result,
                    session,
                    RingUntanglingEvent::ROLLED_BACK,
                    attempt_cursor.attempts_completed
                );
                checkpoint = scan_full();
                append_step(
                    result,
                    session,
                    RingUntanglingEvent::TOPOLOGY_CHECKPOINT,
                    attempt_cursor.attempts_completed,
                    std::nullopt,
                    std::nullopt,
                    checkpoint.state,
                    checkpoint.piercing_pair_count,
                    checkpoint
                );
            }
        }
        result.final_checkpoint = checkpoint;
        result.minimum_piercing_count = std::min(
            result.minimum_piercing_count, global_minimum
        );
    }

    result.final_piercing_count = result.final_checkpoint.piercing_pair_count;
    result.resolved = result.final_piercing_count == 0;
    if (!result.resolved) {
        append_unique(
            result.warning_codes,
            stopped_without_edge
                ? "ring_piercing_has_no_opening_edge"
                : "ring_untangling_attempt_limit_reached"
        );
    }
    return result;
}


}  // namespace hotpot::forcefields
