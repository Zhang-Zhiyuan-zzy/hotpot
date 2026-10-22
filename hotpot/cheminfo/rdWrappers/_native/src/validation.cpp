#include "core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>


namespace hotpot::rdwrappers::core {
namespace {


constexpr std::uint64_t kKnownFields =
    HOTPOT_RD_FIELD_ATOMIC_NUMBERS
    | HOTPOT_RD_FIELD_FORMAL_CHARGES
    | HOTPOT_RD_FIELD_PARTIAL_CHARGES
    | HOTPOT_RD_FIELD_COORDINATES
    | HOTPOT_RD_FIELD_ATOM_AROMATIC
    | HOTPOT_RD_FIELD_BOND_INDICES
    | HOTPOT_RD_FIELD_BOND_ORDERS
    | HOTPOT_RD_FIELD_BOND_KINDS
    | HOTPOT_RD_FIELD_BOND_AROMATIC
    | HOTPOT_RD_FIELD_UNIT_CELL
    | HOTPOT_RD_FIELD_STEREO_RECORDS
    | HOTPOT_RD_FIELD_ATOM_TRANSPORT
    | HOTPOT_RD_FIELD_BOND_TRANSPORT;


constexpr std::uint64_t kRequiredFields =
    HOTPOT_RD_FIELD_ATOMIC_NUMBERS
    | HOTPOT_RD_FIELD_FORMAL_CHARGES
    | HOTPOT_RD_FIELD_ATOM_AROMATIC
    | HOTPOT_RD_FIELD_BOND_INDICES
    | HOTPOT_RD_FIELD_BOND_KINDS
    | HOTPOT_RD_FIELD_BOND_AROMATIC;


void require_pointer(const void* pointer, std::uint64_t count, const char* name) {
    if (count != 0 && pointer == nullptr) {
        throw std::invalid_argument(std::string(name) + " is null");
    }
}


void validate_string(hotpot_rd_string_view_v1 value, const char* name) {
    require_pointer(value.data, value.size, name);
}


void validate_pair(
    const std::int32_t* pair,
    std::uint64_t atom_count,
    const char* name
) {
    if (pair[0] < 0 || pair[1] < 0
        || static_cast<std::uint64_t>(pair[0]) >= atom_count
        || static_cast<std::uint64_t>(pair[1]) >= atom_count) {
        throw std::invalid_argument(std::string(name) + " atom index is out of range");
    }
    if (pair[0] == pair[1]) {
        throw std::invalid_argument(std::string(name) + " contains a self-pair");
    }
}


void validate_bound(const double* bound, const char* name) {
    if (!std::isfinite(bound[0]) || !std::isfinite(bound[1])
        || bound[0] < 0.0 || bound[1] < bound[0]) {
        throw std::invalid_argument(std::string(name) + " has invalid bounds");
    }
}


std::uint64_t pair_key(std::int32_t first, std::int32_t second) {
    const auto low = static_cast<std::uint32_t>(std::min(first, second));
    const auto high = static_cast<std::uint32_t>(std::max(first, second));
    return (static_cast<std::uint64_t>(low) << 32) | high;
}


}  // namespace


Options validate_options(const hotpot_rd_options_v1& options) {
    if (options.struct_size != sizeof(hotpot_rd_options_v1)) {
        throw AbiMismatch("hotpot_rd_options_v1 struct_size mismatch");
    }
    if (options.abi_version != HOTPOT_RD_C_API_VERSION) {
        throw AbiMismatch("hotpot_rd_options_v1 ABI version mismatch");
    }
    if (options.conformer_count == 0
        || options.conformer_count > std::numeric_limits<unsigned int>::max()) {
        throw std::invalid_argument("conformer_count is out of range");
    }
    if (options.has_seed > 1 || options.has_prune_rms_threshold > 1
        || options.use_random_coordinates > 1 || options.reserved != 0) {
        throw std::invalid_argument("hotpot_rd_options_v1 contains invalid flags");
    }
    if (options.has_seed
        && (options.seed < -1
            || options.seed > std::numeric_limits<int>::max())) {
        throw std::invalid_argument("seed is outside RDKit's supported range");
    }
    if (options.has_prune_rms_threshold
        && (!std::isfinite(options.prune_rms_threshold)
            || options.prune_rms_threshold < 0.0)) {
        throw std::invalid_argument("prune_rms_threshold must be finite and nonnegative");
    }
    return Options{
        options.conformer_count,
        options.has_seed ? std::optional<std::int64_t>(options.seed) : std::nullopt,
        options.has_prune_rms_threshold
            ? std::optional<double>(options.prune_rms_threshold)
            : std::nullopt,
        options.use_random_coordinates != 0,
    };
}


void validate_molecule_view(const hotpot_rd_molecule_view_v1& molecule) {
    if (molecule.struct_size != sizeof(hotpot_rd_molecule_view_v1)) {
        throw AbiMismatch("hotpot_rd_molecule_view_v1 struct_size mismatch");
    }
    if (molecule.abi_version != HOTPOT_RD_C_API_VERSION) {
        throw AbiMismatch("hotpot_rd_molecule_view_v1 ABI version mismatch");
    }
    if (molecule.schema_version != 1 || molecule.reserved != 0) {
        throw std::invalid_argument("unsupported molecule schema or reserved value");
    }
    if (molecule.atom_count == 0
        || molecule.atom_count > std::numeric_limits<unsigned int>::max()
        || molecule.bond_count > std::numeric_limits<unsigned int>::max()) {
        throw std::invalid_argument("molecule dimensions are out of range");
    }
    if ((molecule.field_mask & ~kKnownFields) != 0) {
        throw std::invalid_argument("molecule field_mask contains unknown fields");
    }
    if ((molecule.field_mask & kRequiredFields) != kRequiredFields) {
        throw std::invalid_argument("molecule is missing an RDKit ETKDG required field");
    }

    require_pointer(molecule.atomic_numbers, molecule.atom_count, "atomic_numbers");
    require_pointer(molecule.formal_charges, molecule.atom_count, "formal_charges");
    require_pointer(molecule.atom_aromatic, molecule.atom_count, "atom_aromatic");
    require_pointer(molecule.bond_indices, 2 * molecule.bond_count, "bond_indices");
    require_pointer(molecule.bond_kinds, molecule.bond_count, "bond_kinds");
    require_pointer(molecule.bond_aromatic, molecule.bond_count, "bond_aromatic");

    if ((molecule.field_mask & HOTPOT_RD_FIELD_PARTIAL_CHARGES) != 0) {
        require_pointer(molecule.partial_charges, molecule.atom_count, "partial_charges");
    } else if (molecule.partial_charges != nullptr) {
        throw std::invalid_argument("absent partial_charges field carries data");
    }
    if ((molecule.field_mask & HOTPOT_RD_FIELD_COORDINATES) != 0) {
        require_pointer(molecule.coordinates, 3 * molecule.atom_count, "coordinates");
    } else if (molecule.coordinates != nullptr) {
        throw std::invalid_argument("absent coordinates field carries data");
    }
    if ((molecule.field_mask & HOTPOT_RD_FIELD_BOND_ORDERS) != 0) {
        require_pointer(molecule.bond_orders, molecule.bond_count, "bond_orders");
    } else if (molecule.bond_orders != nullptr) {
        throw std::invalid_argument("absent bond_orders field carries data");
    }
    if ((molecule.field_mask & HOTPOT_RD_FIELD_UNIT_CELL) != 0) {
        require_pointer(molecule.unit_cell, 6, "unit_cell");
    } else if (molecule.unit_cell != nullptr) {
        throw std::invalid_argument("absent unit_cell field carries data");
    }

    for (std::uint64_t atom = 0; atom < molecule.atom_count; ++atom) {
        if (molecule.atomic_numbers[atom] <= 0) {
            throw std::invalid_argument("atomic_numbers must be positive");
        }
        if (molecule.atom_aromatic[atom] > 1) {
            throw std::invalid_argument("atom_aromatic contains a non-boolean value");
        }
        if ((molecule.field_mask & HOTPOT_RD_FIELD_PARTIAL_CHARGES) != 0
            && !std::isfinite(molecule.partial_charges[atom])) {
            throw std::invalid_argument("partial_charges must be finite");
        }
        if ((molecule.field_mask & HOTPOT_RD_FIELD_COORDINATES) != 0) {
            for (std::uint64_t axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(molecule.coordinates[3 * atom + axis])) {
                    throw std::invalid_argument("coordinates must be finite");
                }
            }
        }
    }

    for (std::uint64_t bond = 0; bond < molecule.bond_count; ++bond) {
        validate_pair(molecule.bond_indices + 2 * bond, molecule.atom_count, "bond");
        if (molecule.bond_kinds[bond] < HOTPOT_RD_BOND_SINGLE
            || molecule.bond_kinds[bond] > HOTPOT_RD_BOND_UNKNOWN) {
            throw std::invalid_argument("bond_kinds contains an invalid code");
        }
        if (molecule.bond_aromatic[bond] > 1) {
            throw std::invalid_argument("bond_aromatic contains a non-boolean value");
        }
        if ((molecule.field_mask & HOTPOT_RD_FIELD_BOND_ORDERS) != 0
            && (!std::isfinite(molecule.bond_orders[bond])
                || molecule.bond_orders[bond] < 0.0)) {
            throw std::invalid_argument("bond_orders must be finite and nonnegative");
        }
    }
    if ((molecule.field_mask & HOTPOT_RD_FIELD_UNIT_CELL) != 0) {
        for (std::uint32_t index = 0; index < 6; ++index) {
            if (!std::isfinite(molecule.unit_cell[index])) {
                throw std::invalid_argument("unit_cell must be finite");
            }
        }
    }

    const bool stereo_present =
        (molecule.field_mask & HOTPOT_RD_FIELD_STEREO_RECORDS) != 0;
    if (!stereo_present && (molecule.stereo_records != nullptr
                            || molecule.stereo_record_count != 0
                            || molecule.stereo_anchor_indices != nullptr
                            || molecule.stereo_anchor_index_count != 0
                            || molecule.stereo_carriers != nullptr
                            || molecule.stereo_carrier_count != 0)) {
        throw std::invalid_argument("absent stereo field carries data");
    }
    if (stereo_present) {
        require_pointer(molecule.stereo_records, molecule.stereo_record_count, "stereo_records");
        require_pointer(molecule.stereo_anchor_indices, molecule.stereo_anchor_index_count, "stereo_anchor_indices");
        require_pointer(molecule.stereo_carriers, molecule.stereo_carrier_count, "stereo_carriers");
        for (std::uint64_t index = 0; index < molecule.stereo_record_count; ++index) {
            const auto& record = molecule.stereo_records[index];
            if (record.group_kind > HOTPOT_RD_STEREO_GROUP_OR
                || record.has_group > 1 || record.reserved != 0
                || record.anchor_offset > molecule.stereo_anchor_index_count
                || record.anchor_count > molecule.stereo_anchor_index_count - record.anchor_offset
                || record.carrier_offset > molecule.stereo_carrier_count
                || record.carrier_count > molecule.stereo_carrier_count - record.carrier_offset) {
                throw std::invalid_argument("stereo record is structurally invalid");
            }
            if ((!record.has_group
                 && (record.group_kind != HOTPOT_RD_STEREO_GROUP_NONE
                     || record.group_id != 0
                     || record.group_scope.size != 0))
                || (record.has_group
                    && (record.group_kind == HOTPOT_RD_STEREO_GROUP_NONE
                        || record.group_id < 0
                        || static_cast<std::uint64_t>(record.group_id)
                            > std::numeric_limits<unsigned int>::max()
                        || record.group_scope.size == 0))) {
                throw std::invalid_argument(
                    "stereo record has an inconsistent group identity"
                );
            }
            bool valid_anchor_length = false;
            switch (record.anchor_kind) {
                case HOTPOT_RD_STEREO_ANCHOR_ATOM:
                    valid_anchor_length = record.anchor_count == 1;
                    break;
                case HOTPOT_RD_STEREO_ANCHOR_BOND:
                    valid_anchor_length = record.anchor_count == 2;
                    break;
                case HOTPOT_RD_STEREO_ANCHOR_AXIS:
                    valid_anchor_length = record.anchor_count >= 3;
                    break;
                default:
                    throw std::invalid_argument(
                        "stereo record has an invalid anchor kind"
                    );
            }
            if (!valid_anchor_length) {
                throw std::invalid_argument("stereo record has an invalid anchor length");
            }
            for (std::uint64_t offset = 0; offset < record.anchor_count; ++offset) {
                if (molecule.stereo_anchor_indices[record.anchor_offset + offset]
                    >= molecule.atom_count) {
                    throw std::invalid_argument("stereo anchor atom is out of range");
                }
            }
            validate_string(record.group_scope, "stereo group_scope");
            validate_string(record.descriptor, "stereo descriptor");
            validate_string(record.source_namespace, "stereo source_namespace");
            validate_string(record.source_code, "stereo source_code");
        }
        for (std::uint64_t index = 0; index < molecule.stereo_carrier_count; ++index) {
            const auto& carrier = molecule.stereo_carriers[index];
            if (carrier.kind < HOTPOT_RD_STEREO_CARRIER_ATOM
                || carrier.kind > HOTPOT_RD_STEREO_CARRIER_VACANCY
                || carrier.has_atom > 1
                || (carrier.kind == HOTPOT_RD_STEREO_CARRIER_ATOM) != (carrier.has_atom != 0)
                || (carrier.has_atom && carrier.atom >= molecule.atom_count)) {
                throw std::invalid_argument("stereo carrier is structurally invalid");
            }
        }
    }

    const bool atom_transport_present =
        (molecule.field_mask & HOTPOT_RD_FIELD_ATOM_TRANSPORT) != 0;
    if (!atom_transport_present
        && (molecule.atom_transport != nullptr || molecule.atom_transport_count != 0)) {
        throw std::invalid_argument("absent atom transport field carries data");
    }
    if (atom_transport_present) {
        require_pointer(molecule.atom_transport, molecule.atom_transport_count, "atom_transport");
        std::unordered_set<std::uint64_t> atoms;
        for (std::uint64_t index = 0; index < molecule.atom_transport_count; ++index) {
            const auto& record = molecule.atom_transport[index];
            if (record.atom >= molecule.atom_count || !atoms.insert(record.atom).second
                || (record.presence_mask & ~(HOTPOT_RD_ATOM_TRANSPORT_ISOTOPE
                                             | HOTPOT_RD_ATOM_TRANSPORT_RADICAL_ELECTRONS)) != 0
                || record.has_current_hydrogens > 1 || record.has_source_hydrogens > 1) {
                throw std::invalid_argument("atom transport record is structurally invalid");
            }
            validate_string(record.source_namespace, "atom source_namespace");
            const auto validate_hydrogens = [](const hotpot_rd_hydrogen_state_v1& state) {
                if ((state.presence_mask & ~(HOTPOT_RD_HYDROGEN_TRANSPORT_IMPLICIT
                                             | HOTPOT_RD_HYDROGEN_TRANSPORT_BRACKET_EXPLICIT
                                             | HOTPOT_RD_HYDROGEN_TRANSPORT_NO_IMPLICIT)) != 0
                    || state.implicit_hydrogens < 0
                    || state.bracket_explicit_hydrogens < 0
                    || state.no_implicit > 1) {
                    throw std::invalid_argument("hydrogen transport state is invalid");
                }
            };
            if (record.has_current_hydrogens) {
                validate_hydrogens(record.current_hydrogens);
            }
            if (record.has_source_hydrogens) {
                validate_hydrogens(record.source_hydrogens);
            }
        }
    }

    const bool bond_transport_present =
        (molecule.field_mask & HOTPOT_RD_FIELD_BOND_TRANSPORT) != 0;
    if (!bond_transport_present
        && (molecule.bond_transport != nullptr || molecule.bond_transport_count != 0)) {
        throw std::invalid_argument("absent bond transport field carries data");
    }
    if (bond_transport_present) {
        require_pointer(molecule.bond_transport, molecule.bond_transport_count, "bond_transport");
        std::unordered_set<std::uint64_t> pairs;
        for (std::uint64_t index = 0; index < molecule.bond_transport_count; ++index) {
            const auto& record = molecule.bond_transport[index];
            const std::int32_t pair[2] = {
                static_cast<std::int32_t>(record.first_atom),
                static_cast<std::int32_t>(record.second_atom),
            };
            if (record.first_atom > std::numeric_limits<std::int32_t>::max()
                || record.second_atom > std::numeric_limits<std::int32_t>::max()) {
                throw std::invalid_argument("bond transport atom index is out of range");
            }
            validate_pair(pair, molecule.atom_count, "bond transport");
            if (!pairs.insert(pair_key(pair[0], pair[1])).second) {
                throw std::invalid_argument("duplicate bond transport record");
            }
            validate_string(record.display_direction, "bond display_direction");
        }
    }
}


void validate_constraints_view(
    const hotpot_rd_constraints_view_v1& constraints,
    std::uint64_t atom_count
) {
    if (constraints.struct_size != sizeof(hotpot_rd_constraints_view_v1)) {
        throw AbiMismatch("hotpot_rd_constraints_view_v1 struct_size mismatch");
    }
    if (constraints.abi_version != HOTPOT_RD_C_API_VERSION) {
        throw AbiMismatch("hotpot_rd_constraints_view_v1 ABI version mismatch");
    }
    require_pointer(constraints.coordination_pairs, 2 * constraints.coordination_pair_count, "coordination_pairs");
    require_pointer(constraints.metal_donor_bounds, 2 * constraints.coordination_pair_count, "metal_donor_bounds");
    require_pointer(constraints.donor_pair_indices, 2 * constraints.donor_pair_count, "donor_pair_indices");
    require_pointer(constraints.donor_pair_bounds, 2 * constraints.donor_pair_count, "donor_pair_bounds");

    std::unordered_set<std::uint64_t> pairs;
    for (std::uint64_t index = 0; index < constraints.coordination_pair_count; ++index) {
        const auto* pair = constraints.coordination_pairs + 2 * index;
        validate_pair(pair, atom_count, "coordination constraint");
        validate_bound(constraints.metal_donor_bounds + 2 * index, "coordination constraint");
        if (!pairs.insert(pair_key(pair[0], pair[1])).second) {
            throw std::invalid_argument("duplicate coordination constraint pair");
        }
    }
    for (std::uint64_t index = 0; index < constraints.donor_pair_count; ++index) {
        const auto* pair = constraints.donor_pair_indices + 2 * index;
        validate_pair(pair, atom_count, "donor-pair constraint");
        validate_bound(constraints.donor_pair_bounds + 2 * index, "donor-pair constraint");
        if (!pairs.insert(pair_key(pair[0], pair[1])).second) {
            throw std::invalid_argument("duplicate constraint pair");
        }
    }
}


}  // namespace hotpot::rdwrappers::core
