#include "core.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>


static hotpot_rd_string_view_v1 text(const char* value) {
    return {value, static_cast<std::uint64_t>(std::strlen(value))};
}


int main() {
    const std::int32_t atomic_numbers[] = {6, 6, 6, 6};
    const std::int32_t formal_charges[] = {0, 0, 0, 0};
    const std::uint8_t atom_aromatic[] = {0, 0, 0, 0};
    const std::int32_t bond_indices[] = {0, 1, 1, 2, 2, 3};
    const std::uint8_t bond_kinds[] = {
        HOTPOT_RD_BOND_SINGLE,
        HOTPOT_RD_BOND_DOUBLE,
        HOTPOT_RD_BOND_SINGLE,
    };
    const std::uint8_t bond_aromatic[] = {0, 0, 0};
    const std::uint64_t anchors[] = {0, 2, 1};
    const hotpot_rd_stereo_carrier_v1 carriers[] = {
        {HOTPOT_RD_STEREO_CARRIER_ATOM, 1, 3},
        {HOTPOT_RD_STEREO_CARRIER_ATOM, 1, 0},
    };
    const hotpot_rd_stereo_record_v1 stereo[] = {
        {
            HOTPOT_RD_STEREO_ANCHOR_ATOM,
            HOTPOT_RD_STEREO_GROUP_ABS,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            text("scope-a"),
            text("tetrahedral"),
            text("rdkit-2026.09.1"),
            text("CHI_UNSPECIFIED"),
        },
        {
            HOTPOT_RD_STEREO_ANCHOR_BOND,
            HOTPOT_RD_STEREO_GROUP_NONE,
            1,
            2,
            0,
            2,
            0,
            0,
            0,
            text(""),
            text("double-bond"),
            text("rdkit-2026.09.1"),
            text("STEREOE"),
        },
    };
    const hotpot_rd_atom_transport_v1 atom_transport[] = {
        {
            3,
            HOTPOT_RD_ATOM_TRANSPORT_ISOTOPE
                | HOTPOT_RD_ATOM_TRANSPORT_RADICAL_ELECTRONS,
            18,
            1,
            1,
            1,
            {HOTPOT_RD_HYDROGEN_TRANSPORT_NO_IMPLICIT, 0, 0, 1},
            {HOTPOT_RD_HYDROGEN_TRANSPORT_NO_IMPLICIT, 0, 0, 0},
            text("rdkit-2026.09.1"),
        },
    };
    const hotpot_rd_bond_transport_v1 bond_transport[] = {
        {0, 1, text("BEGINWEDGE")},
    };

    hotpot_rd_molecule_view_v1 source{};
    source.struct_size = sizeof(source);
    source.abi_version = HOTPOT_RD_C_API_VERSION;
    source.schema_version = 1;
    source.atom_count = 4;
    source.bond_count = 3;
    source.field_mask = HOTPOT_RD_FIELD_ATOMIC_NUMBERS
        | HOTPOT_RD_FIELD_FORMAL_CHARGES
        | HOTPOT_RD_FIELD_ATOM_AROMATIC
        | HOTPOT_RD_FIELD_BOND_INDICES
        | HOTPOT_RD_FIELD_BOND_KINDS
        | HOTPOT_RD_FIELD_BOND_AROMATIC
        | HOTPOT_RD_FIELD_STEREO_RECORDS
        | HOTPOT_RD_FIELD_ATOM_TRANSPORT
        | HOTPOT_RD_FIELD_BOND_TRANSPORT;
    source.atomic_numbers = atomic_numbers;
    source.formal_charges = formal_charges;
    source.atom_aromatic = atom_aromatic;
    source.bond_indices = bond_indices;
    source.bond_kinds = bond_kinds;
    source.bond_aromatic = bond_aromatic;
    source.stereo_records = stereo;
    source.stereo_record_count = 2;
    source.stereo_anchor_indices = anchors;
    source.stereo_anchor_index_count = 3;
    source.stereo_carriers = carriers;
    source.stereo_carrier_count = 2;
    source.atom_transport = atom_transport;
    source.atom_transport_count = 1;
    source.bond_transport = bond_transport;
    source.bond_transport_count = 1;

    hotpot::rdwrappers::core::validate_molecule_view(source);
    auto molecule = hotpot::rdwrappers::core::build_molecule(source);
    assert(molecule->getNumAtoms() == 4);
    assert(molecule->getAtomWithIdx(0)->getChiralTag()
           == RDKit::Atom::CHI_UNSPECIFIED);
    assert(molecule->getAtomWithIdx(3)->getIsotope() == 18);
    assert(molecule->getAtomWithIdx(3)->getNumRadicalElectrons() == 1);
    assert(molecule->getAtomWithIdx(3)->getNoImplicit());
    assert(molecule->getBondBetweenAtoms(0, 1)->getBondDir()
           == RDKit::Bond::BEGINWEDGE);
    assert(molecule->getBondBetweenAtoms(1, 2)->getStereo()
           == RDKit::Bond::STEREOE);
    assert(molecule->getBondBetweenAtoms(1, 2)->getStereoAtoms()[0] == 0);
    assert(molecule->getBondBetweenAtoms(1, 2)->getStereoAtoms()[1] == 3);
    assert(molecule->getStereoGroups().size() == 1);
    assert(molecule->getStereoGroups()[0].getReadId() == 0);

    const std::int32_t coordination_numbers[] = {7, 30, 6};
    const std::int32_t coordination_charges[] = {0, 0, 0};
    const std::uint8_t coordination_atom_aromatic[] = {0, 0, 0};
    const std::int32_t coordination_indices[] = {0, 1, 1, 2};
    const std::uint8_t coordination_kinds[] = {
        HOTPOT_RD_BOND_DATIVE,
        HOTPOT_RD_BOND_ZERO,
    };
    const std::uint8_t coordination_bond_aromatic[] = {0, 0};
    hotpot_rd_molecule_view_v1 coordination{};
    coordination.struct_size = sizeof(coordination);
    coordination.abi_version = HOTPOT_RD_C_API_VERSION;
    coordination.schema_version = 1;
    coordination.atom_count = 3;
    coordination.bond_count = 2;
    coordination.field_mask = HOTPOT_RD_FIELD_ATOMIC_NUMBERS
        | HOTPOT_RD_FIELD_FORMAL_CHARGES
        | HOTPOT_RD_FIELD_ATOM_AROMATIC
        | HOTPOT_RD_FIELD_BOND_INDICES
        | HOTPOT_RD_FIELD_BOND_KINDS
        | HOTPOT_RD_FIELD_BOND_AROMATIC;
    coordination.atomic_numbers = coordination_numbers;
    coordination.formal_charges = coordination_charges;
    coordination.atom_aromatic = coordination_atom_aromatic;
    coordination.bond_indices = coordination_indices;
    coordination.bond_kinds = coordination_kinds;
    coordination.bond_aromatic = coordination_bond_aromatic;
    hotpot::rdwrappers::core::validate_molecule_view(coordination);
    auto complex = hotpot::rdwrappers::core::build_molecule(coordination);
    const auto* dative = complex->getBondBetweenAtoms(0, 1);
    const auto* zero = complex->getBondBetweenAtoms(1, 2);
    assert(dative->getBondType() == RDKit::Bond::DATIVE);
    assert(dative->getBeginAtomIdx() == 0 && dative->getEndAtomIdx() == 1);
    assert(zero->getBondType() == RDKit::Bond::ZERO);

    const std::int32_t aromatic_numbers[] = {6, 6, 6, 6, 6, 6};
    const std::int32_t aromatic_charges[] = {0, 0, 0, 0, 0, 0};
    const std::uint8_t aromatic_atoms[] = {1, 1, 1, 1, 1, 1};
    const std::int32_t aromatic_indices[] = {
        0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0,
    };
    const std::uint8_t aromatic_kinds[] = {
        HOTPOT_RD_BOND_AROMATIC,
        HOTPOT_RD_BOND_AROMATIC,
        HOTPOT_RD_BOND_AROMATIC,
        HOTPOT_RD_BOND_AROMATIC,
        HOTPOT_RD_BOND_AROMATIC,
        HOTPOT_RD_BOND_AROMATIC,
    };
    const std::uint8_t aromatic_bonds[] = {1, 1, 1, 1, 1, 1};
    hotpot_rd_molecule_view_v1 aromatic{};
    aromatic.struct_size = sizeof(aromatic);
    aromatic.abi_version = HOTPOT_RD_C_API_VERSION;
    aromatic.schema_version = 1;
    aromatic.atom_count = 6;
    aromatic.bond_count = 6;
    aromatic.field_mask = coordination.field_mask;
    aromatic.atomic_numbers = aromatic_numbers;
    aromatic.formal_charges = aromatic_charges;
    aromatic.atom_aromatic = aromatic_atoms;
    aromatic.bond_indices = aromatic_indices;
    aromatic.bond_kinds = aromatic_kinds;
    aromatic.bond_aromatic = aromatic_bonds;
    hotpot::rdwrappers::core::validate_molecule_view(aromatic);
    auto benzene = hotpot::rdwrappers::core::build_molecule(aromatic);
    for (const auto* atom : benzene->atoms()) assert(atom->getIsAromatic());
    for (const auto* bond : benzene->bonds()) {
        assert(bond->getIsAromatic());
        assert(bond->getBondType() == RDKit::Bond::AROMATIC);
    }

    const std::int32_t chiral_numbers[] = {6, 9, 17, 35};
    const std::int32_t chiral_charges[] = {0, 0, 0, 0};
    const std::uint8_t chiral_aromatic_atoms[] = {0, 0, 0, 0};
    const std::int32_t chiral_indices[] = {0, 1, 0, 2, 0, 3};
    const std::uint8_t chiral_kinds[] = {
        HOTPOT_RD_BOND_SINGLE,
        HOTPOT_RD_BOND_SINGLE,
        HOTPOT_RD_BOND_SINGLE,
    };
    const std::uint8_t chiral_aromatic_bonds[] = {0, 0, 0};
    const std::uint64_t chiral_anchor[] = {0};
    const hotpot_rd_stereo_carrier_v1 odd_carriers[] = {
        {HOTPOT_RD_STEREO_CARRIER_ATOM, 1, 2},
        {HOTPOT_RD_STEREO_CARRIER_ATOM, 1, 1},
        {HOTPOT_RD_STEREO_CARRIER_ATOM, 1, 3},
        {HOTPOT_RD_STEREO_CARRIER_IMPLICIT_H, 0, 0},
    };
    const hotpot_rd_stereo_record_v1 chiral_record[] = {
        {
            HOTPOT_RD_STEREO_ANCHOR_ATOM,
            HOTPOT_RD_STEREO_GROUP_NONE,
            0,
            1,
            0,
            4,
            0,
            0,
            0,
            text(""),
            text("tetrahedral"),
            text("rdkit-2026.09.1"),
            text("CHI_TETRAHEDRAL_CW"),
        },
    };
    hotpot_rd_molecule_view_v1 chiral{};
    chiral.struct_size = sizeof(chiral);
    chiral.abi_version = HOTPOT_RD_C_API_VERSION;
    chiral.schema_version = 1;
    chiral.atom_count = 4;
    chiral.bond_count = 3;
    chiral.field_mask = coordination.field_mask
        | HOTPOT_RD_FIELD_STEREO_RECORDS;
    chiral.atomic_numbers = chiral_numbers;
    chiral.formal_charges = chiral_charges;
    chiral.atom_aromatic = chiral_aromatic_atoms;
    chiral.bond_indices = chiral_indices;
    chiral.bond_kinds = chiral_kinds;
    chiral.bond_aromatic = chiral_aromatic_bonds;
    chiral.stereo_records = chiral_record;
    chiral.stereo_record_count = 1;
    chiral.stereo_anchor_indices = chiral_anchor;
    chiral.stereo_anchor_index_count = 1;
    chiral.stereo_carriers = odd_carriers;
    chiral.stereo_carrier_count = 4;
    hotpot::rdwrappers::core::validate_molecule_view(chiral);
    auto odd_chiral = hotpot::rdwrappers::core::build_molecule(chiral);
    assert(odd_chiral->getAtomWithIdx(0)->getChiralTag()
           == RDKit::Atom::CHI_TETRAHEDRAL_CCW);

    hotpot_rd_stereo_carrier_v1 explicit_h_carriers[] = {
        odd_carriers[0],
        odd_carriers[1],
        odd_carriers[2],
    };
    hotpot_rd_stereo_record_v1 explicit_h_record = chiral_record[0];
    explicit_h_record.carrier_count = 3;
    const hotpot_rd_atom_transport_v1 explicit_h_transport[] = {
        {
            0,
            0,
            0,
            0,
            1,
            1,
            {
                HOTPOT_RD_HYDROGEN_TRANSPORT_IMPLICIT
                    | HOTPOT_RD_HYDROGEN_TRANSPORT_BRACKET_EXPLICIT
                    | HOTPOT_RD_HYDROGEN_TRANSPORT_NO_IMPLICIT,
                0,
                1,
                1,
            },
            {
                HOTPOT_RD_HYDROGEN_TRANSPORT_IMPLICIT
                    | HOTPOT_RD_HYDROGEN_TRANSPORT_BRACKET_EXPLICIT
                    | HOTPOT_RD_HYDROGEN_TRANSPORT_NO_IMPLICIT,
                0,
                1,
                1,
            },
            text("rdkit-2026.09.1"),
        },
    };
    hotpot_rd_molecule_view_v1 explicit_h_chiral = chiral;
    explicit_h_chiral.field_mask |= HOTPOT_RD_FIELD_ATOM_TRANSPORT;
    explicit_h_chiral.stereo_records = &explicit_h_record;
    explicit_h_chiral.stereo_carriers = explicit_h_carriers;
    explicit_h_chiral.stereo_carrier_count = 3;
    explicit_h_chiral.atom_transport = explicit_h_transport;
    explicit_h_chiral.atom_transport_count = 1;
    hotpot::rdwrappers::core::validate_molecule_view(explicit_h_chiral);
    auto bracket_h_chiral =
        hotpot::rdwrappers::core::build_molecule(explicit_h_chiral);
    assert(bracket_h_chiral->getAtomWithIdx(0)->getNumExplicitHs() == 1);
    assert(bracket_h_chiral->getAtomWithIdx(0)->getChiralTag()
           == RDKit::Atom::CHI_TETRAHEDRAL_CCW);

    const hotpot_rd_bond_transport_v1 reversed_end_direction[] = {
        {1, 0, text("ENDDOWNRIGHT")},
    };
    hotpot_rd_molecule_view_v1 reversed_direction = source;
    reversed_direction.bond_transport = reversed_end_direction;
    reversed_direction.bond_transport_count = 1;
    auto reversed_direction_molecule =
        hotpot::rdwrappers::core::build_molecule(reversed_direction);
    assert(reversed_direction_molecule->getBondBetweenAtoms(0, 1)->getBondDir()
           == RDKit::Bond::ENDUPRIGHT);

    const hotpot_rd_bond_transport_v1 reversed_either_double[] = {
        {2, 1, text("EITHERDOUBLE")},
    };
    reversed_direction.bond_transport = reversed_either_double;
    auto reversed_either_molecule =
        hotpot::rdwrappers::core::build_molecule(reversed_direction);
    assert(reversed_either_molecule->getBondBetweenAtoms(1, 2)->getBondDir()
           == RDKit::Bond::EITHERDOUBLE);

    const hotpot_rd_bond_transport_v1 reversed_wedge[] = {
        {1, 0, text("BEGINWEDGE")},
    };
    reversed_direction.bond_transport = reversed_wedge;
    bool reversed_wedge_rejected = false;
    try {
        (void)hotpot::rdwrappers::core::build_molecule(reversed_direction);
    } catch (const hotpot::rdwrappers::core::UnsupportedInput&) {
        reversed_wedge_rejected = true;
    }
    assert(reversed_wedge_rejected);

    hotpot_rd_stereo_record_v1 reversed_atrop_records[] = {
        stereo[0],
        stereo[1],
    };
    reversed_atrop_records[1].source_code = text("STEREOATROPCW");
    hotpot_rd_molecule_view_v1 reversed_atrop = source;
    reversed_atrop.stereo_records = reversed_atrop_records;
    bool reversed_atrop_rejected = false;
    try {
        (void)hotpot::rdwrappers::core::build_molecule(reversed_atrop);
    } catch (const hotpot::rdwrappers::core::UnsupportedInput&) {
        reversed_atrop_rejected = true;
    }
    assert(reversed_atrop_rejected);

    hotpot_rd_atom_transport_v1 unknown_atom_transport = atom_transport[0];
    unknown_atom_transport.source_namespace = text("future-toolkit");
    hotpot_rd_molecule_view_v1 unknown_atom_namespace = source;
    unknown_atom_namespace.atom_transport = &unknown_atom_transport;
    bool unknown_atom_namespace_rejected = false;
    try {
        (void)hotpot::rdwrappers::core::build_molecule(unknown_atom_namespace);
    } catch (const hotpot::rdwrappers::core::UnsupportedInput&) {
        unknown_atom_namespace_rejected = true;
    }
    assert(unknown_atom_namespace_rejected);

    hotpot_rd_stereo_carrier_v1 unsupported_carriers[] = {
        odd_carriers[0],
        {HOTPOT_RD_STEREO_CARRIER_LONE_PAIR, 0, 0},
        odd_carriers[2],
        odd_carriers[3],
    };
    hotpot_rd_molecule_view_v1 unsupported_chiral = chiral;
    unsupported_chiral.stereo_carriers = unsupported_carriers;
    bool unsupported_rejected = false;
    try {
        (void)hotpot::rdwrappers::core::build_molecule(unsupported_chiral);
    } catch (const hotpot::rdwrappers::core::UnsupportedInput&) {
        unsupported_rejected = true;
    }
    assert(unsupported_rejected);

    hotpot_rd_stereo_record_v1 duplicate_records[] = {
        chiral_record[0],
        chiral_record[0],
    };
    const std::uint64_t duplicate_anchors[] = {0, 0};
    duplicate_records[1].anchor_offset = 1;
    hotpot_rd_molecule_view_v1 duplicate_chiral = chiral;
    duplicate_chiral.stereo_records = duplicate_records;
    duplicate_chiral.stereo_record_count = 2;
    duplicate_chiral.stereo_anchor_indices = duplicate_anchors;
    duplicate_chiral.stereo_anchor_index_count = 2;
    hotpot::rdwrappers::core::validate_molecule_view(duplicate_chiral);
    bool duplicate_rejected = false;
    try {
        (void)hotpot::rdwrappers::core::build_molecule(duplicate_chiral);
    } catch (const hotpot::rdwrappers::core::UnsupportedInput&) {
        duplicate_rejected = true;
    }
    assert(duplicate_rejected);

    hotpot_rd_stereo_record_v1 inconsistent_group = chiral_record[0];
    inconsistent_group.group_kind = HOTPOT_RD_STEREO_GROUP_AND;
    hotpot_rd_molecule_view_v1 invalid_group = chiral;
    invalid_group.stereo_records = &inconsistent_group;
    bool invalid_group_rejected = false;
    try {
        hotpot::rdwrappers::core::validate_molecule_view(invalid_group);
    } catch (const std::invalid_argument&) {
        invalid_group_rejected = true;
    }
    assert(invalid_group_rejected);
    return 0;
}
