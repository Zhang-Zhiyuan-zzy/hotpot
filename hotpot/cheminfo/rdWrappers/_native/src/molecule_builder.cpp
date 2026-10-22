#include "core.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/StereoGroup.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>


namespace hotpot::rdwrappers::core {
namespace {


std::string copy_string(hotpot_rd_string_view_v1 value) {
    return value.data == nullptr
        ? std::string{}
        : std::string(value.data, static_cast<std::size_t>(value.size));
}


RDKit::Bond::BondType bond_type(std::uint8_t kind) {
    switch (kind) {
        case HOTPOT_RD_BOND_SINGLE:
            return RDKit::Bond::SINGLE;
        case HOTPOT_RD_BOND_DOUBLE:
            return RDKit::Bond::DOUBLE;
        case HOTPOT_RD_BOND_TRIPLE:
            return RDKit::Bond::TRIPLE;
        case HOTPOT_RD_BOND_AROMATIC:
            return RDKit::Bond::AROMATIC;
        case HOTPOT_RD_BOND_ZERO:
            return RDKit::Bond::ZERO;
        case HOTPOT_RD_BOND_DATIVE:
            return RDKit::Bond::DATIVE;
        case HOTPOT_RD_BOND_UNKNOWN:
            throw UnsupportedInput("RDKit cannot construct an UNKNOWN bond kind");
        default:
            throw std::invalid_argument("bond kind code is invalid");
    }
}


double expected_bond_order(std::uint8_t kind) {
    switch (kind) {
        case HOTPOT_RD_BOND_SINGLE:
        case HOTPOT_RD_BOND_DATIVE:
            return 1.0;
        case HOTPOT_RD_BOND_DOUBLE:
            return 2.0;
        case HOTPOT_RD_BOND_TRIPLE:
            return 3.0;
        case HOTPOT_RD_BOND_AROMATIC:
            return 1.5;
        case HOTPOT_RD_BOND_ZERO:
            return 0.0;
        case HOTPOT_RD_BOND_UNKNOWN:
            throw UnsupportedInput("RDKit cannot construct an UNKNOWN bond kind");
        default:
            throw std::invalid_argument("bond kind code is invalid");
    }
}


RDKit::Atom::ChiralType atom_chiral_type(const std::string& code) {
    if (code == "CHI_UNSPECIFIED") return RDKit::Atom::CHI_UNSPECIFIED;
    if (code == "CHI_TETRAHEDRAL_CW") return RDKit::Atom::CHI_TETRAHEDRAL_CW;
    if (code == "CHI_TETRAHEDRAL_CCW") return RDKit::Atom::CHI_TETRAHEDRAL_CCW;
    if (code == "CHI_OTHER") return RDKit::Atom::CHI_OTHER;
    if (code == "CHI_TETRAHEDRAL") return RDKit::Atom::CHI_TETRAHEDRAL;
    if (code == "CHI_ALLENE") return RDKit::Atom::CHI_ALLENE;
    if (code == "CHI_SQUAREPLANAR") return RDKit::Atom::CHI_SQUAREPLANAR;
    if (code == "CHI_TRIGONALBIPYRAMIDAL") {
        return RDKit::Atom::CHI_TRIGONALBIPYRAMIDAL;
    }
    if (code == "CHI_OCTAHEDRAL") return RDKit::Atom::CHI_OCTAHEDRAL;
    throw UnsupportedInput("unsupported RDKit atom stereo source_code: " + code);
}


using AtomCarrierToken = std::pair<std::uint32_t, std::uint64_t>;


// An RDKit-source tetrahedral record's source_code is relative to its ordered
// carrier sequence. RDKit stores that same tag relative to atomBonds() order,
// with one implicit-H carrier appended last. An odd permutation between those
// orders therefore requires inversion of CW/CCW.
std::vector<AtomCarrierToken> native_atom_carrier_order(
    const RDKit::RWMol& molecule,
    const RDKit::Atom& atom
) {
    std::vector<AtomCarrierToken> result;
    result.reserve(atom.getDegree() + 1);
    for (const auto* bond : molecule.atomBonds(&atom)) {
        result.emplace_back(
            HOTPOT_RD_STEREO_CARRIER_ATOM,
            bond->getOtherAtomIdx(atom.getIdx())
        );
    }
    const auto bracket_explicit_hydrogens = atom.getNumExplicitHs();
    const auto implicit_hydrogens = atom.getNumImplicitHs();
    if (bracket_explicit_hydrogens > 1
        || implicit_hydrogens > 1
        || bracket_explicit_hydrogens + implicit_hydrogens > 1) {
        throw UnsupportedInput(
            "RDKit tetrahedral atom stereo cannot represent multiple "
            "non-graph hydrogen carriers"
        );
    }
    if (implicit_hydrogens == 1) {
        result.emplace_back(HOTPOT_RD_STEREO_CARRIER_IMPLICIT_H, 0);
    }
    return result;
}


std::vector<AtomCarrierToken> transported_atom_carrier_order(
    const hotpot_rd_molecule_view_v1& source,
    const hotpot_rd_stereo_record_v1& record
) {
    std::vector<AtomCarrierToken> result;
    result.reserve(static_cast<std::size_t>(record.carrier_count));
    for (std::uint64_t offset = 0; offset < record.carrier_count; ++offset) {
        const auto& carrier =
            source.stereo_carriers[record.carrier_offset + offset];
        if (carrier.kind == HOTPOT_RD_STEREO_CARRIER_ATOM) {
            result.emplace_back(carrier.kind, carrier.atom);
        } else if (carrier.kind == HOTPOT_RD_STEREO_CARRIER_IMPLICIT_H) {
            result.emplace_back(carrier.kind, 0);
        } else {
            throw UnsupportedInput(
                "RDKit tetrahedral atom stereo cannot represent lone-pair "
                "or vacancy carriers"
            );
        }
    }
    return result;
}


bool odd_permutation(
    std::vector<AtomCarrierToken> source,
    const std::vector<AtomCarrierToken>& target
) {
    if (source.size() != target.size()) {
        throw UnsupportedInput(
            "atom stereo carrier count does not match RDKit neighbor order"
        );
    }
    bool odd = false;
    for (std::size_t index = 0; index < target.size(); ++index) {
        const auto match = std::find(
            source.begin() + static_cast<std::ptrdiff_t>(index),
            source.end(),
            target[index]
        );
        if (match == source.end()) {
            throw UnsupportedInput(
                "atom stereo carriers are not a permutation of RDKit neighbors"
            );
        }
        const auto match_index = static_cast<std::size_t>(
            std::distance(source.begin(), match)
        );
        if (match_index != index) {
            std::swap(source[index], source[match_index]);
            odd = !odd;
        }
    }
    return odd;
}


void apply_atom_stereo(
    RDKit::RWMol& molecule,
    RDKit::Atom& atom,
    const hotpot_rd_molecule_view_v1& source,
    const hotpot_rd_stereo_record_v1& record,
    const std::string& source_code
) {
    auto chiral_type = atom_chiral_type(source_code);
    if (chiral_type == RDKit::Atom::CHI_UNSPECIFIED) {
        if (record.carrier_count != 0) {
            throw UnsupportedInput(
                "CHI_UNSPECIFIED atom stereo cannot carry an ordered sequence"
            );
        }
        atom.setChiralTag(chiral_type);
        return;
    }
    if (chiral_type != RDKit::Atom::CHI_TETRAHEDRAL_CW
        && chiral_type != RDKit::Atom::CHI_TETRAHEDRAL_CCW) {
        throw UnsupportedInput(
            "RDKit atom stereo source_code requires an unfrozen permutation "
            "property: " + source_code
        );
    }

    const auto transported = transported_atom_carrier_order(source, record);
    const auto native = native_atom_carrier_order(molecule, atom);
    const auto virtual_hydrogens =
        atom.getNumExplicitHs() + atom.getNumImplicitHs();
    const auto total_carriers = atom.getDegree() + virtual_hydrogens;
    const bool valid_three_carrier_virtual_h =
        native.size() == 3 && atom.getNumExplicitHs() == 1
        && virtual_hydrogens == 1;
    if (total_carriers != 4
        || (!valid_three_carrier_virtual_h && native.size() != 4)
        || transported.size() != native.size()) {
        throw UnsupportedInput(
            "RDKit tetrahedral atom stereo requires four represented "
            "carriers or three atom carriers plus one bracket-explicit "
            "virtual hydrogen"
        );
    }
    if (odd_permutation(transported, native)) {
        chiral_type = chiral_type == RDKit::Atom::CHI_TETRAHEDRAL_CW
            ? RDKit::Atom::CHI_TETRAHEDRAL_CCW
            : RDKit::Atom::CHI_TETRAHEDRAL_CW;
    }
    atom.setChiralTag(chiral_type);
}


RDKit::Bond::BondStereo bond_stereo_type(const std::string& code) {
    if (code == "STEREONONE") return RDKit::Bond::STEREONONE;
    if (code == "STEREOANY") return RDKit::Bond::STEREOANY;
    if (code == "STEREOZ") return RDKit::Bond::STEREOZ;
    if (code == "STEREOE") return RDKit::Bond::STEREOE;
    if (code == "STEREOCIS") return RDKit::Bond::STEREOCIS;
    if (code == "STEREOTRANS") return RDKit::Bond::STEREOTRANS;
    if (code == "STEREOATROPCW") return RDKit::Bond::STEREOATROPCW;
    if (code == "STEREOATROPCCW") return RDKit::Bond::STEREOATROPCCW;
    throw UnsupportedInput("unsupported RDKit bond stereo source_code: " + code);
}


RDKit::Bond::BondDir bond_direction(const std::string& value) {
    if (value == "NONE") return RDKit::Bond::NONE;
    if (value == "BEGINWEDGE") return RDKit::Bond::BEGINWEDGE;
    if (value == "BEGINDASH") return RDKit::Bond::BEGINDASH;
    if (value == "ENDDOWNRIGHT") return RDKit::Bond::ENDDOWNRIGHT;
    if (value == "ENDUPRIGHT") return RDKit::Bond::ENDUPRIGHT;
    if (value == "EITHERDOUBLE") return RDKit::Bond::EITHERDOUBLE;
    if (value == "UNKNOWN") return RDKit::Bond::UNKNOWN;
    throw UnsupportedInput("unsupported RDKit bond display_direction: " + value);
}


RDKit::StereoGroupType group_type(std::uint32_t kind) {
    switch (kind) {
        case HOTPOT_RD_STEREO_GROUP_ABS:
            return RDKit::StereoGroupType::STEREO_ABSOLUTE;
        case HOTPOT_RD_STEREO_GROUP_AND:
            return RDKit::StereoGroupType::STEREO_AND;
        case HOTPOT_RD_STEREO_GROUP_OR:
            return RDKit::StereoGroupType::STEREO_OR;
        default:
            throw std::invalid_argument("stereo group kind is invalid");
    }
}


struct GroupKey {
    std::string scope;
    std::uint32_t kind;
    std::int64_t id;

    auto operator<=>(const GroupKey&) const = default;
};


struct GroupMembers {
    std::vector<RDKit::Atom*> atoms;
    std::vector<RDKit::Bond*> bonds;
};


void add_group_atom(GroupMembers& members, RDKit::Atom* atom) {
    if (std::find(members.atoms.begin(), members.atoms.end(), atom)
        == members.atoms.end()) {
        members.atoms.push_back(atom);
    }
}


void add_group_bond(GroupMembers& members, RDKit::Bond* bond) {
    if (std::find(members.bonds.begin(), members.bonds.end(), bond)
        == members.bonds.end()) {
        members.bonds.push_back(bond);
    }
}


RDKit::Bond* find_bond(
    RDKit::RWMol& molecule,
    std::uint64_t first,
    std::uint64_t second,
    const char* context
) {
    auto* bond = molecule.getBondBetweenAtoms(
        static_cast<unsigned int>(first),
        static_cast<unsigned int>(second)
    );
    if (bond == nullptr) {
        throw UnsupportedInput(std::string(context) + " references a missing bond");
    }
    return bond;
}


void apply_atom_transport(
    RDKit::RWMol& molecule,
    const hotpot_rd_molecule_view_v1& source
) {
    for (std::uint64_t index = 0; index < source.atom_transport_count; ++index) {
        const auto& record = source.atom_transport[index];
        const auto source_namespace = copy_string(record.source_namespace);
        const bool is_rdkit_transport = source_namespace == "rdkit-2026.09.1";
        const bool is_hotpot_legacy = source_namespace == "hotpot-legacy/1";
        if (!is_rdkit_transport && !is_hotpot_legacy) {
            throw UnsupportedInput(
                "atom transport source_namespace is not representable by "
                "the pinned RDKit adapter"
            );
        }
        auto* atom = molecule.getAtomWithIdx(static_cast<unsigned int>(record.atom));
        if (is_hotpot_legacy) {
            if (record.presence_mask != 0
                || !record.has_current_hydrogens
                || record.has_source_hydrogens
                || record.current_hydrogens.presence_mask
                    != HOTPOT_RD_HYDROGEN_TRANSPORT_IMPLICIT) {
                throw UnsupportedInput(
                    "hotpot-legacy/1 atom transport must contain only the "
                    "logical implicit-hydrogen count"
                );
            }
            atom->setNumExplicitHs(static_cast<unsigned int>(
                record.current_hydrogens.implicit_hydrogens
            ));
            continue;
        }
        if ((record.presence_mask & HOTPOT_RD_ATOM_TRANSPORT_ISOTOPE) != 0) {
            if (record.isotope < 0) {
                throw UnsupportedInput("RDKit isotope cannot be negative");
            }
            atom->setIsotope(static_cast<unsigned int>(record.isotope));
        }
        if ((record.presence_mask & HOTPOT_RD_ATOM_TRANSPORT_RADICAL_ELECTRONS) != 0) {
            if (record.radical_electrons < 0) {
                throw UnsupportedInput("RDKit radical electron count cannot be negative");
            }
            atom->setNumRadicalElectrons(
                static_cast<unsigned int>(record.radical_electrons)
            );
        }
        if (!record.has_current_hydrogens) {
            continue;
        }
        const auto& hydrogen = record.current_hydrogens;
        if ((hydrogen.presence_mask
             & HOTPOT_RD_HYDROGEN_TRANSPORT_BRACKET_EXPLICIT) != 0) {
            atom->setNumExplicitHs(
                static_cast<unsigned int>(hydrogen.bracket_explicit_hydrogens)
            );
        }
        if ((hydrogen.presence_mask & HOTPOT_RD_HYDROGEN_TRANSPORT_NO_IMPLICIT) != 0) {
            atom->setNoImplicit(hydrogen.no_implicit != 0);
        }
    }
}


void validate_implicit_hydrogens(
    const RDKit::RWMol& molecule,
    const hotpot_rd_molecule_view_v1& source
) {
    for (std::uint64_t index = 0; index < source.atom_transport_count; ++index) {
        const auto& record = source.atom_transport[index];
        if (copy_string(record.source_namespace) == "hotpot-legacy/1") {
            continue;
        }
        if (!record.has_current_hydrogens
            || (record.current_hydrogens.presence_mask
                & HOTPOT_RD_HYDROGEN_TRANSPORT_IMPLICIT) == 0) {
            continue;
        }
        const auto* atom = molecule.getAtomWithIdx(
            static_cast<unsigned int>(record.atom)
        );
        if (atom->getNumImplicitHs()
            != static_cast<unsigned int>(
                record.current_hydrogens.implicit_hydrogens
            )) {
            throw UnsupportedInput(
                "current implicit-hydrogen state is not representable by RDKit"
            );
        }
    }
}


void apply_bond_transport(
    RDKit::RWMol& molecule,
    const hotpot_rd_molecule_view_v1& source
) {
    for (std::uint64_t index = 0; index < source.bond_transport_count; ++index) {
        const auto& record = source.bond_transport[index];
        if (record.display_direction.data == nullptr) {
            continue;
        }
        auto* bond = find_bond(
            molecule,
            record.first_atom,
            record.second_atom,
            "bond transport"
        );
        auto direction = bond_direction(copy_string(record.display_direction));
        if (bond->getBeginAtomIdx() != record.first_atom) {
            switch (direction) {
                case RDKit::Bond::NONE:
                case RDKit::Bond::EITHERDOUBLE:
                    break;
                case RDKit::Bond::ENDDOWNRIGHT:
                    direction = RDKit::Bond::ENDUPRIGHT;
                    break;
                case RDKit::Bond::ENDUPRIGHT:
                    direction = RDKit::Bond::ENDDOWNRIGHT;
                    break;
                default:
                    throw UnsupportedInput(
                        "reversed RDKit bond display direction has no frozen "
                        "lossless representation"
                    );
            }
        }
        bond->setBondDir(direction);
    }
}


void apply_stereo(
    RDKit::RWMol& molecule,
    const hotpot_rd_molecule_view_v1& source
) {
    std::map<GroupKey, GroupMembers> groups;
    std::vector<GroupKey> group_order;
    std::set<std::tuple<std::uint32_t, std::uint64_t, std::uint64_t>> anchors;

    for (std::uint64_t index = 0; index < source.stereo_record_count; ++index) {
        const auto& record = source.stereo_records[index];
        const auto source_namespace = copy_string(record.source_namespace);
        if (source_namespace != "rdkit-2026.09.1") {
            throw UnsupportedInput(
                "stereo source_namespace is not representable by the RDKit adapter: "
                + source_namespace
            );
        }
        const auto source_code = copy_string(record.source_code);
        RDKit::Atom* anchor_atom = nullptr;
        RDKit::Bond* anchor_bond = nullptr;
        if (record.anchor_kind == HOTPOT_RD_STEREO_ANCHOR_ATOM) {
            const auto atom_index =
                source.stereo_anchor_indices[record.anchor_offset];
            if (!anchors.emplace(record.anchor_kind, atom_index, atom_index).second) {
                throw UnsupportedInput("duplicate atom stereo anchor");
            }
            anchor_atom = molecule.getAtomWithIdx(
                static_cast<unsigned int>(atom_index)
            );
            apply_atom_stereo(
                molecule,
                *anchor_atom,
                source,
                record,
                source_code
            );
        } else {
            const auto first = source.stereo_anchor_indices[record.anchor_offset];
            const auto second =
                source.stereo_anchor_indices[record.anchor_offset + 1];
            const auto low = std::min(first, second);
            const auto high = std::max(first, second);
            if (!anchors.emplace(record.anchor_kind, low, high).second) {
                throw UnsupportedInput("duplicate bond stereo anchor");
            }
            anchor_bond = find_bond(molecule, first, second, "stereo record");
            if (anchor_bond->getBeginAtomIdx() != first
                && (source_code == "STEREOATROPCW"
                    || source_code == "STEREOATROPCCW")) {
                throw UnsupportedInput(
                    "RDKit atropisomer stereo has no frozen reversed-anchor "
                    "transformation"
                );
            }

            std::vector<unsigned int> carrier_atoms;
            for (std::uint64_t carrier_index = 0;
                 carrier_index < record.carrier_count;
                 ++carrier_index) {
                const auto& carrier = source.stereo_carriers[
                    record.carrier_offset + carrier_index
                ];
                if (carrier.kind != HOTPOT_RD_STEREO_CARRIER_ATOM) {
                    throw UnsupportedInput(
                        "RDKit bond stereo requires atom carriers"
                    );
                }
                carrier_atoms.push_back(static_cast<unsigned int>(carrier.atom));
            }
            if (!carrier_atoms.empty()) {
                if (carrier_atoms.size() != 2) {
                    throw UnsupportedInput(
                        "RDKit bond stereo requires exactly two atom carriers"
                    );
                }
                const auto anchor_first = static_cast<unsigned int>(first);
                if (anchor_bond->getBeginAtomIdx() != anchor_first) {
                    std::swap(carrier_atoms[0], carrier_atoms[1]);
                }
                try {
                    anchor_bond->setStereoAtoms(
                        carrier_atoms[0],
                        carrier_atoms[1]
                    );
                } catch (const std::bad_alloc&) {
                    throw;
                } catch (const std::exception& error) {
                    throw UnsupportedInput(
                        std::string("RDKit rejected bond stereo carriers: ")
                        + error.what()
                    );
                }
            }
            try {
                anchor_bond->setStereo(bond_stereo_type(source_code));
            } catch (const std::bad_alloc&) {
                throw;
            } catch (const std::exception& error) {
                throw UnsupportedInput(
                    std::string("RDKit rejected bond stereo: ") + error.what()
                );
            }
        }

        if (!record.has_group) {
            continue;
        }
        if (record.group_kind == HOTPOT_RD_STEREO_GROUP_NONE
            || record.group_id < 0
            || static_cast<std::uint64_t>(record.group_id)
                > std::numeric_limits<unsigned int>::max()) {
            throw UnsupportedInput("stereo group identity is not representable by RDKit");
        }
        GroupKey key{
            copy_string(record.group_scope),
            record.group_kind,
            record.group_id,
        };
        if (!groups.contains(key)) {
            groups.emplace(key, GroupMembers{});
            group_order.push_back(key);
        }
        auto& members = groups.at(key);
        if (anchor_atom != nullptr) add_group_atom(members, anchor_atom);
        if (anchor_bond != nullptr) add_group_bond(members, anchor_bond);
    }

    std::vector<RDKit::StereoGroup> native_groups;
    native_groups.reserve(group_order.size());
    try {
        for (const auto& key : group_order) {
            auto& members = groups.at(key);
            native_groups.emplace_back(
                group_type(key.kind),
                std::move(members.atoms),
                std::move(members.bonds),
                static_cast<unsigned int>(key.id)
            );
        }
        molecule.setStereoGroups(std::move(native_groups));
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& error) {
        throw UnsupportedInput(
            std::string("RDKit rejected an enhanced stereo group: ")
            + error.what()
        );
    }
}


}  // namespace


std::unique_ptr<RDKit::RWMol> build_molecule(
    const hotpot_rd_molecule_view_v1& source
) {
    auto molecule = std::make_unique<RDKit::RWMol>();
    molecule->beginBatchEdit();
    for (std::uint64_t index = 0; index < source.atom_count; ++index) {
        auto* atom = new RDKit::Atom(
            static_cast<unsigned int>(source.atomic_numbers[index])
        );
        atom->setFormalCharge(source.formal_charges[index]);
        atom->setIsAromatic(source.atom_aromatic[index] != 0);
        molecule->addAtom(atom, false, true);
    }
    for (std::uint64_t index = 0; index < source.bond_count; ++index) {
        const bool aromatic_kind =
            source.bond_kinds[index] == HOTPOT_RD_BOND_AROMATIC;
        const bool aromatic_flag = source.bond_aromatic[index] != 0;
        if (aromatic_kind != aromatic_flag) {
            throw UnsupportedInput(
                "bond aromatic flag and bond kind disagree"
            );
        }
        if ((source.field_mask & HOTPOT_RD_FIELD_BOND_ORDERS) != 0
            && std::abs(
                source.bond_orders[index]
                - expected_bond_order(source.bond_kinds[index])
            ) > 1e-12) {
            throw UnsupportedInput("bond order and bond kind disagree");
        }
        const auto first = source.bond_indices[2 * index];
        const auto second = source.bond_indices[2 * index + 1];
        molecule->addBond(
            static_cast<unsigned int>(first),
            static_cast<unsigned int>(second),
            bond_type(source.bond_kinds[index])
        );
        auto* bond = molecule->getBondBetweenAtoms(
            static_cast<unsigned int>(first),
            static_cast<unsigned int>(second)
        );
        bond->setIsAromatic(aromatic_flag);
    }
    molecule->commitBatchEdit();

    apply_atom_transport(*molecule, source);
    apply_bond_transport(*molecule, source);

    try {
        constexpr unsigned int sanitize_operations =
            RDKit::MolOps::SANITIZE_PROPERTIES
            | RDKit::MolOps::SANITIZE_SYMMRINGS
            | RDKit::MolOps::SANITIZE_KEKULIZE
            | RDKit::MolOps::SANITIZE_SETAROMATICITY
            | RDKit::MolOps::SANITIZE_SETCONJUGATION
            | RDKit::MolOps::SANITIZE_SETHYBRIDIZATION
            | RDKit::MolOps::SANITIZE_ADJUSTHS;
        unsigned int failed_operation = RDKit::MolOps::SANITIZE_NONE;
        RDKit::MolOps::sanitizeMol(
            *molecule,
            failed_operation,
            sanitize_operations
        );
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& error) {
        throw UnsupportedInput(
            std::string("RDKit sanitization rejected the Hotpot graph: ")
            + error.what()
        );
    }
    validate_implicit_hydrogens(*molecule, source);
    apply_stereo(*molecule, source);
    return molecule;
}


}  // namespace hotpot::rdwrappers::core
