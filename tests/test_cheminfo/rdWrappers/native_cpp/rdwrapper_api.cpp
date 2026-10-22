#include <hotpot/rdWrappers/rdwrapper.hpp>

#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


struct hotpot_rd_sampler {
    hotpot_rd_options_v1 options;
};


struct hotpot_rd_result {
    std::vector<double> coordinates;
    std::vector<hotpot_rd_conformer_record_v1> records;
};


namespace {


hotpot_rd_string_view_v1 string_view(const char* value, std::uint64_t size) {
    return hotpot_rd_string_view_v1{value, size};
}


}  // namespace


extern "C" {


const char* hotpot_rd_status_message(hotpot_rd_status status) {
    return status == HOTPOT_RD_STATUS_OK ? "ok" : "mock error";
}


hotpot_rd_status hotpot_rd_sampler_create(
    const hotpot_rd_options_v1* options,
    hotpot_rd_sampler** out_sampler
) {
    if (options == nullptr || out_sampler == nullptr
        || options->abi_version != HOTPOT_RD_C_API_VERSION
        || options->struct_size != sizeof(hotpot_rd_options_v1)) {
        return HOTPOT_RD_STATUS_INVALID_ARGUMENT;
    }
    *out_sampler = new hotpot_rd_sampler{*options};
    return HOTPOT_RD_STATUS_OK;
}


void hotpot_rd_sampler_destroy(hotpot_rd_sampler* sampler) {
    delete sampler;
}


hotpot_rd_status hotpot_rd_generate(
    hotpot_rd_sampler* sampler,
    const hotpot_rd_molecule_view_v1* molecule,
    const hotpot_rd_constraints_view_v1* constraints,
    hotpot_rd_result** out_result
) {
    if (sampler == nullptr || molecule == nullptr || out_result == nullptr
        || molecule->abi_version != HOTPOT_RD_C_API_VERSION
        || molecule->atom_count != 2
        || molecule->atomic_numbers[0] != 6
        || molecule->bond_kinds[0] != HOTPOT_RD_BOND_DATIVE
        || constraints != nullptr) {
        return HOTPOT_RD_STATUS_INVALID_ARGUMENT;
    }
    assert((molecule->field_mask & HOTPOT_RD_FIELD_STEREO_RECORDS) != 0);
    assert((molecule->field_mask & HOTPOT_RD_FIELD_ATOM_TRANSPORT) != 0);
    assert((molecule->field_mask & HOTPOT_RD_FIELD_BOND_TRANSPORT) != 0);
    assert(molecule->stereo_record_count == 1);
    assert(molecule->stereo_anchor_index_count == 2);
    assert(molecule->stereo_carrier_count == 2);
    assert(molecule->stereo_records[0].anchor_kind
           == HOTPOT_RD_STEREO_ANCHOR_BOND);
    assert(molecule->stereo_anchor_indices[0] == 0);
    assert(molecule->stereo_anchor_indices[1] == 1);
    assert(molecule->stereo_carriers[0].kind
           == HOTPOT_RD_STEREO_CARRIER_ATOM);
    assert(molecule->stereo_carriers[0].atom == 1);
    assert(molecule->stereo_records[0].group_scope.data == nullptr);
    assert(molecule->stereo_records[0].descriptor.data != nullptr);
    assert(molecule->stereo_records[0].descriptor.size == 0);
    assert(molecule->stereo_records[0].source_namespace.size
           == sizeof("rdkit-2026.09.1") - 1);
    assert(std::string(
               molecule->stereo_records[0].source_namespace.data,
               static_cast<std::size_t>(
                   molecule->stereo_records[0].source_namespace.size
               )
           ) == "rdkit-2026.09.1");
    assert(molecule->atom_transport_count == 1);
    assert(molecule->atom_transport[0].atom == 0);
    assert(molecule->atom_transport[0].isotope == 13);
    assert(molecule->atom_transport[0].has_current_hydrogens == 1);
    assert(molecule->atom_transport[0].current_hydrogens.no_implicit == 1);
    assert(molecule->atom_transport[0].has_source_hydrogens == 1);
    assert(molecule->atom_transport[0]
               .source_hydrogens.bracket_explicit_hydrogens == 2);
    assert(molecule->atom_transport[0].source_namespace.data == nullptr);
    assert(molecule->bond_transport_count == 1);
    assert(molecule->bond_transport[0].first_atom == 1);
    assert(molecule->bond_transport[0].second_atom == 0);
    assert(molecule->bond_transport[0].display_direction.data != nullptr);
    assert(molecule->bond_transport[0].display_direction.size == 0);
    auto* result = new hotpot_rd_result;
    const auto conformer_count = sampler->options.conformer_count;
    result->coordinates.reserve(
        static_cast<std::size_t>(conformer_count * molecule->atom_count * 3)
    );
    for (std::uint64_t conformer = 0; conformer < conformer_count; ++conformer) {
        result->coordinates.insert(
            result->coordinates.end(),
            {
                static_cast<double>(conformer), 0.0, 0.0,
                static_cast<double>(conformer), 1.0, 0.0,
            }
        );
        result->records.push_back({conformer, static_cast<std::int64_t>(10 + conformer)});
    }
    *out_result = result;
    return HOTPOT_RD_STATUS_OK;
}


hotpot_rd_generation_status hotpot_rd_result_status(
    const hotpot_rd_result*
) {
    return HOTPOT_RD_GENERATION_SUCCESS;
}


const double* hotpot_rd_result_coordinates(
    const hotpot_rd_result* result,
    std::uint64_t* conformer_count,
    std::uint64_t* atom_count
) {
    *conformer_count = result->records.size();
    *atom_count = 2;
    return result->coordinates.data();
}


const hotpot_rd_conformer_record_v1* hotpot_rd_result_records(
    const hotpot_rd_result* result,
    std::uint64_t* record_count
) {
    *record_count = result->records.size();
    return result->records.data();
}


hotpot_rd_string_view_v1 hotpot_rd_result_engine(const hotpot_rd_result*) {
    return string_view("ETKDGv3", 7);
}


hotpot_rd_string_view_v1 hotpot_rd_result_engine_version(
    const hotpot_rd_result*
) {
    return string_view("2026.09.1", sizeof("2026.09.1") - 1);
}


hotpot_rd_string_view_v1 hotpot_rd_result_diagnostic(
    const hotpot_rd_result*
) {
    return hotpot_rd_string_view_v1{nullptr, 0};
}


void hotpot_rd_result_destroy(hotpot_rd_result* result) {
    delete result;
}


std::uint32_t hotpot_rd_runtime_abi_version(void) {
    return HOTPOT_RD_C_API_VERSION;
}


hotpot_rd_string_view_v1 hotpot_rd_runtime_hotpot_version(void) {
    return string_view("0.5.4.0", 7);
}


hotpot_rd_string_view_v1 hotpot_rd_runtime_rdkit_version(void) {
    return string_view("2026.09.1", sizeof("2026.09.1") - 1);
}


hotpot_rd_string_view_v1 hotpot_rd_runtime_rdkit_commit(void) {
    return string_view("fece8caa", 8);
}


}  // extern "C"


int main() {
    namespace native = hotpot::native;
    namespace rd = hotpot::rdwrappers;

    static_assert(std::is_standard_layout_v<hotpot_rd_options_v1>);
    static_assert(std::is_standard_layout_v<hotpot_rd_molecule_view_v1>);
    static_assert(std::is_standard_layout_v<hotpot_rd_stereo_record_v1>);
    static_assert(std::is_standard_layout_v<hotpot_rd_hydrogen_state_v1>);
    static_assert(std::is_standard_layout_v<hotpot_rd_atom_transport_v1>);
    static_assert(!std::is_copy_constructible_v<rd::ETKDGSampler>);
    static_assert(std::is_move_constructible_v<rd::ETKDGSampler>);
    static_assert(
        native::molecule_field_mask(native::MoleculeField::BOND_KINDS)
        == HOTPOT_RD_FIELD_BOND_KINDS
    );
    static_assert(
        native::molecule_field_mask(native::MoleculeField::STEREO_RECORDS)
        == HOTPOT_RD_FIELD_STEREO_RECORDS
    );
    static_assert(
        native::molecule_field_mask(native::MoleculeField::ATOM_TRANSPORT)
        == HOTPOT_RD_FIELD_ATOM_TRANSPORT
    );
    static_assert(
        native::molecule_field_mask(native::MoleculeField::BOND_TRANSPORT)
        == HOTPOT_RD_FIELD_BOND_TRANSPORT
    );

    native::OwnedMoleculeData absent_transport;
    const auto absent_holder = rd::detail::to_c_molecule_view(
        absent_transport.view()
    );
    const auto absent_view = absent_holder.view();
    assert(absent_view.stereo_records == nullptr);
    assert(absent_view.stereo_record_count == 0);
    assert(absent_view.stereo_anchor_indices == nullptr);
    assert(absent_view.stereo_carriers == nullptr);
    assert(absent_view.atom_transport == nullptr);
    assert(absent_view.bond_transport == nullptr);

    native::OwnedMoleculeData molecule;
    molecule.atom_count = 2;
    molecule.bond_count = 1;
    molecule.fields =
        native::molecule_field_mask(native::MoleculeField::ATOMIC_NUMBERS)
        | native::molecule_field_mask(native::MoleculeField::BOND_INDICES)
        | native::molecule_field_mask(native::MoleculeField::BOND_KINDS)
        | native::molecule_field_mask(native::MoleculeField::STEREO_RECORDS)
        | native::molecule_field_mask(native::MoleculeField::ATOM_TRANSPORT)
        | native::molecule_field_mask(native::MoleculeField::BOND_TRANSPORT);
    molecule.atomic_numbers = {6, 7};
    molecule.bond_indices = {{{0, 1}}};
    molecule.bond_kinds = {native::BondKind::DATIVE};

    auto& stereo = molecule.stereo_records;
    stereo.anchor_kinds = {
        static_cast<std::uint32_t>(native::StereoAnchorKind::BOND),
    };
    stereo.anchor_offsets = {0};
    stereo.anchor_counts = {2};
    stereo.anchor_indices = {0, 1};
    stereo.carrier_offsets = {0};
    stereo.carrier_counts = {2};
    stereo.carrier_kinds = {
        static_cast<std::uint32_t>(native::StereoCarrierKind::ATOM),
        static_cast<std::uint32_t>(native::StereoCarrierKind::ATOM),
    };
    stereo.carrier_has_atoms = {1, 1};
    stereo.carrier_atoms = {1, 0};
    stereo.group_kinds = {
        static_cast<std::uint32_t>(native::StereoGroupKind::NONE),
    };
    stereo.group_ids = {0};
    stereo.group_present = {0};
    stereo.group_scopes.data = {0};
    stereo.group_scopes.offsets = {0};
    stereo.group_scopes.lengths = {0};
    stereo.group_scopes.present = {0};
    stereo.descriptors.data = {0};
    stereo.descriptors.offsets = {0};
    stereo.descriptors.lengths = {0};
    stereo.descriptors.present = {1};
    const std::string source_namespace = "rdkit-2026.09.1";
    stereo.source_namespaces.data.assign(
        source_namespace.begin(), source_namespace.end()
    );
    stereo.source_namespaces.offsets = {0};
    stereo.source_namespaces.lengths = {source_namespace.size()};
    stereo.source_namespaces.present = {1};
    const std::string source_code = "STEREONONE";
    stereo.source_codes.data.assign(source_code.begin(), source_code.end());
    stereo.source_codes.offsets = {0};
    stereo.source_codes.lengths = {source_code.size()};
    stereo.source_codes.present = {1};

    auto& atoms = molecule.atom_transport;
    atoms.atoms = {0};
    atoms.presence_masks = {
        static_cast<std::uint32_t>(native::AtomTransportField::ISOTOPE),
    };
    atoms.isotopes = {13};
    atoms.radical_electrons = {0};
    atoms.has_current_hydrogens = {1};
    atoms.current_hydrogen_presence_masks = {
        static_cast<std::uint32_t>(
            native::HydrogenTransportField::NO_IMPLICIT
        ),
    };
    atoms.current_implicit_hydrogens = {0};
    atoms.current_bracket_explicit_hydrogens = {0};
    atoms.current_no_implicit = {1};
    atoms.has_source_hydrogens = {1};
    atoms.source_hydrogen_presence_masks = {
        static_cast<std::uint32_t>(
            native::HydrogenTransportField::BRACKET_EXPLICIT
        ),
    };
    atoms.source_implicit_hydrogens = {0};
    atoms.source_bracket_explicit_hydrogens = {2};
    atoms.source_no_implicit = {0};
    atoms.source_namespaces.data = {0};
    atoms.source_namespaces.offsets = {0};
    atoms.source_namespaces.lengths = {0};
    atoms.source_namespaces.present = {0};

    auto& bonds = molecule.bond_transport;
    bonds.atom_indices = {{{1, 0}}};
    bonds.display_directions.data = {0};
    bonds.display_directions.offsets = {0};
    bonds.display_directions.lengths = {0};
    bonds.display_directions.present = {1};

    const std::array<native::BondIndex, 1> constraint_pairs{{{0, 1}}};
    const std::array<rd::DistanceBounds, 0> missing_bounds{};
    bool mismatched_constraints_rejected = false;
    try {
        (void)rd::detail::to_c_constraints_view(
            rd::CoordinationEmbeddingConstraintsView{
                native::ArrayView<native::BondIndex>(constraint_pairs),
                native::ArrayView<rd::DistanceBounds>(missing_bounds),
                {},
                {},
            }
        );
    } catch (const std::invalid_argument&) {
        mismatched_constraints_rejected = true;
    }
    assert(mismatched_constraints_rejected);

    rd::ETKDGOptions options;
    options.conformer_count = 2;
    options.seed = 20261022;
    rd::ETKDGSampler sampler(options);
    const auto result = sampler.generate(molecule.view());
    assert(result.status == rd::GenerationStatus::SUCCESS);
    assert(result.conformer_count == 2);
    assert(result.atom_count == 2);
    assert(result.coordinates.size() == 4);
    assert(result.coordinates[2][0] == 1.0);
    assert(result.records[1].ordinal == 1);
    assert(result.records[1].engine_conformer_id == 11);
    assert(result.engine == "ETKDGv3");
    assert(result.engine_version == "2026.09.1");
    assert(!result.diagnostic.has_value());

    const auto runtime = rd::runtime_info();
    assert(runtime.hotpot_version == "0.5.4.0");
    assert(runtime.rdkit_version == "2026.09.1");
    assert(runtime.native);
}
