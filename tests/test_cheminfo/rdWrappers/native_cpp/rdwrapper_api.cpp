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
    static_assert(!std::is_copy_constructible_v<rd::ETKDGSampler>);
    static_assert(std::is_move_constructible_v<rd::ETKDGSampler>);
    static_assert(
        native::molecule_field_mask(native::MoleculeField::BOND_KINDS)
        == HOTPOT_RD_FIELD_BOND_KINDS
    );

    native::OwnedMoleculeData molecule;
    molecule.atom_count = 2;
    molecule.bond_count = 1;
    molecule.fields =
        native::molecule_field_mask(native::MoleculeField::ATOMIC_NUMBERS)
        | native::molecule_field_mask(native::MoleculeField::BOND_INDICES)
        | native::molecule_field_mask(native::MoleculeField::BOND_KINDS);
    molecule.atomic_numbers = {6, 7};
    molecule.bond_indices = {{{0, 1}}};
    molecule.bond_kinds = {native::BondKind::DATIVE};

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
