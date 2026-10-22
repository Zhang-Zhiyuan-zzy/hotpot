#include <hotpot/rdWrappers/c_api.h>

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


static hotpot_rd_options_v1 options(void) {
    hotpot_rd_options_v1 value = {0};
    value.struct_size = sizeof(value);
    value.abi_version = HOTPOT_RD_C_API_VERSION;
    value.conformer_count = 2;
    value.seed = 0x5eed;
    value.has_seed = 1;
    return value;
}


static hotpot_rd_molecule_view_v1 methane(void) {
    static const int32_t atomic_numbers[] = {6, 1, 1, 1, 1};
    static const int32_t formal_charges[] = {0, 0, 0, 0, 0};
    static const uint8_t atom_aromatic[] = {0, 0, 0, 0, 0};
    static const int32_t bond_indices[] = {0, 1, 0, 2, 0, 3, 0, 4};
    static const uint8_t bond_kinds[] = {1, 1, 1, 1};
    static const uint8_t bond_aromatic[] = {0, 0, 0, 0};
    hotpot_rd_molecule_view_v1 value = {0};
    value.struct_size = sizeof(value);
    value.abi_version = HOTPOT_RD_C_API_VERSION;
    value.schema_version = 1;
    value.atom_count = 5;
    value.bond_count = 4;
    value.field_mask = HOTPOT_RD_FIELD_ATOMIC_NUMBERS
        | HOTPOT_RD_FIELD_FORMAL_CHARGES
        | HOTPOT_RD_FIELD_ATOM_AROMATIC
        | HOTPOT_RD_FIELD_BOND_INDICES
        | HOTPOT_RD_FIELD_BOND_KINDS
        | HOTPOT_RD_FIELD_BOND_AROMATIC;
    value.atomic_numbers = atomic_numbers;
    value.formal_charges = formal_charges;
    value.atom_aromatic = atom_aromatic;
    value.bond_indices = bond_indices;
    value.bond_kinds = bond_kinds;
    value.bond_aromatic = bond_aromatic;
    return value;
}


int main(void) {
    assert(hotpot_rd_runtime_abi_version() == HOTPOT_RD_C_API_VERSION);
    hotpot_rd_string_view_v1 version = hotpot_rd_runtime_rdkit_version();
    assert(version.size == strlen("2026.09.1"));
    assert(memcmp(version.data, "2026.09.1", version.size) == 0);
    hotpot_rd_string_view_v1 hotpot_version =
        hotpot_rd_runtime_hotpot_version();
    assert(hotpot_version.size == strlen("0.5.4.0"));
    assert(memcmp(hotpot_version.data, "0.5.4.0", hotpot_version.size) == 0);
    hotpot_rd_string_view_v1 commit = hotpot_rd_runtime_rdkit_commit();
    assert(commit.size == strlen("fece8caa860bdf6c9c82bdb9253f22a0cf87150c"));
    assert(memcmp(
        commit.data,
        "fece8caa860bdf6c9c82bdb9253f22a0cf87150c",
        commit.size
    ) == 0);

    hotpot_rd_options_v1 invalid = options();
    invalid.abi_version += 1;
    hotpot_rd_sampler* rejected = (hotpot_rd_sampler*)(uintptr_t)1;
    assert(hotpot_rd_sampler_create(&invalid, &rejected)
           == HOTPOT_RD_STATUS_UNSUPPORTED_ABI);
    assert(rejected == NULL);

    hotpot_rd_options_v1 config = options();
    hotpot_rd_sampler* sampler = NULL;
    assert(hotpot_rd_sampler_create(&config, &sampler) == HOTPOT_RD_STATUS_OK);
    assert(sampler != NULL);

    hotpot_rd_molecule_view_v1 molecule = methane();
    const int32_t atom_snapshot[5] = {6, 1, 1, 1, 1};
    hotpot_rd_result* first = NULL;
    hotpot_rd_result* second = NULL;
    assert(hotpot_rd_generate(sampler, &molecule, NULL, &first)
           == HOTPOT_RD_STATUS_OK);
    assert(hotpot_rd_generate(sampler, &molecule, NULL, &second)
           == HOTPOT_RD_STATUS_OK);
    assert(memcmp(molecule.atomic_numbers, atom_snapshot, sizeof(atom_snapshot)) == 0);
    assert(hotpot_rd_result_status(first) == HOTPOT_RD_GENERATION_SUCCESS);
    assert(hotpot_rd_result_status(second) == HOTPOT_RD_GENERATION_SUCCESS);
    hotpot_rd_string_view_v1 engine = hotpot_rd_result_engine(first);
    assert(engine.size == strlen("RDKit ETKDGv3"));
    assert(memcmp(engine.data, "RDKit ETKDGv3", engine.size) == 0);

    uint64_t first_count = 0;
    uint64_t first_atoms = 0;
    uint64_t second_count = 0;
    uint64_t second_atoms = 0;
    const double* first_coordinates = hotpot_rd_result_coordinates(
        first, &first_count, &first_atoms
    );
    const double* second_coordinates = hotpot_rd_result_coordinates(
        second, &second_count, &second_atoms
    );
    assert(first_count == 2 && second_count == 2);
    assert(first_atoms == 5 && second_atoms == 5);
    assert(memcmp(
        first_coordinates,
        second_coordinates,
        (size_t)(first_count * first_atoms * 3) * sizeof(double)
    ) == 0);
    for (uint64_t index = 0; index < first_count * first_atoms * 3; ++index) {
        assert(isfinite(first_coordinates[index]));
    }

    uint64_t record_count = 0;
    const hotpot_rd_conformer_record_v1* records =
        hotpot_rd_result_records(first, &record_count);
    assert(record_count == 2);
    assert(records[0].ordinal == 0 && records[1].ordinal == 1);
    assert(hotpot_rd_result_diagnostic(first).data == NULL);

    static const uint8_t unknown_kinds[] = {
        HOTPOT_RD_BOND_UNKNOWN,
        HOTPOT_RD_BOND_SINGLE,
        HOTPOT_RD_BOND_SINGLE,
        HOTPOT_RD_BOND_SINGLE,
    };
    hotpot_rd_molecule_view_v1 unsupported = methane();
    unsupported.bond_kinds = unknown_kinds;
    hotpot_rd_result* unsupported_result = NULL;
    assert(hotpot_rd_generate(
        sampler, &unsupported, NULL, &unsupported_result
    ) == HOTPOT_RD_STATUS_OK);
    assert(hotpot_rd_result_status(unsupported_result)
           == HOTPOT_RD_GENERATION_UNSUPPORTED);
    const hotpot_rd_string_view_v1 diagnostic =
        hotpot_rd_result_diagnostic(unsupported_result);
    assert(diagnostic.data != NULL && diagnostic.size != 0);

    static const int32_t coordination_pair[] = {0, 1};
    static const double coordination_bound[] = {1.08, 1.10};
    hotpot_rd_constraints_view_v1 constraints = {0};
    constraints.struct_size = sizeof(constraints);
    constraints.abi_version = HOTPOT_RD_C_API_VERSION;
    constraints.coordination_pair_count = 1;
    constraints.coordination_pairs = coordination_pair;
    constraints.metal_donor_bounds = coordination_bound;
    hotpot_rd_result* constrained_result = NULL;
    assert(hotpot_rd_generate(
        sampler, &molecule, &constraints, &constrained_result
    ) == HOTPOT_RD_STATUS_OK);
    assert(hotpot_rd_result_status(constrained_result)
           <= HOTPOT_RD_GENERATION_PARTIAL);

    hotpot_rd_result_destroy(first);
    hotpot_rd_result_destroy(second);
    hotpot_rd_result_destroy(unsupported_result);
    hotpot_rd_result_destroy(constrained_result);
    hotpot_rd_sampler_destroy(sampler);
    return 0;
}
