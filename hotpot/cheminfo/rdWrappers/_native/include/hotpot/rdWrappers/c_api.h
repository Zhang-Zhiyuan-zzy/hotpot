#ifndef HOTPOT_RDWRAPPERS_C_API_H
#define HOTPOT_RDWRAPPERS_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(HOTPOT_RDWRAPPERS_BUILDING_LIBRARY)
#define HOTPOT_RD_API __declspec(dllexport)
#elif defined(_WIN32)
#define HOTPOT_RD_API __declspec(dllimport)
#elif defined(__GNUC__)
#define HOTPOT_RD_API __attribute__((visibility("default")))
#else
#define HOTPOT_RD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define HOTPOT_RD_C_API_VERSION 1u

typedef enum hotpot_rd_status {
    HOTPOT_RD_STATUS_OK = 0,
    HOTPOT_RD_STATUS_INVALID_ARGUMENT = 1,
    HOTPOT_RD_STATUS_UNSUPPORTED_ABI = 2,
    HOTPOT_RD_STATUS_OUT_OF_MEMORY = 3,
    HOTPOT_RD_STATUS_INTERNAL_ERROR = 4
} hotpot_rd_status;

typedef enum hotpot_rd_generation_status {
    HOTPOT_RD_GENERATION_SUCCESS = 0,
    HOTPOT_RD_GENERATION_PARTIAL = 1,
    HOTPOT_RD_GENERATION_UNSUPPORTED = 2,
    HOTPOT_RD_GENERATION_FAILED = 3
} hotpot_rd_generation_status;

typedef enum hotpot_rd_bond_kind {
    HOTPOT_RD_BOND_SINGLE = 1,
    HOTPOT_RD_BOND_DOUBLE = 2,
    HOTPOT_RD_BOND_TRIPLE = 3,
    HOTPOT_RD_BOND_AROMATIC = 4,
    HOTPOT_RD_BOND_ZERO = 5,
    HOTPOT_RD_BOND_DATIVE = 6,
    HOTPOT_RD_BOND_UNKNOWN = 7
} hotpot_rd_bond_kind;

typedef enum hotpot_rd_molecule_field {
    HOTPOT_RD_FIELD_ATOMIC_NUMBERS = UINT64_C(1) << 0,
    HOTPOT_RD_FIELD_FORMAL_CHARGES = UINT64_C(1) << 1,
    HOTPOT_RD_FIELD_PARTIAL_CHARGES = UINT64_C(1) << 2,
    HOTPOT_RD_FIELD_COORDINATES = UINT64_C(1) << 3,
    HOTPOT_RD_FIELD_ATOM_AROMATIC = UINT64_C(1) << 4,
    HOTPOT_RD_FIELD_BOND_INDICES = UINT64_C(1) << 5,
    HOTPOT_RD_FIELD_BOND_ORDERS = UINT64_C(1) << 6,
    HOTPOT_RD_FIELD_BOND_KINDS = UINT64_C(1) << 7,
    HOTPOT_RD_FIELD_BOND_AROMATIC = UINT64_C(1) << 8,
    HOTPOT_RD_FIELD_UNIT_CELL = UINT64_C(1) << 9,
    HOTPOT_RD_FIELD_STEREO_RECORDS = UINT64_C(1) << 10,
    HOTPOT_RD_FIELD_ATOM_TRANSPORT = UINT64_C(1) << 11,
    HOTPOT_RD_FIELD_BOND_TRANSPORT = UINT64_C(1) << 12
} hotpot_rd_molecule_field;

typedef enum hotpot_rd_stereo_anchor_kind {
    /* ATOM carries exactly one atom index; BOND carries exactly two. */
    HOTPOT_RD_STEREO_ANCHOR_ATOM = 1,
    HOTPOT_RD_STEREO_ANCHOR_BOND = 2,
    /* AXIS carries an ordered axis path of at least three atom indices. */
    HOTPOT_RD_STEREO_ANCHOR_AXIS = 3
} hotpot_rd_stereo_anchor_kind;

typedef enum hotpot_rd_stereo_carrier_kind {
    HOTPOT_RD_STEREO_CARRIER_ATOM = 1,
    HOTPOT_RD_STEREO_CARRIER_IMPLICIT_H = 2,
    HOTPOT_RD_STEREO_CARRIER_LONE_PAIR = 3,
    HOTPOT_RD_STEREO_CARRIER_VACANCY = 4
} hotpot_rd_stereo_carrier_kind;

typedef enum hotpot_rd_stereo_group_kind {
    HOTPOT_RD_STEREO_GROUP_NONE = 0,
    HOTPOT_RD_STEREO_GROUP_ABS = 1,
    HOTPOT_RD_STEREO_GROUP_AND = 2,
    HOTPOT_RD_STEREO_GROUP_OR = 3
} hotpot_rd_stereo_group_kind;

typedef enum hotpot_rd_atom_transport_field {
    HOTPOT_RD_ATOM_TRANSPORT_ISOTOPE = 1u << 0,
    HOTPOT_RD_ATOM_TRANSPORT_RADICAL_ELECTRONS = 1u << 1
} hotpot_rd_atom_transport_field;

typedef enum hotpot_rd_hydrogen_transport_field {
    HOTPOT_RD_HYDROGEN_TRANSPORT_IMPLICIT = 1u << 0,
    HOTPOT_RD_HYDROGEN_TRANSPORT_BRACKET_EXPLICIT = 1u << 1,
    HOTPOT_RD_HYDROGEN_TRANSPORT_NO_IMPLICIT = 1u << 2
} hotpot_rd_hydrogen_transport_field;

typedef struct hotpot_rd_string_view_v1 {
    const char *data;
    uint64_t size;
} hotpot_rd_string_view_v1;

typedef struct hotpot_rd_stereo_carrier_v1 {
    uint32_t kind;
    uint32_t has_atom;
    uint64_t atom;
} hotpot_rd_stereo_carrier_v1;

typedef struct hotpot_rd_stereo_record_v1 {
    uint32_t anchor_kind;
    uint32_t group_kind;
    uint64_t anchor_offset;
    uint64_t anchor_count;
    uint64_t carrier_offset;
    uint64_t carrier_count;
    int64_t group_id;
    uint32_t has_group;
    uint32_t reserved;
    hotpot_rd_string_view_v1 group_scope;
    hotpot_rd_string_view_v1 descriptor;
    hotpot_rd_string_view_v1 source_namespace;
    hotpot_rd_string_view_v1 source_code;
} hotpot_rd_stereo_record_v1;

typedef struct hotpot_rd_hydrogen_state_v1 {
    uint32_t presence_mask;
    int32_t implicit_hydrogens;
    int32_t bracket_explicit_hydrogens;
    uint32_t no_implicit;
} hotpot_rd_hydrogen_state_v1;

typedef struct hotpot_rd_atom_transport_v1 {
    uint64_t atom;
    uint32_t presence_mask;
    int32_t isotope;
    int32_t radical_electrons;
    uint32_t has_current_hydrogens;
    uint32_t has_source_hydrogens;
    hotpot_rd_hydrogen_state_v1 current_hydrogens;
    hotpot_rd_hydrogen_state_v1 source_hydrogens;
    hotpot_rd_string_view_v1 source_namespace;
} hotpot_rd_atom_transport_v1;

typedef struct hotpot_rd_bond_transport_v1 {
    uint64_t first_atom;
    uint64_t second_atom;
    hotpot_rd_string_view_v1 display_direction;
} hotpot_rd_bond_transport_v1;

typedef struct hotpot_rd_options_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t conformer_count;
    int64_t seed;
    double prune_rms_threshold;
    uint32_t has_seed;
    uint32_t has_prune_rms_threshold;
    uint32_t use_random_coordinates;
    uint32_t reserved;
} hotpot_rd_options_v1;

typedef struct hotpot_rd_molecule_view_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t schema_version;
    uint32_t reserved;
    uint64_t atom_count;
    uint64_t bond_count;
    uint64_t field_mask;
    const int32_t *atomic_numbers;
    const int32_t *formal_charges;
    const double *partial_charges;
    const double *coordinates;
    const uint8_t *atom_aromatic;
    const int32_t *bond_indices;
    const double *bond_orders;
    const uint8_t *bond_kinds;
    const uint8_t *bond_aromatic;
    const double *unit_cell;
    const hotpot_rd_stereo_record_v1 *stereo_records;
    uint64_t stereo_record_count;
    const uint64_t *stereo_anchor_indices;
    uint64_t stereo_anchor_index_count;
    const hotpot_rd_stereo_carrier_v1 *stereo_carriers;
    uint64_t stereo_carrier_count;
    const hotpot_rd_atom_transport_v1 *atom_transport;
    uint64_t atom_transport_count;
    const hotpot_rd_bond_transport_v1 *bond_transport;
    uint64_t bond_transport_count;
} hotpot_rd_molecule_view_v1;

typedef struct hotpot_rd_constraints_view_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t coordination_pair_count;
    /* Directed zero-based (donor, metal) rows naming existing dative bonds. */
    const int32_t *coordination_pairs;
    /* Per-row (lower, upper) distances in Angstrom. */
    const double *metal_donor_bounds;
    uint64_t donor_pair_count;
    /* Unordered donor atom pairs. */
    const int32_t *donor_pair_indices;
    /* Per-row (lower, upper) distances in Angstrom. */
    const double *donor_pair_bounds;
} hotpot_rd_constraints_view_v1;

typedef struct hotpot_rd_conformer_record_v1 {
    uint64_t ordinal;
    int64_t engine_conformer_id;
} hotpot_rd_conformer_record_v1;

typedef struct hotpot_rd_sampler hotpot_rd_sampler;
typedef struct hotpot_rd_result hotpot_rd_result;

HOTPOT_RD_API const char *hotpot_rd_status_message(hotpot_rd_status status);

HOTPOT_RD_API hotpot_rd_status hotpot_rd_sampler_create(
    const hotpot_rd_options_v1 *options,
    hotpot_rd_sampler **out_sampler
);

HOTPOT_RD_API void hotpot_rd_sampler_destroy(hotpot_rd_sampler *sampler);

HOTPOT_RD_API hotpot_rd_status hotpot_rd_generate(
    hotpot_rd_sampler *sampler,
    const hotpot_rd_molecule_view_v1 *molecule,
    const hotpot_rd_constraints_view_v1 *constraints,
    hotpot_rd_result **out_result
);

HOTPOT_RD_API hotpot_rd_generation_status hotpot_rd_result_status(
    const hotpot_rd_result *result
);

HOTPOT_RD_API const double *hotpot_rd_result_coordinates(
    const hotpot_rd_result *result,
    uint64_t *conformer_count,
    uint64_t *atom_count
);

HOTPOT_RD_API const hotpot_rd_conformer_record_v1 *hotpot_rd_result_records(
    const hotpot_rd_result *result,
    uint64_t *record_count
);

HOTPOT_RD_API hotpot_rd_string_view_v1 hotpot_rd_result_engine(
    const hotpot_rd_result *result
);

HOTPOT_RD_API hotpot_rd_string_view_v1 hotpot_rd_result_engine_version(
    const hotpot_rd_result *result
);

HOTPOT_RD_API hotpot_rd_string_view_v1 hotpot_rd_result_diagnostic(
    const hotpot_rd_result *result
);

HOTPOT_RD_API void hotpot_rd_result_destroy(hotpot_rd_result *result);

HOTPOT_RD_API uint32_t hotpot_rd_runtime_abi_version(void);
HOTPOT_RD_API hotpot_rd_string_view_v1 hotpot_rd_runtime_hotpot_version(void);
HOTPOT_RD_API hotpot_rd_string_view_v1 hotpot_rd_runtime_rdkit_version(void);
HOTPOT_RD_API hotpot_rd_string_view_v1 hotpot_rd_runtime_rdkit_commit(void);

#ifdef __cplusplus
}
#endif

#endif
