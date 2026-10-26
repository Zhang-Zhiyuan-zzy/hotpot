# M_17-19 complete API evolution tree

Status: post-merge implementation inventory.

This document records the complete interface evolution from the common branch
point through the semantic merge. It covers public and private Python, C, and
C++ interfaces at class/function/method level. It is an API tree, not a
compatibility promise: deleted names have no alias in the final tree.

## Legend and lineage

```text
2704f3c  common ancestor
├── 6639754 ... f4b9a94  Plan 17 rdWrapper/stereochemistry line
│   └── [source: rdwrapper]
├── b912c8a             Plan 19 native-Graph line
│   └── [source: graph]
└── b5e04d6 ... ca0289e semantic merge and closure
    └── [source: merge]
```

Every tree node uses these dimensions:

```text
[public | private]
[Python | C | C++]
[add: purpose]
[delete]
[rename: old -> new]
[refurbish: retained name, changed implementation/contract]
[merge: old owners -> one final owner]
[source: rdwrapper | graph | merge]
```

A label on a parent node applies to every child unless a child overrides it.
“Module-public/package-private” means a function appears in the submodule's
`__all__`, but is not re-exported from `hotpot.cheminfo.graph`.

## Complete interface tree

```text
hotpot
└── cheminfo
    ├── stereo.py
    │   ├── public Python [add: stereochemical transport and semantics]
    │   │   [source: rdwrapper]
    │   │   ├── enum StereoAnchorKind
    │   │   │   ├── ATOM
    │   │   │   ├── BOND
    │   │   │   └── AXIS
    │   │   ├── enum StereoCarrierKind
    │   │   │   ├── ATOM
    │   │   │   ├── IMPLICIT_H
    │   │   │   ├── LONE_PAIR
    │   │   │   └── VACANCY
    │   │   ├── enum StereoGroupKind: ABS, AND, OR
    │   │   ├── enum StereoType
    │   │   │   ├── TETRAHEDRAL, DOUBLE_BOND, ALLENE, ATROPISOMERIC
    │   │   │   ├── SQUARE_PLANAR, TRIGONAL_BIPYRAMIDAL
    │   │   │   ├── SQUARE_PYRAMIDAL, OCTAHEDRAL
    │   │   │   └── UNKNOWN
    │   │   ├── enum StereoIssueCode
    │   │   │   ├── INVALID_ANCHOR, INVALID_CARRIER, INVALID_DESCRIPTOR
    │   │   │   ├── NON_STEREOGENIC, DEGENERATE_GEOMETRY
    │   │   │   ├── COORDINATE_CONTRADICTION, INCOMPLETE_MAPPING
    │   │   │   └── UNSUPPORTED
    │   │   ├── class StereoGroupRef(scope: str, kind: StereoGroupKind, id: int)
    │   │   ├── class StereoCarrier(
    │   │   │       kind: StereoCarrierKind,
    │   │   │       atom: int | None = None,
    │   │   │   )
    │   │   ├── class StereoAnchor(
    │   │   │       kind: StereoAnchorKind,
    │   │   │       atom_indices: tuple[int, ...],
    │   │   │   )
    │   │   ├── class StereoRecord(
    │   │   │       anchor: StereoAnchor,
    │   │   │       carriers: tuple[StereoCarrier, ...],
    │   │   │       descriptor: str,
    │   │   │       group: StereoGroupRef | None,
    │   │   │       source_namespace: str,
    │   │   │       source_code: str,
    │   │   │   )
    │   │   ├── class HydrogenTransportState(
    │   │   │       implicit_hydrogens: int | None,
    │   │   │       bracket_explicit_hydrogens: int | None,
    │   │   │       no_implicit: bool | None,
    │   │   │   )
    │   │   ├── class AtomTransportRecord(
    │   │   │       atom: int,
    │   │   │       isotope: int | None,
    │   │   │       radical_electrons: int | None,
    │   │   │       current_hydrogens: HydrogenTransportState | None,
    │   │   │       source_hydrogens: HydrogenTransportState | None,
    │   │   │       source_namespace: str | None,
    │   │   │   )
    │   │   ├── class BondTransportRecord(
    │   │   │       atom_indices: tuple[int, int],
    │   │   │       display_direction: str | None,
    │   │   │   )
    │   │   ├── class StereoPayload(
    │   │   │       schema_version: int,
    │   │   │       records: tuple[StereoRecord, ...],
    │   │   │       atom_facts: tuple[AtomTransportRecord, ...],
    │   │   │       bond_facts: tuple[BondTransportRecord, ...],
    │   │   │   )
    │   │   ├── class StereoValidationIssue(
    │   │   │       code: StereoIssueCode,
    │   │   │       message: str,
    │   │   │       record_index: int | None = None,
    │   │   │       atom_indices: tuple[int, ...] = (),
    │   │   │   )
    │   │   ├── class StereoValidationReport(
    │   │   │       is_valid: bool,
    │   │   │       issues: tuple[StereoValidationIssue, ...],
    │   │   │   )
    │   │   ├── class StereoDerivedDescriptor(
    │   │   │       record_index: int,
    │   │   │       stereo_type: StereoType,
    │   │   │       label: str,
    │   │   │       system: str,
    │   │   │       is_absolute: bool,
    │   │   │   )
    │   │   ├── class StereoAssignmentReport(
    │   │   │       payload: StereoPayload,
    │   │   │       unresolved: tuple[StereoValidationIssue, ...],
    │   │   │       derived: tuple[StereoDerivedDescriptor, ...] = (),
    │   │   │   )
    │   │   ├── class StereoAlignmentReport(
    │   │   │       is_equivalent: bool,
    │   │   │       aligned_payload: StereoPayload,
    │   │   │       issues: tuple[StereoValidationIssue, ...],
    │   │   │   )
    │   │   ├── get_stereo_payload(mol: Molecule) -> StereoPayload
    │   │   ├── set_stereo_payload(mol: Molecule, payload: StereoPayload) -> None
    │   │   ├── validate_stereochemistry(
    │   │   │       mol: Molecule,
    │   │   │       *, coordinates: NDArray[np.float64] | None = None,
    │   │   │   ) -> StereoValidationReport
    │   │   ├── assign_stereochemistry(
    │   │   │       mol: Molecule,
    │   │   │       *, coordinates: NDArray[np.float64] | None = None,
    │   │   │   ) -> StereoAssignmentReport
    │   │   └── align_stereochemistry(
    │   │           reference: Molecule,
    │   │           target: Molecule,
    │   │           atom_mapping: Mapping[int, int],
    │   │       ) -> StereoAlignmentReport
    │   └── private Python [add: semantic implementation] [source: rdwrapper]
    │       ├── descriptor/graph types
    │       │   ├── enum _Level2DescriptorFamily
    │       │   ├── class _Level2Descriptor(family, value)
    │       │   ├── class _Level2AlgebraResult(descriptor, issues)
    │       │   ├── class _Gate6Graph(
    │       │   │       target_spec, directed_dative_edges,
    │       │   │       real_atom_count, virtual_hydrogens,
    │       │   │   )
    │       │   ├── enum _Gate6Equivalence: EQUIVALENT, DISTINCT, UNKNOWN
    │       │   ├── class _Gate6SemanticContext(...)
    │       │   ├── class _Gate7CoordinationSite(
    │       │   │       record_index, descriptor, center, carrier_rows,
    │       │   │   )
    │       │   ├── class _Gate7CoordinateCandidate(descriptor, score_radians)
    │       │   └── class _AlignmentMatch(
    │       │           reference_index, target_index, mapped, target, transformed,
    │       │       )
    │       ├── Level-1 transport functions
    │       │   ├── _bond_key(atom_indices)
    │       │   ├── _axis_bond_keys(atom_indices)
    │       │   └── _remap_stereo_payload(
    │       │           payload, atom_mapping, *, allowed_bonds=None,
    │       │       )
    │       ├── Level-2 descriptor-algebra functions
    │       │   ├── _algebra_issue(code, message, *, atom_indices=())
    │       │   ├── _parse_level2_descriptor_head(text)
    │       │   ├── _parse_level2_descriptor(text)
    │       │   ├── _render_level2_descriptor(descriptor)
    │       │   ├── _declares_gate7_coordination(text)
    │       │   ├── _validate_stereo_carriers(carriers, *, expected_count)
    │       │   ├── _validate_stereo_anchor(anchor, *, family)
    │       │   ├── _validated_permutation(permutation)
    │       │   ├── _permutation_parity(permutation)
    │       │   ├── _permutation_rank(permutation)
    │       │   ├── _permutation_unrank(size, rank)
    │       │   ├── _permutation_sign(permutation)
    │       │   ├── _octahedral_rotations()
    │       │   ├── _coordination_representatives(size, rotations)
    │       │   ├── _carrier_reordering(source, reordered)
    │       │   ├── _is_endpoint_pair_reordering(permutation)
    │       │   ├── _endpoint_pair_reordering(
    │       │   │       source, reordered, *, endpoint_groups_reversed,
    │       │   │   )
    │       │   └── _transform_level2_descriptor(
    │       │           descriptor, source_carriers, reordered_carriers,
    │       │           *, source_anchor=None, reordered_anchor=None,
    │       │       )
    │       ├── Gate-6 represented-graph functions
    │       │   ├── _gate6_issue(message, *, atom_indices=())
    │       │   ├── _gate6_semantics(degree, valence)
    │       │   ├── _gate6_edge_is_cyclic(edge, adjacency)
    │       │   ├── _gate6_projection(mol)
    │       │   ├── _gate6_query_spec(
    │       │   │       graph, pins, *, hydrogen_slot_exemptions=None,
    │       │   │   )
    │       │   ├── _gate6_carrier_keys(record, descriptor, graph)
    │       │   ├── _gate6_materialize_record(record, descriptor, graph)
    │       │   ├── _gate6_semantic_context(
    │       │   │       payload, graph, *, excluded_record_index,
    │       │   │       valid_record_indices, unsupported_record_indices,
    │       │   │   )
    │       │   ├── _gate6_dative_direction_is_preserved(graph, mapping)
    │       │   ├── _gate6_opaque_record_is_fixed(record, graph, mapping)
    │       │   ├── _transform_supported_record(descriptor, mapped, target)
    │       │   ├── _gate6_mapping_preserves_stereo(graph, context, mapping)
    │       │   ├── _gate6_carrier_pair_equivalence(
    │       │   │       native_target, graph, context, record, carrier_keys, pair,
    │       │   │   )
    │       │   └── _gate6_stereogenicity_issues(
    │       │           mol, graph_report, *, projection_result=None,
    │       │       )
    │       ├── graph-validation-gate functions
    │       │   ├── _record_atom_indices(record)
    │       │   ├── _record_issues(issues, *, record_index, atom_indices)
    │       │   ├── _gate2_issue(
    │       │   │       code, message, *, record_index, atom_indices=(),
    │       │   │   )
    │       │   ├── _validate_gate2_double_bond_carriers(carriers)
    │       │   ├── _validate_gate5_axial_carriers(carriers)
    │       │   ├── _molecule_stereo_topology(mol)
    │       │   ├── _molecule_stereo_bond_kinds(mol)
    │       │   ├── _validate_gate2_atom_ranges(
    │       │   │       record, *, record_index, atom_count,
    │       │   │   )
    │       │   ├── _validate_gate2_tetrahedral_graph(
    │       │   │       record, *, record_index, atom_count, adjacency,
    │       │   │   )
    │       │   ├── _validate_gate2_double_bond_graph(
    │       │   │       record, *, record_index, atom_count, adjacency, bond_orders,
    │       │   │   )
    │       │   ├── _validate_gate5_axial_graph(
    │       │   │       record, *, record_index, atom_count, adjacency,
    │       │   │       bond_orders, bond_kinds,
    │       │   │   )
    │       │   ├── _validate_gate7_coordination_record(
    │       │   │       record, descriptor, *, record_index, atom_count, adjacency,
    │       │   │   )
    │       │   ├── _validate_gate2_stereochemistry(
    │       │   │       mol, *, projection_result=None,
    │       │   │   )
    │       │   └── _validate_graph_stereochemistry(
    │       │           mol, *, projection_result=None,
    │       │       )
    │       ├── Gate-7 coordination/geometry functions
    │       │   ├── _gate7_represented_adjacency(graph)
    │       │   ├── _gate7_same_center_issues(
    │       │   │       payload, descriptors, *, atom_count,
    │       │   │   )
    │       │   ├── _gate7_carrier_components(site, adjacency)
    │       │   ├── _gate7_chelate_pairs(site, adjacency)
    │       │   ├── _gate7_coupled_multicenter_issues(sites, adjacency)
    │       │   ├── _gate7_site_from_record(record_index, record, descriptor)
    │       │   ├── _gate7_mapped_chelate_pairs(pairs, atom_mapping)
    │       │   ├── _validate_gate7_coordination_semantics(
    │       │   │       mol, payload, descriptors, blocked_record_indices,
    │       │   │       *, projection_result=None,
    │       │   │   )
    │       │   ├── _geometry_issue(
    │       │   │       message, *, record_index=None, atom_indices=(),
    │       │   │   )
    │       │   ├── _finite_coordinate_array(mol, coordinates)
    │       │   ├── _normalized_geometry_sign(value, scale)
    │       │   ├── _is_relatively_degenerate(length, local_scale)
    │       │   ├── _gate7_ideal_carrier_directions(descriptor)
    │       │   ├── _gate7_proper_rotation_fit(ideal, observed)
    │       │   ├── _gate7_coordination_candidates(family, observed)
    │       │   ├── _select_gate7_coordination_descriptor(
    │       │   │       declared_family, candidates_by_family,
    │       │   │       *, record_index, atom_indices,
    │       │   │   )
    │       │   ├── _gate7_center_rooted_directions(site, coordinates)
    │       │   ├── _gate7_coordination_coordinate_descriptor(site, coordinates)
    │       │   ├── _tetrahedral_coordinate_descriptor(
    │       │   │       record, coordinates, *, record_index,
    │       │   │   )
    │       │   ├── _double_bond_coordinate_descriptor(
    │       │   │       record, coordinates, *, record_index,
    │       │   │   )
    │       │   ├── _axial_coordinate_descriptor(
    │       │   │       record, coordinates, *, record_index,
    │       │   │   )
    │       │   ├── _gate4_coordinate_assignments(mol, coordinates, graph_report)
    │       │   ├── _records_atom_union(records, indices)
    │       │   ├── _coordinate_group_consistency_issues(payload, assignments)
    │       │   ├── _gate7_projection_result(mol)
    │       │   ├── _direct_priority_order(mol, carriers, atom_facts)
    │       │   └── _direct_priority_assignments(mol, payload, graph_report)
    │       └── alignment functions
    │           ├── _alignment_validation_issues(side, report)
    │           ├── _alignment_mapping_issues(reference, target, atom_mapping)
    │           ├── _mapped_alignment_record(record, atom_mapping)
    │           ├── _alignment_record_identity(family, anchor)
    │           ├── _transform_gate3_double_bond(descriptor, source, target)
    │           ├── _transform_gate5_axial(descriptor, source, target)
    │           ├── _relative_group(record)
    │           └── _alignment_group_issues(
    │                   reference_payload, target_payload, matches,
    │               )
    │
    ├── rdWrappers
    │   ├── public Python [add: native ETKDG component] [source: rdwrapper]
    │   │   ├── enum GenerationStatus(IntEnum)
    │   │   │   ├── SUCCESS = 0
    │   │   │   ├── PARTIAL = 1
    │   │   │   ├── UNSUPPORTED = 2
    │   │   │   └── FAILED = 3
    │   │   ├── class ETKDGOptions(
    │   │   │       *, conformer_count: int = 1, seed: int | None = None,
    │   │   │       prune_rms_threshold: float | None = None,
    │   │   │       use_random_coordinates: bool = False,
    │   │   │   )
    │   │   ├── class CoordinationEmbeddingConstraints(
    │   │   │       coordination_pairs: NDArray[np.int32],
    │   │   │       metal_donor_bounds: NDArray[np.float64],
    │   │   │       donor_pair_indices: NDArray[np.int32] | None = None,
    │   │   │       donor_pair_bounds: NDArray[np.float64] | None = None,
    │   │   │   )
    │   │   ├── class ConformerRecord(
    │   │   │       ordinal: int, engine_conformer_id: int,
    │   │   │   )
    │   │   ├── class ConformerBatch(
    │   │   │       status: GenerationStatus,
    │   │   │       coordinates: NDArray[np.float64],
    │   │   │       records: tuple[ConformerRecord, ...],
    │   │   │       engine: str,
    │   │   │       engine_version: str,
    │   │   │       diagnostic: str | None,
    │   │   │   )
    │   │   ├── class RuntimeInfo(
    │   │   │       hotpot_version: str, rdkit_version: str, native: bool,
    │   │   │   )
    │   │   ├── class ETKDGSampler(options: ETKDGOptions | None = None)
    │   │   │   └── generate(
    │   │   │           mol: Molecule,
    │   │   │           *, constraints: CoordinationEmbeddingConstraints | None = None,
    │   │   │       ) -> ConformerBatch
    │   │   ├── generate_conformers(
    │   │   │       mol: Molecule,
    │   │   │       *, options: ETKDGOptions | None = None,
    │   │   │       constraints: CoordinationEmbeddingConstraints | None = None,
    │   │   │   ) -> ConformerBatch
    │   │   └── runtime_info() -> RuntimeInfo
    │   ├── private Python native bridge [add] [source: rdwrapper]
    │   │   ├── _backend()
    │   │   ├── generate(
    │   │   │       mol: Molecule,
    │   │   │       options: ETKDGOptions,
    │   │   │       constraints: CoordinationEmbeddingConstraints | None,
    │   │   │   ) -> ConformerBatch
    │   │   ├── runtime_info() -> RuntimeInfo
    │   │   └── _rd_native
    │   │       ├── generate(
    │   │       │       molecule_buffers: NativeMoleculeBuffers,
    │   │       │       options: ETKDGOptions,
    │   │       │       constraints: CoordinationEmbeddingConstraints | None = ...,
    │   │       │   ) -> ConformerBatch
    │   │       ├── runtime_info() -> RuntimeInfo
    │   │       ├── C_API_VERSION: int
    │   │       └── RDKIT_COMMIT: str
    │   ├── public C ABI v1 [add: standalone component] [source: rdwrapper]
    │   │   ├── macro HOTPOT_RD_C_API_VERSION = 1u
    │   │   ├── enums
    │   │   │   ├── hotpot_rd_status
    │   │   │   ├── hotpot_rd_generation_status
    │   │   │   ├── hotpot_rd_bond_kind
    │   │   │   ├── hotpot_rd_molecule_field
    │   │   │   ├── hotpot_rd_stereo_anchor_kind
    │   │   │   ├── hotpot_rd_stereo_carrier_kind
    │   │   │   ├── hotpot_rd_stereo_group_kind
    │   │   │   ├── hotpot_rd_atom_transport_field
    │   │   │   └── hotpot_rd_hydrogen_transport_field
    │   │   ├── structs
    │   │   │   ├── hotpot_rd_string_view_v1
    │   │   │   ├── hotpot_rd_stereo_carrier_v1
    │   │   │   ├── hotpot_rd_stereo_record_v1
    │   │   │   ├── hotpot_rd_hydrogen_state_v1
    │   │   │   ├── hotpot_rd_atom_transport_v1
    │   │   │   ├── hotpot_rd_bond_transport_v1
    │   │   │   ├── hotpot_rd_options_v1
    │   │   │   ├── hotpot_rd_molecule_view_v1
    │   │   │   ├── hotpot_rd_constraints_view_v1
    │   │   │   └── hotpot_rd_conformer_record_v1
    │   │   ├── opaque hotpot_rd_sampler
    │   │   ├── opaque hotpot_rd_result
    │   │   ├── hotpot_rd_status_message(hotpot_rd_status) -> const char *
    │   │   ├── hotpot_rd_sampler_create(
    │   │   │       const hotpot_rd_options_v1 *, hotpot_rd_sampler **,
    │   │   │   ) -> hotpot_rd_status
    │   │   ├── hotpot_rd_sampler_destroy(hotpot_rd_sampler *) -> void
    │   │   ├── hotpot_rd_generate(
    │   │   │       hotpot_rd_sampler *, const hotpot_rd_molecule_view_v1 *,
    │   │   │       const hotpot_rd_constraints_view_v1 *, hotpot_rd_result **,
    │   │   │   ) -> hotpot_rd_status
    │   │   ├── hotpot_rd_result_status(
    │   │   │       const hotpot_rd_result *,
    │   │   │   ) -> hotpot_rd_generation_status
    │   │   ├── hotpot_rd_result_coordinates(
    │   │   │       const hotpot_rd_result *, uint64_t *, uint64_t *,
    │   │   │   ) -> const double *
    │   │   ├── hotpot_rd_result_records(
    │   │   │       const hotpot_rd_result *, uint64_t *,
    │   │   │   ) -> const hotpot_rd_conformer_record_v1 *
    │   │   ├── hotpot_rd_result_engine(const hotpot_rd_result *)
    │   │   │   -> hotpot_rd_string_view_v1
    │   │   ├── hotpot_rd_result_engine_version(const hotpot_rd_result *)
    │   │   │   -> hotpot_rd_string_view_v1
    │   │   ├── hotpot_rd_result_diagnostic(const hotpot_rd_result *)
    │   │   │   -> hotpot_rd_string_view_v1
    │   │   ├── hotpot_rd_result_destroy(hotpot_rd_result *) -> void
    │   │   ├── hotpot_rd_runtime_abi_version() -> uint32_t
    │   │   ├── hotpot_rd_runtime_hotpot_version() -> hotpot_rd_string_view_v1
    │   │   ├── hotpot_rd_runtime_rdkit_version() -> hotpot_rd_string_view_v1
    │   │   └── hotpot_rd_runtime_rdkit_commit() -> hotpot_rd_string_view_v1
    │   └── public C++17 SDK [add: standalone component] [source: rdwrapper]
    │       ├── CMake target Hotpot::rdwrappers
    │       └── namespace hotpot::rdwrappers
    │           ├── enum GenerationStatus
    │           ├── struct ETKDGOptions
    │           ├── using DistanceBounds = std::array<double, 2>
    │           ├── struct CoordinationEmbeddingConstraintsView
    │           ├── struct ConformerRecord
    │           ├── struct ConformerGenerationResult
    │           ├── struct RuntimeInfo
    │           ├── class ETKDGSampler
    │           │   ├── ETKDGSampler(ETKDGOptions options = {})
    │           │   └── generate(
    │           │           native::MoleculeBufferView molecule,
    │           │           std::optional<CoordinationEmbeddingConstraintsView>
    │           │               constraints = std::nullopt,
    │           │       ) const -> ConformerGenerationResult
    │           ├── generate_conformers(
    │           │       native::MoleculeBufferView molecule,
    │           │       ETKDGOptions options = {},
    │           │       std::optional<CoordinationEmbeddingConstraintsView>
    │           │           constraints = std::nullopt,
    │           │   ) -> ConformerGenerationResult
    │           └── runtime_info() -> RuntimeInfo
    │
    ├── _native/buffers.py
    │   └── private Python [add: field-selective common transport]
    │       [source: rdwrapper]
    │       ├── enum NativeBondKindCode(IntEnum)
    │       ├── enum NativeMoleculeField(Enum)
    │       ├── enum NativeStereoAnchorKindCode(IntEnum)
    │       ├── enum NativeStereoCarrierKindCode(IntEnum)
    │       ├── enum NativeStereoGroupKindCode(IntEnum)
    │       ├── enum NativeAtomTransportFieldCode(IntFlag)
    │       ├── enum NativeHydrogenTransportFieldCode(IntFlag)
    │       ├── class NativeUtf8Table(data, offsets, lengths, present)
    │       ├── class NativeStereoRecordBuffers(
    │       │       schema_version, anchor_kinds, anchor_offsets, anchor_counts,
    │       │       anchor_indices, carrier_offsets, carrier_counts,
    │       │       carrier_kinds, carrier_has_atoms, carrier_atoms,
    │       │       group_kinds, group_ids, group_present, group_scopes,
    │       │       descriptors, source_namespaces, source_codes,
    │       │   )
    │       ├── class NativeAtomTransportBuffers(
    │       │       schema_version, atoms, presence_masks, isotopes,
    │       │       radical_electrons, has_current_hydrogens,
    │       │       current_hydrogen_presence_masks, current_implicit_hydrogens,
    │       │       current_bracket_explicit_hydrogens, current_no_implicit,
    │       │       has_source_hydrogens, source_hydrogen_presence_masks,
    │       │       source_implicit_hydrogens,
    │       │       source_bracket_explicit_hydrogens, source_no_implicit,
    │       │       source_namespaces,
    │       │   )
    │       ├── class NativeBondTransportBuffers(
    │       │       schema_version, atom_indices, display_directions,
    │       │   )
    │       ├── class NativeMoleculeBuffers(
    │       │       schema_version, atom_count, bond_count,
    │       │       atomic_numbers=None, formal_charges=None,
    │       │       partial_charges=None, coordinates=None,
    │       │       atom_aromatic=None, bond_indices=None, bond_orders=None,
    │       │       bond_kinds=None, bond_aromatic=None, unit_cell=None,
    │       │       stereo_records=None, atom_transport=None,
    │       │       bond_transport=None,
    │       │   )
    │       ├── pack_molecule(
    │       │       mol: Molecule,
    │       │       fields: frozenset[NativeMoleculeField],
    │       │   ) -> NativeMoleculeBuffers
    │       ├── _pack_utf8(values: tuple[str | None, ...]) -> NativeUtf8Table
    │       ├── _pack_stereo_records(payload) -> NativeStereoRecordBuffers
    │       ├── _pack_hydrogen_state(state)
    │       ├── _pack_atom_transport(payload, atoms)
    │       │   -> NativeAtomTransportBuffers
    │       └── _pack_bond_transport(payload) -> NativeBondTransportBuffers
    │
    ├── native C++ transport SDK
    │   └── public C++ [add: common Python/native communication]
    │       [source: rdwrapper]
    │       └── namespace hotpot::native
    │           ├── template<class Value> ArrayView
    │           ├── aliases BondIndex, TransportBondIndex, Coordinate, UnitCell
    │           ├── enums BondKind, MoleculeField, StereoAnchorKind
    │           ├── enums StereoCarrierKind, StereoGroupKind
    │           ├── enums AtomTransportField, HydrogenTransportField
    │           ├── view types
    │           │   ├── Utf8TableView
    │           │   ├── StereoRecordBuffersView
    │           │   ├── AtomTransportBuffersView
    │           │   ├── BondTransportBuffersView
    │           │   ├── MoleculeBufferView
    │           │   └── ConformerBatchView
    │           │       └── validate()
    │           ├── owner types
    │           │   ├── OwnedUtf8TableData
    │           │   │   └── view() const noexcept
    │           │   ├── OwnedStereoRecordData
    │           │   │   └── view() const noexcept
    │           │   ├── OwnedAtomTransportData
    │           │   │   └── view() const noexcept
    │           │   ├── OwnedBondTransportData
    │           │   │   └── view() const noexcept
    │           │   └── OwnedMoleculeData
    │           │       ├── view() const noexcept -> MoleculeBufferView
    │           │       └── validate()
    │           ├── molecule_field_mask(MoleculeField) noexcept
    │           └── validate_molecule_view(const MoleculeBufferView&)
    │
    ├── conformers
    │   └── private Python [add: pluggable conformer-provider seam]
    │       [source: rdwrapper]
    │       ├── class ConformerSamplingError(RuntimeError)
    │       ├── class ConformerSamplingRequest(
    │       │       *, conformer_count: int = 1, seed: int | None = None,
    │       │   )
    │       ├── class ConformerBatch(
    │       │       coordinates: NDArray[np.float64], ordinals: tuple[int, ...],
    │       │   )
    │       ├── protocol ConformerSampler
    │       │   └── sample(
    │       │           molecule: Molecule,
    │       │           request: ConformerSamplingRequest,
    │       │       ) -> ConformerBatch
    │       ├── class OpenBabelConformerAdapter(
    │       │       _builder: Callable[[Molecule], None] | None = None,
    │       │   )
    │       │   └── sample(molecule, request) -> ConformerBatch
    │       ├── class RDKitConformerAdapter(materialized_hydrogens: bool = False)
    │       │   └── sample(molecule, request) -> ConformerBatch
    │       └── _materialized_hydrogen_projection(molecule: Molecule) -> Molecule
    │
    ├── forcefields
    │   ├── private Python call chain
    │   │   └── [refurbish: add trailing
    │   │       sampler: Optional[ConformerSampler] = None]
    │   │       [source: rdwrapper]
    │   │       ├── _build_ligand_proxies(..., sampler=None)
    │   │       ├── _build_ligand_proxies_worker(..., sampler=None)
    │   │       ├── _prepare_complex_working_mol(..., sampler=None)
    │   │       ├── _build_complex3d_workflow(..., sampler=None)
    │   │       ├── _complexes_build_workflow(..., sampler=None)
    │   │       └── _build_and_optimize_workflow(..., sampler=None)
    │   └── private Python working-copy transaction [source: merge]
    │       ├── _WorkingCopyCommit
    │       │   └── stereo_payload: StereoPayload
    │       │       [add: commit stereochemistry atomically]
    │       └── _MoleculeCommitSnapshot
    │           ├── [delete] graph, row_to_index, angles, torsions, rings
    │           ├── [delete] cycle_basis_rings, ring_indices_cache
    │           ├── [delete] ligand_rings, ligand_cycle_basis_rings
    │           ├── [delete] ligand_rings_signature, obmol
    │           ├── [delete] Atom._neighbours/Atom._bonds snapshots
    │           ├── atom_pair_items [refurbish: retain authoritative facts]
    │           └── stereo_payload [add: exact restore]
    │
    ├── obWrappers
    │   ├── public C++ [source: rdwrapper]
    │   │   └── namespace hotpot::obwrappers
    │   │       ├── BondKind
    │   │       │   [rename+merge: independent enum ->
    │   │       │    using BondKind = hotpot::native::BondKind]
    │   │       ├── class MoleculeData
    │   │       │   [refurbish+merge: independent owner ->
    │   │       │    derives hotpot::native::OwnedMoleculeData]
    │   │       │   └── view() const noexcept
    │   │       │       -> hotpot::native::MoleculeBufferView
    │   │       ├── validate_openbabel_compatibility(
    │   │       │       const hotpot::native::MoleculeBufferView&,
    │   │       │   ) [add]
    │   │       ├── make_obmol(const hotpot::native::MoleculeBufferView&)
    │   │       │   [add: non-owning input]
    │   │       └── make_obmol(const MoleculeData&)
    │   │           [refurbish: delegate to shared view]
    │   ├── private Python _MoleculeBuffers
    │   │   [merge: independent dataclass -> NativeMoleculeBuffers alias]
    │   │   [source: rdwrapper]
    │   ├── private Python _pack_molecule(mol)
    │   │   [refurbish: pack_molecule(mol, _OPENBABEL_FIELDS)]
    │   │   [source: rdwrapper]
    │   └── private C++ force-field bridge [source: rdwrapper]
    │       └── [rename+refurbish:
    │           ligand_molecule_data(const ComplexSessionInput&)
    │           -> ligand_molecule_view(const ComplexSessionInput&) noexcept]
    │           return: owned obwrappers::MoleculeData
    │           -> non-owning native::MoleculeBufferView
    │
    ├── core.py
    │   ├── class Molecule
    │   │   ├── private topology authority [source: graph]
    │   │   │   ├── [merge+rename:
    │   │   │   │    _update_graph(clear_conformers=True)
    │   │   │   │    + _update_atom_neighbours_bonds()
    │   │   │   │    -> _update_topology(
    │   │   │   │           clear_conformers: bool = True,
    │   │   │   │       ) -> None]
    │   │   │   ├── _topology_index:
    │   │   │   │   dict[id(atom), tuple[atom_row, tuple[Bond, ...]]]
    │   │   │   │   [add: sole persistent topology index]
    │   │   │   ├── _graph [delete]
    │   │   │   ├── _edge_with_attrs() [delete]
    │   │   │   ├── _node_with_attrs() [delete]
    │   │   │   ├── _ring_graph(*, ligand_skeleton=False) [delete]
    │   │   │   ├── _uncached_cycle_basis_rings(
    │   │   │   │       *, ligand_skeleton=False,
    │   │   │   │   ) [delete]
    │   │   │   ├── _cycle_basis_rings [delete]
    │   │   │   └── _ligand_cycle_basis_rings [delete]
    │   │   ├── public graph surfaces [source: graph]
    │   │   │   ├── graph [delete]
    │   │   │   ├── simple_graph [delete]
    │   │   │   ├── atom_bond_graph [delete]
    │   │   │   ├── adjacency_matrix [delete persistent property/cache]
    │   │   │   ├── super_acyclic_graph() [delete]
    │   │   │   ├── cycle_basis_rings [delete]
    │   │   │   ├── ligand_cycle_basis_rings [delete]
    │   │   │   ├── cycle_basis -> list[tuple[int, ...]]
    │   │   │   │   [add: deterministic native default basis]
    │   │   │   ├── all_cycle_basis -> list[list[tuple[int, ...]]]
    │   │   │   │   [add: bounded complete basis family]
    │   │   │   ├── basis_rings -> list[Ring]
    │   │   │   │   [add: default-basis domain projection]
    │   │   │   └── all_basis_rings -> list[list[Ring]]
    │   │   │       [add: all-bases domain projection]
    │   │   ├── public mutation [source: merge]
    │   │   │   ├── remove_atom(atom: int | Atom) -> None [retain]
    │   │   │   ├── remove_atoms(atoms: Iterable[Atom]) -> None [retain]
    │   │   │   ├── remove_bond(bond: Bond) -> None
    │   │   │   │   [refurbish: delegates to sole bulk implementation]
    │   │   │   └── remove_bonds(bonds: Iterable[Bond]) -> None
    │   │   │       [merge: public bulk path + _rm_bond + _rm_bonds;
    │   │   │        one stereo prune and topology refresh]
    │   │   ├── private mutation [source: merge]
    │   │   │   ├── [merge+rename:
    │   │   │   │    _rm_atom(atom) + _rm_atoms(atoms)
    │   │   │   │    -> _remove_atoms_from_topology(
    │   │   │   │           atoms: Iterable[Atom | int],
    │   │   │   │       ) -> None]
    │   │   │   ├── _rm_bond(bond) [delete after merge]
    │   │   │   └── _rm_bonds(bonds) [delete after merge]
    │   │   ├── private stereochemical lifecycle [source: rdwrapper/merge]
    │   │   │   ├── _stereo_payload: StereoPayload [add]
    │   │   │   ├── _structural_bond_rows()
    │   │   │   │   -> tuple[tuple[int, int], ...] [add]
    │   │   │   ├── _set_remapped_stereo_payload(
    │   │   │   │       source: Molecule,
    │   │   │   │       atom_mapping: Mapping[int, int],
    │   │   │   │       *, filter_bonds: bool,
    │   │   │   │   ) -> None [add]
    │   │   │   ├── _remap_stereo_after_atom_removal(
    │   │   │   │       source_atoms: tuple[Atom, ...],
    │   │   │   │   ) -> None [add]
    │   │   │   └── _remove_stereo_bond_rows(
    │   │   │           removed_bond_rows: Iterable[tuple[int, int]],
    │   │   │       ) -> None [add]
    │   │   ├── __copy__(), copy(), components, add_component()
    │   │   │   [refurbish: exact StereoPayload copy/remap/filter]
    │   │   │   [source: rdwrapper/merge]
    │   │   └── shortest_path_index(...)
    │   │       [refurbish: private BFS -> graph.paths.shortest_path]
    │   │       [source: merge]
    │   ├── class Atom [source: graph]
    │   │   ├── _neighbours [delete stored adjacency]
    │   │   ├── Atom-specific _bonds [delete stored adjacency]
    │   │   ├── bonds -> list[Bond]
    │   │   │   [refurbish: Molecule._topology_index lookup]
    │   │   ├── neighbours -> list[Atom]
    │   │   │   [refurbish: Molecule._topology_index lookup]
    │   │   └── cycle_basis_rings [delete]
    │   └── class MolBlock
    │       └── cycle_basis_rings [delete] [source: graph]
    │
    ├── graph
    │   ├── components.py
    │   │   └── connected_components(
    │   │           node_count: int,
    │   │           edges: Iterable[tuple[int, int]],
    │   │       ) -> tuple[tuple[int, ...], ...]
    │   │       [module-public/package-private Python]
    │   │       [add: direct explicit-edge algorithm] [source: graph]
    │   ├── paths.py
    │   │   ├── shortest_path(
    │   │   │       node_count: int, edges: Iterable[tuple[int, int]],
    │   │   │       source: int, target: int,
    │   │   │   ) -> list[int]
    │   │   ├── all_simple_paths(
    │   │   │       node_count: int, edges: Iterable[tuple[int, int]],
    │   │   │       source: int, target: int, cutoff: int | None = None,
    │   │   │   ) -> list[list[int]]
    │   │   ├── longest_simple_path(
    │   │   │       node_count: int, edges: Iterable[tuple[int, int]],
    │   │   │       source: int | None = None, target: int | None = None,
    │   │   │   ) -> list[int]
    │   │   └── depth_first_path(
    │   │           node_count: int, edges: Iterable[tuple[int, int]],
    │   │           start_node: int = 0,
    │   │           scope_nodes: Container[int] | None = None,
    │   │           min_depth: int | None = None,
    │   │           max_depth: int | None = None,
    │   │       ) -> list[int] | None
    │   │       [module-public/package-private Python]
    │   │       [add: direct explicit-edge algorithms] [source: graph]
    │   ├── graph_dfs_path
    │   │   [public Python]
    │   │   [refurbish: graph_dfs_path(
    │   │       graph: nx.Graph, start_node=None, scope_nodes=None,
    │   │       min_deep=None, max_deep=None)
    │   │       -> graph_dfs_path(
    │   │       node_count: int, edges: Iterable[tuple[int, int]],
    │   │       start_node: int = None, scope_nodes: Container = None,
    │   │       min_deep: int = None, max_deep: int = None)
    │   │       -> Optional[list[int]]]
    │   │   [source: graph]
    │   ├── graph_dfs_paths(
    │   │       graph: nx.Graph, start_node: int, scope_nodes=None,
    │   │       min_deep=None, max_deep=None,
    │   │   ) -> list[list[int]]
    │   │   [public Python] [delete: constant-empty legacy surface]
    │   │   [source: merge]
    │   ├── private C++ base graph [source: graph]
    │   │   └── namespace hotpot::graph
    │   │       ├── using NodeKey = std::int64_t
    │   │       ├── using Position = std::uint32_t
    │   │       ├── using EdgeIndex = std::uint32_t
    │   │       ├── struct Edge { Position first; Position second; }
    │   │       ├── class Graph final
    │   │       │   ├── Graph(
    │   │       │   │       std::vector<NodeKey> ordered_keys,
    │   │       │   │       const std::vector<pair<NodeKey, NodeKey>>&
    │   │       │   │           ordered_edges,
    │   │       │   │   )
    │   │       │   ├── node_count() const noexcept -> std::size_t
    │   │       │   ├── edge_count() const noexcept -> std::size_t
    │   │       │   ├── keys() const noexcept -> const vector<NodeKey>&
    │   │       │   ├── edges() const noexcept -> const vector<Edge>&
    │   │       │   ├── offsets() const noexcept -> const vector<Position>&
    │   │       │   ├── neighbors() const noexcept -> const vector<Position>&
    │   │       │   ├── adjacency_edges() const noexcept
    │   │       │   │   -> const vector<EdgeIndex>&
    │   │       │   ├── degrees() const noexcept -> const vector<Position>&
    │   │       │   ├── neighbors(Position) const noexcept
    │   │       │   │   -> pair<NeighborIterator, NeighborIterator>
    │   │       │   ├── incident_edges(Position) const noexcept
    │   │       │   │   -> pair<IncidentEdgeIterator, IncidentEdgeIterator>
    │   │       │   ├── degree(Position) const noexcept -> Position
    │   │       │   ├── self_loop_edge(Position) const noexcept -> int32_t
    │   │       │   ├── edge_between(Position, Position) const noexcept
    │   │       │   │   -> int32_t
    │   │       │   └── position_for_key(NodeKey) const -> Position [private]
    │   │       ├── using NodeCycle = std::vector<NodeKey>
    │   │       ├── using CycleBasis = std::vector<NodeCycle>
    │   │       ├── cycle_basis(const Graph&) -> CycleBasis
    │   │       └── all_cycle_basis(
    │   │               const Graph&, std::size_t max_bases = 10'000,
    │   │           ) -> std::vector<CycleBasis>
    │   └── native extension ownership [source: graph/merge]
    │       ├── [rename+merge:
    │       │    hotpot.cheminfo.graph._relevant_cycles
    │       │    -> hotpot.cheminfo.graph._graph_native]
    │       ├── relevant_cycles(...) [retain]
    │       ├── cycle_basis(...) [add]
    │       ├── all_cycle_basis(...) [add]
    │       └── C++ RelevantEdge
    │           [rename: generic relevant-cycle Edge -> RelevantEdge;
    │            avoids collision with hotpot::graph::Edge]
    │
    ├── InternalCoordinates
    │   ├── calc_zmat_idx(
    │   │       graph_or_link: Union[nx.Graph, np.ndarray],
    │   │   )
    │   │   [public Python old signature]
    │   │   [refurbish -> calc_zmat_idx(
    │   │       link_matrix: np.ndarray, node_count: Optional[int] = None)]
    │   │   [source: graph]
    │   └── calc_zmat(
    │           graph_or_link: Union[nx.Graph, np.ndarray],
    │           coords: np.ndarray,
    │       )
    │       [public Python old signature]
    │       [refurbish -> calc_zmat(
    │           link_matrix: np.ndarray, coords: np.ndarray)]
    │       [source: graph]
    │
    ├── search
    │   ├── public Python query/results [source: graph/merge]
    │   │   ├── class Substructure
    │   │   │   ├── query_graph [delete]
    │   │   │   ├── construct_graph() [delete]
    │   │   │   ├── query_atoms [retain]
    │   │   │   └── query_bonds [retain]
    │   │   ├── class Searcher
    │   │   │   ├── _node_match(mol_node, query_node)
    │   │   │   │   [private delete]
    │   │   │   └── _edge_match(mol_edge, query_edge)
    │   │   │       [private delete]
    │   │   ├── class Hits
    │   │   │   ├── [rename+refurbish constructor:
    │   │   │   │    Hits(sub, mol, graph_matcher, get_hit=True,
    │   │   │   │         max_matches=None)
    │   │   │   │    -> Hits(sub, mol, mapping_source,
    │   │   │   │         get_hit: bool = True, max_matches: int = None)]
    │   │   │   ├── graph_matcher [rename -> mapping_source]
    │   │   │   └── _get_nodes_set() [private delete]
    │   │   └── class Hit
    │   │       └── [refurbish constructor:
    │   │            Hit(mol, sub, atom_indices, mappings=())
    │   │            -> Hit(mol, sub, atom_indices, mappings)]
    │   ├── private Python query-graph cache [delete] [source: graph]
    │   │   ├── Substructure._query_graph_signature
    │   │   ├── Substructure._invalidate_query_graph()
    │   │   ├── Substructure._graph_signature()
    │   │   └── Substructure._cached_graph_is_current(signature)
    │   ├── private Python mapping protocol [source: merge]
    │   │   ├── [rename+refurbish:
    │   │   │    _NativeMappingSource.subgraph_monomorphisms_iter()
    │   │   │    (molecule-index -> query-index)
    │   │   │    -> _NativeMappingSource.iter_mappings()
    │   │   │    (query-index -> molecule-index)]
    │   │   ├── [rename+refurbish:
    │   │   │    _MappingIterator._query_to_molecule(mol_to_query)
    │   │   │    -> _MappingIterator._readonly_mapping(query_to_molecule)]
    │   │   ├── [merge+rename:
    │   │   │    _NativeSpecTarget + scattered NativeTarget construction
    │   │   │    -> _NativeTarget(target_spec: dict)]
    │   │   │   ├── problem(query_spec: dict)
    │   │   │   ├── mapping_source(query_spec: dict)
    │   │   │   │   -> _NativeMappingSource
    │   │   │   ├── iter_mappings(query_spec: dict)
    │   │   │   │   -> Iterable[dict[int, int]]
    │   │   │   └── automorphism(node_labels, edge_labels)
    │   │   ├── _mapping_sources_with_automorphism(queries, mol) [add]
    │   │   ├── _exact_topology_labels(mol, keys, edges) [add]
    │   │   ├── _integer_labels(values) [add]
    │   │   └── class _NativeAutomorphismTask(...) [add]
    │   │       └── equivalent(first: int, second: int) -> bool
    │   ├── private C++ RI/search [source: graph/merge]
    │   │   ├── [rename+move+refurbish:
    │   │   │    hotpot::search::GraphData in search/_native/graph_data.*
    │   │   │    -> hotpot::graph::Graph in graph/_native/graph.*]
    │   │   ├── SearchProblem::query()/target()
    │   │   │   [refurbish: const GraphData& -> const graph::Graph&]
    │   │   ├── build_query_plan(...)
    │   │   │   [refurbish: const GraphData& -> const graph::Graph&]
    │   │   ├── NativeProblem::query_
    │   │   │   [refurbish: GraphData -> graph::Graph]
    │   │   ├── NativeTarget::graph_
    │   │   │   [refurbish: GraphData -> graph::Graph]
    │   │   └── class NativeAutomorphism final : SearchProblem [add]
    │   │       ├── NativeAutomorphism(
    │   │       │       shared_ptr<NativeTarget> target,
    │   │       │       vector<int64_t> node_labels,
    │   │       │       vector<int64_t> edge_labels,
    │   │       │   )
    │   │       └── equivalent(int64_t first, int64_t second) const -> bool
    │   ├── private Python pybind surface [add] [source: merge]
    │   │   └── class _NativeAutomorphism(
    │   │           target: NativeTarget,
    │   │           node_labels: tuple[int, ...],
    │   │           edge_labels: tuple[int, ...],
    │   │       )
    │   │       └── equivalent(first: int, second: int) -> bool
    │   └── obsolete Python implementation island [private delete]
    │       [source: merge]
    │       ├── _logic.py
    │       │   ├── Chiral, BinaryLogicMeta, FieldSetABC, SpFieldSet
    │       │   ├── RealsSet, FieldSet, AtomicNumSet, BoolSet, ChiralSet
    │       │   ├── LogicDict, LogicTuple, OrTuple, AndTuple
    │       │   └── value_check()
    │       ├── _logic_tuple.py
    │       │   └── LogicalTuple, AndTuple, OrTuple, NotTuple
    │       ├── fields.py
    │       │   ├── Field, Chiral, BoolFieldMeta, BoolField
    │       │   ├── register_field()
    │       │   └── get_field()
    │       ├── logic.py
    │       │   ├── MutexValue, MutexBool, LogicSet, SetUnion
    │       │   ├── AndDict, LogicTuple, AndTuple, OrTuple
    │       │   └── set_replace_add()
    │       ├── logic2.py: LogicSet, ProductDict
    │       ├── match_func.py
    │       │   ├── MatcherType, LogicType
    │       │   ├── in_matcher(), min_max_matcher()
    │       │   └── logic_merge()
    │       ├── _smarts.py
    │       │   ├── TokenType, tokenize(), parse_bracket_atom()
    │       │   └── substructure_from_smarts() and private parser helpers
    │       └── smarts_parser/
    │           ├── TokenType, SegmentType, AttrType, MutexChiral
    │           ├── validate_brackets(), extract_brackets()
    │           ├── extract_numeric_pairs(), extract_branches()
    │           ├── tokenize_smarts(), validate_token_continuous()
    │           ├── parse_bracket_atom(), parse_simple_atom()
    │           └── interpret_tokens()
    │
    ├── AImodels/mca
    │   ├── public Python [source: graph]
    │   │   ├── class MoleculeGraph [delete]
    │   │   │   ├── to_rdmol() [delete]
    │   │   │   └── to_rdkit [delete alias]
    │   │   ├── graph_adapter.py [delete]
    │   │   └── predictors/result/is_Nuc_site
    │   │       [retain: public business contract unchanged]
    │   ├── private Python graph/isomorphism [source: graph/merge]
    │   │   ├── _atom_label(atom) [delete]
    │   │   ├── _bond_label(bond) [delete]
    │   │   ├── _labeled_graph(mol) [delete]
    │   │   ├── _rooted_isomorphic(graph, first, second) [delete]
    │   │   ├── [refurbish:
    │   │   │    _refined_node_colors(graph: nx.Graph)
    │   │   │    -> _refined_node_colors(automorphism)]
    │   │   └── [refurbish:
    │   │        _remove_automorphic_sites(mol, sites)
    │   │        -> _remove_automorphic_sites(automorphism, sites)]
    │   └── _shortest_paths(adjacency)
    │       [private Python, signature retained]
    │       [refurbish: dense for node count < 256; SciPy sparse for >= 256;
    │        int32 result and 510 clipping retained] [source: graph]
    │
    ├── obconvert.py
    │   └── add_hydrogens(mol)
    │       [private/unexported Python delete: depended on deleted _rm_atoms]
    │       [source: merge]
    │
    └── transport/stereo compatibility cleanup
        ├── private Python [source: merge]
        │   ├── _EMPTY_STEREO_PAYLOAD [delete]
        │   ├── get_stereo_payload() getattr fallback [delete]
        │   ├── _HOTPOT_LEGACY_NAMESPACE [delete]
        │   ├── _utf8_values(table) [delete]
        │   ├── _add_legacy_hydrogen_transport(mol, buffers) [delete]
        │   └── [refurbish:
        │        _pack_atom_transport(payload)
        │        -> _pack_atom_transport(payload, atoms)]
        └── private C++ [source: merge]
            └── native "hotpot-legacy/1" parser branch [delete]
```

## Ownership boundaries in the final tree

- `graph.paths` and `graph.components` are module-public/package-private. They
  deliberately are not re-exported from `hotpot.cheminfo.graph`.
- `hotpot::graph` and `hotpot::search` headers are private build APIs. They are
  not installed SDK surfaces.
- The installed C++ SDK surfaces are `hotpot::native` transport and
  `hotpot::rdwrappers`; the standalone C ABI is `hotpot_rd_*` version 1.
- `Hotpot::rdwrappers` is the CMake target; `hotpot::rdwrappers` is the C++
  namespace.
- The current SMARTS implementation remains `search/smarts.py` plus query IR
  and native RI. Deleted `_smarts.py`/`smarts_parser/` names are not aliases for
  the current parser.
- Public force-field workflow signatures and defaults are unchanged; only the
  private conformer-provider chain acquired the optional sampler seam.
- MCA site order, site type, prediction result shape, and `is_Nuc_site`
  semantics are retained. Only graph construction, automorphism execution, and
  large-graph shortest-path ownership changed.

## Final single-owner audit

```text
Topology refresh implementation                         1
Persistent topology index                               1
Private native Graph implementation                     1
RI-style matcher implementation                         1
Generic Python/C++ molecule transport                   1
RDKit native conformer-generation pipeline              1
StereoPayload owner per Molecule                        1
Single/Bulk Bond-removal implementation                 1

Compatibility aliases retained                          0
Production NetworkX APIs/imports                        0
MoleculeGraph adapters                                  0
RDKit-specific duplicate molecule buffers               0
Old _relevant_cycles extension                          0
search/_native/graph_data implementation                0
hotpot-legacy/1 translators                             0
```

Thus the merge does not preserve parallel legacy paths: each retained behavior
has one final owner, and every removed surface is represented above as a
deletion, rename, refurbishment, or merge into that owner.
