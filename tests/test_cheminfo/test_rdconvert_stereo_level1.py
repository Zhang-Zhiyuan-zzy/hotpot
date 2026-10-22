from dataclasses import replace

import pytest
from rdkit import Chem, rdBase

from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.rdconvert import from_rdmol, to_rdmol
from hotpot.cheminfo.stereo import (
    BondTransportRecord,
    HydrogenTransportState,
    StereoCarrier,
    StereoCarrierKind,
    StereoGroupKind,
    get_stereo_payload,
    set_stereo_payload,
)


def _round_trip(smiles):
    source = Chem.MolFromSmiles(smiles)
    molecule = from_rdmol(source, Molecule())
    return source, molecule, to_rdmol(molecule, kekulize=False)


def _atom_state(atom):
    return (
        str(atom.GetChiralTag()),
        atom.GetIsotope(),
        atom.GetNumRadicalElectrons(),
        atom.GetNumImplicitHs(),
        atom.GetNumExplicitHs(),
        atom.GetNoImplicit(),
    )


def _bond_state(bond):
    return (
        bond.GetBeginAtomIdx(),
        bond.GetEndAtomIdx(),
        str(bond.GetStereo()),
        tuple(bond.GetStereoAtoms()),
        str(bond.GetBondDir()),
    )


def _external_payload_state(payload):
    records = tuple(
        (
            record.anchor,
            record.carriers,
            record.descriptor,
            None
            if record.group is None
            else (record.group.kind, record.group.id),
            record.source_namespace,
            record.source_code,
        )
        for record in payload.records
    )
    return records, payload.atom_facts, payload.bond_facts


def test_level1_adapter_is_pinned_to_rdkit_2026_09_1():
    assert rdBase.rdkitVersion == "2026.09.1"

    molecule = from_rdmol(Chem.MolFromSmiles("CC"), Molecule())
    payload = get_stereo_payload(molecule)
    assert {
        record.source_namespace for record in payload.records
    } == {"rdkit-2026.09.1"}
    assert {
        fact.source_namespace for fact in payload.atom_facts
    } == {"rdkit-2026.09.1"}


def test_tetrahedral_and_unspecified_chiral_tags_round_trip_in_atom_order():
    source, molecule, exported = _round_trip("F[C@](Cl)(Br)I")
    payload = get_stereo_payload(molecule)
    atom_records = [
        record
        for record in payload.records
        if record.anchor.kind.value == "atom"
    ]

    assert [record.anchor.atom_indices for record in atom_records] == [
        (index,) for index in range(source.GetNumAtoms())
    ]
    assert [record.source_code for record in atom_records] == [
        str(atom.GetChiralTag()) for atom in source.GetAtoms()
    ]
    assert [_atom_state(atom) for atom in exported.GetAtoms()] == [
        _atom_state(atom) for atom in source.GetAtoms()
    ]


def test_non_tetrahedral_tag_is_transported_but_export_is_explicitly_limited():
    source = Chem.RWMol()
    source.AddAtom(Chem.Atom(46))
    for atomic_number in (9, 17, 35, 53):
        source.AddAtom(Chem.Atom(atomic_number))
        source.AddBond(0, source.GetNumAtoms() - 1, Chem.BondType.SINGLE)
    source.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_SQUAREPLANAR)

    molecule = from_rdmol(source.GetMol(), Molecule())

    assert get_stereo_payload(molecule).records[0].source_code == (
        "CHI_SQUAREPLANAR"
    )
    with pytest.raises(ValueError, match="unfrozen permutation property"):
        to_rdmol(molecule, kekulize=False)


def _implicit_h_tetrahedral_molecule():
    source = Chem.RWMol()
    source.AddAtom(Chem.Atom(6))
    for atomic_number in (9, 17, 35):
        source.AddAtom(Chem.Atom(atomic_number))
        source.AddBond(0, source.GetNumAtoms() - 1, Chem.BondType.SINGLE)
    source.GetAtomWithIdx(0).SetChiralTag(
        Chem.ChiralType.CHI_TETRAHEDRAL_CW
    )
    source.UpdatePropertyCache(strict=False)
    assert source.GetAtomWithIdx(0).GetNumImplicitHs() == 1
    return source.GetMol()


def test_tetra_carriers_follow_bond_order_and_append_only_true_implicit_h():
    source = _implicit_h_tetrahedral_molecule()
    molecule = from_rdmol(source, Molecule())
    record = get_stereo_payload(molecule).records[0]

    assert tuple(carrier.kind for carrier in record.carriers) == (
        StereoCarrierKind.ATOM,
        StereoCarrierKind.ATOM,
        StereoCarrierKind.ATOM,
        StereoCarrierKind.IMPLICIT_H,
    )
    assert tuple(carrier.atom for carrier in record.carriers) == (
        1,
        2,
        3,
        None,
    )
    exported = to_rdmol(molecule, kekulize=False)
    assert exported.GetAtomWithIdx(0).GetChiralTag() == (
        source.GetAtomWithIdx(0).GetChiralTag()
    )


def test_bracket_explicit_virtual_h_uses_three_atom_carriers_and_round_trips():
    source = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
    chiral_atom = next(
        atom
        for atom in source.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )
    molecule = from_rdmol(source, Molecule())
    record = get_stereo_payload(molecule).records[chiral_atom.GetIdx()]

    assert chiral_atom.GetNumExplicitHs() == 1
    assert chiral_atom.GetNumImplicitHs() == 0
    assert all(
        carrier.kind is StereoCarrierKind.ATOM
        for carrier in record.carriers
    )
    assert len(record.carriers) == 3

    exported = to_rdmol(molecule, kekulize=False)
    exported_chiral_atom = exported.GetAtomWithIdx(chiral_atom.GetIdx())

    assert exported_chiral_atom.GetNumExplicitHs() == 1
    assert exported_chiral_atom.GetNumImplicitHs() == 0
    assert exported_chiral_atom.GetChiralTag() == chiral_atom.GetChiralTag()
    assert Chem.MolToSmiles(exported, isomericSmiles=True) == (
        Chem.MolToSmiles(source, isomericSmiles=True)
    )


def test_bracket_explicit_h_odd_bond_reordering_preserves_configuration():
    source = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
    imported = from_rdmol(source, Molecule())
    chiral_index = next(
        atom.GetIdx()
        for atom in source.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )
    incident = [
        bond
        for bond in imported.bonds
        if chiral_index in (bond.atom1.idx, bond.atom2.idx)
    ]
    other = [bond for bond in imported.bonds if bond not in incident]
    reordered = Molecule()
    for atom in imported.atoms:
        reordered._create_atom(**atom.attr_dict)
    for bond in (incident[1], incident[0], *incident[2:], *other):
        reordered._add_bond(
            bond.atom1.idx,
            bond.atom2.idx,
            **bond.attr_dict,
        )
    reordered._update_graph()
    set_stereo_payload(reordered, get_stereo_payload(imported))

    exported = to_rdmol(reordered, kekulize=False)

    assert exported.GetAtomWithIdx(chiral_index).GetChiralTag() != (
        source.GetAtomWithIdx(chiral_index).GetChiralTag()
    )
    assert Chem.MolToSmiles(exported, isomericSmiles=True) == (
        Chem.MolToSmiles(source, isomericSmiles=True)
    )


def test_odd_tetra_carrier_permutation_flips_cw_ccw():
    source = _implicit_h_tetrahedral_molecule()
    molecule = from_rdmol(source, Molecule())
    payload = get_stereo_payload(molecule)
    record = payload.records[0]
    permuted = replace(
        record,
        carriers=(
            record.carriers[1],
            record.carriers[0],
            *record.carriers[2:],
        ),
    )
    set_stereo_payload(
        molecule,
        replace(payload, records=(permuted, *payload.records[1:])),
    )

    exported = to_rdmol(molecule, kekulize=False)

    assert exported.GetAtomWithIdx(0).GetChiralTag() == (
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            lambda carriers: (
                carriers[0],
                carriers[0],
                *carriers[2:],
            ),
            "contain duplicates",
        ),
        (
            lambda carriers: (
                StereoCarrier(StereoCarrierKind.LONE_PAIR),
                *carriers[1:],
            ),
            "cannot represent lone-pair",
        ),
        (lambda carriers: carriers[:2], "requires three or four"),
        (
            lambda carriers: (
                StereoCarrier(StereoCarrierKind.ATOM, 0),
                *carriers[1:],
            ),
            "not a permutation",
        ),
    ],
)
def test_unprovable_tetra_carrier_mappings_are_rejected(replacement, message):
    molecule = from_rdmol(_implicit_h_tetrahedral_molecule(), Molecule())
    payload = get_stereo_payload(molecule)
    record = payload.records[0]
    changed = replace(record, carriers=replacement(record.carriers))
    set_stereo_payload(
        molecule,
        replace(payload, records=(changed, *payload.records[1:])),
    )

    with pytest.raises(ValueError, match=message):
        to_rdmol(molecule, kekulize=False)


@pytest.mark.parametrize(
    ("source", "hydrogens"),
    [
        (
            _implicit_h_tetrahedral_molecule,
            HydrogenTransportState(0, 0, True),
        ),
        (
            lambda: Chem.MolFromSmiles("F[C@](Cl)(Br)I"),
            HydrogenTransportState(0, 1, True),
        ),
    ],
)
def test_tetra_center_requires_exactly_four_graph_or_virtual_neighbors(
    source,
    hydrogens,
):
    molecule = from_rdmol(source(), Molecule())
    payload = get_stereo_payload(molecule)
    center = next(
        record.anchor.atom_indices[0]
        for record in payload.records
        if record.source_code in (
            "CHI_TETRAHEDRAL_CW",
            "CHI_TETRAHEDRAL_CCW",
        )
    )
    changed_facts = tuple(
        replace(fact, current_hydrogens=hydrogens)
        if fact.atom == center
        else fact
        for fact in payload.atom_facts
    )
    set_stereo_payload(
        molecule,
        replace(payload, atom_facts=changed_facts),
    )

    with pytest.raises(ValueError, match="exactly four graph or virtual"):
        to_rdmol(molecule, kekulize=False)


@pytest.mark.parametrize("smiles", ["F/C=C/F", "F/C=C\\F"])
def test_double_bond_stereo_controlling_atoms_and_display_direction_round_trip(
    smiles,
):
    source, molecule, exported = _round_trip(smiles)
    payload = get_stereo_payload(molecule)

    assert [_bond_state(bond) for bond in exported.GetBonds()] == [
        _bond_state(bond) for bond in source.GetBonds()
    ]
    assert [fact.display_direction for fact in payload.bond_facts] == [
        str(bond.GetBondDir()) for bond in source.GetBonds()
    ]


@pytest.mark.parametrize("smiles", ["F/C=C/F", "F/C=C\\F"])
def test_bond_stereo_remains_equivalent_when_target_anchor_is_reversed(smiles):
    source = Chem.MolFromSmiles(smiles)
    imported = from_rdmol(source, Molecule())
    reversed_molecule = Molecule()
    for atom in imported.atoms:
        reversed_molecule._create_atom(**atom.attr_dict)
    for bond in imported.bonds:
        first, second = bond.atom1.idx, bond.atom2.idx
        if bond.bond_order == 2:
            first, second = second, first
        reversed_molecule._add_bond(first, second, **bond.attr_dict)
    reversed_molecule._update_graph()
    set_stereo_payload(
        reversed_molecule,
        get_stereo_payload(imported),
    )

    exported = to_rdmol(reversed_molecule, kekulize=False)
    source_double = next(
        bond for bond in source.GetBonds() if bond.GetBondTypeAsDouble() == 2
    )
    exported_double = next(
        bond for bond in exported.GetBonds() if bond.GetBondTypeAsDouble() == 2
    )

    assert exported_double.GetBeginAtomIdx() == source_double.GetEndAtomIdx()
    assert tuple(exported_double.GetStereoAtoms()) == tuple(
        reversed(source_double.GetStereoAtoms())
    )
    assert exported_double.GetStereo() == source_double.GetStereo()
    assert Chem.MolToSmiles(exported, isomericSmiles=True) == (
        Chem.MolToSmiles(source, isomericSmiles=True)
    )


def test_reversed_atropisomer_anchor_is_explicitly_rejected():
    source = Chem.RWMol()
    for atomic_number in (9, 6, 6, 17):
        source.AddAtom(Chem.Atom(atomic_number))
    source.AddBond(0, 1, Chem.BondType.SINGLE)
    source.AddBond(1, 2, Chem.BondType.SINGLE)
    source.AddBond(2, 3, Chem.BondType.SINGLE)
    atrop_bond = source.GetBondBetweenAtoms(1, 2)
    atrop_bond.SetStereoAtoms(0, 3)
    atrop_bond.SetStereo(Chem.BondStereo.STEREOATROPCW)
    imported = from_rdmol(source.GetMol(), Molecule())
    reversed_molecule = Molecule()
    for atom in imported.atoms:
        reversed_molecule._create_atom(**atom.attr_dict)
    for bond in imported.bonds:
        first, second = bond.atom1.idx, bond.atom2.idx
        if frozenset((first, second)) == frozenset((1, 2)):
            first, second = second, first
        reversed_molecule._add_bond(first, second, **bond.attr_dict)
    reversed_molecule._update_graph()
    set_stereo_payload(
        reversed_molecule,
        get_stereo_payload(imported),
    )

    with pytest.raises(ValueError, match="frozen reversed-anchor"):
        to_rdmol(reversed_molecule, kekulize=False)


def test_slash_display_direction_flips_when_its_bond_endpoint_order_reverses():
    source = Chem.MolFromSmiles("F/C=C/F")
    imported = from_rdmol(source, Molecule())
    directed_source = next(
        bond
        for bond in source.GetBonds()
        if bond.GetBondDir() == Chem.BondDir.ENDUPRIGHT
    )
    reversed_molecule = Molecule()
    for atom in imported.atoms:
        reversed_molecule._create_atom(**atom.attr_dict)
    for bond in imported.bonds:
        first, second = bond.atom1.idx, bond.atom2.idx
        if bond.idx == directed_source.GetIdx():
            first, second = second, first
        reversed_molecule._add_bond(first, second, **bond.attr_dict)
    reversed_molecule._update_graph()
    set_stereo_payload(
        reversed_molecule,
        get_stereo_payload(imported),
    )

    exported = to_rdmol(reversed_molecule, kekulize=False)
    directed_export = exported.GetBondWithIdx(directed_source.GetIdx())

    assert directed_export.GetBeginAtomIdx() == directed_source.GetEndAtomIdx()
    assert directed_export.GetBondDir() == Chem.BondDir.ENDDOWNRIGHT
    assert Chem.MolToSmiles(exported, isomericSmiles=True) == (
        Chem.MolToSmiles(source, isomericSmiles=True)
    )


def test_crossed_double_display_is_endpoint_order_invariant():
    source = Chem.RWMol()
    source.AddAtom(Chem.Atom(6))
    source.AddAtom(Chem.Atom(6))
    source.AddBond(0, 1, Chem.BondType.DOUBLE)
    source.GetBondWithIdx(0).SetBondDir(Chem.BondDir.EITHERDOUBLE)
    imported = from_rdmol(source.GetMol(), Molecule())
    reversed_molecule = Molecule()
    for atom in imported.atoms:
        reversed_molecule._create_atom(**atom.attr_dict)
    reversed_molecule._add_bond(1, 0, **imported.bonds[0].attr_dict)
    reversed_molecule._update_graph()
    set_stereo_payload(
        reversed_molecule,
        get_stereo_payload(imported),
    )

    exported = to_rdmol(reversed_molecule, kekulize=False)

    assert exported.GetBondWithIdx(0).GetBeginAtomIdx() == 1
    assert exported.GetBondWithIdx(0).GetBondDir() == (
        Chem.BondDir.EITHERDOUBLE
    )


@pytest.mark.parametrize(
    "direction",
    [Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH, Chem.BondDir.UNKNOWN],
)
def test_reversed_endpoint_anchored_display_direction_is_rejected(direction):
    source = Chem.RWMol()
    source.AddAtom(Chem.Atom(6))
    source.AddAtom(Chem.Atom(8))
    source.AddBond(0, 1, Chem.BondType.SINGLE)
    source.GetBondWithIdx(0).SetBondDir(direction)
    imported = from_rdmol(source.GetMol(), Molecule())
    reversed_molecule = Molecule()
    for atom in imported.atoms:
        reversed_molecule._create_atom(**atom.attr_dict)
    reversed_molecule._add_bond(1, 0, **imported.bonds[0].attr_dict)
    reversed_molecule._update_graph()
    set_stereo_payload(
        reversed_molecule,
        get_stereo_payload(imported),
    )

    with pytest.raises(ValueError, match="endpoint-anchored bond display"):
        to_rdmol(reversed_molecule, kekulize=False)


def test_duplicate_atom_facts_stereo_anchors_and_bond_facts_are_rejected():
    _, molecule, _ = _round_trip("F/C=C/F")
    payload = get_stereo_payload(molecule)
    atom_record = payload.records[0]
    bond_record = next(
        record
        for record in payload.records
        if record.anchor.kind.value == "bond"
    )

    set_stereo_payload(
        molecule,
        replace(
            payload,
            atom_facts=payload.atom_facts + (payload.atom_facts[0],),
        ),
    )
    with pytest.raises(ValueError, match="Duplicate RDKit atom transport"):
        to_rdmol(molecule, kekulize=False)

    set_stereo_payload(
        molecule,
        replace(payload, records=payload.records + (atom_record,)),
    )
    with pytest.raises(ValueError, match="Duplicate RDKit stereo anchor"):
        to_rdmol(molecule, kekulize=False)

    reversed_bond_record = replace(
        bond_record,
        anchor=replace(
            bond_record.anchor,
            atom_indices=tuple(reversed(bond_record.anchor.atom_indices)),
        ),
    )
    set_stereo_payload(
        molecule,
        replace(payload, records=payload.records + (reversed_bond_record,)),
    )
    with pytest.raises(ValueError, match="Duplicate RDKit stereo anchor"):
        to_rdmol(molecule, kekulize=False)

    first_bond_fact = payload.bond_facts[0]
    reversed_bond_fact = replace(
        first_bond_fact,
        atom_indices=tuple(reversed(first_bond_fact.atom_indices)),
    )
    set_stereo_payload(
        molecule,
        replace(
            payload,
            bond_facts=payload.bond_facts + (reversed_bond_fact,),
        ),
    )
    with pytest.raises(ValueError, match="Duplicate RDKit bond transport"):
        to_rdmol(molecule, kekulize=False)


def test_conflicting_source_enhanced_group_membership_is_rejected():
    source = Chem.RWMol(Chem.MolFromSmiles("F[C@](Cl)(Br)I"))
    first = Chem.CreateStereoGroup(
        Chem.StereoGroupType.STEREO_AND,
        source,
        [1],
        [],
        1,
    )
    second = Chem.CreateStereoGroup(
        Chem.StereoGroupType.STEREO_OR,
        source,
        [1],
        [],
        2,
    )
    source.SetStereoGroups([first, second])

    with pytest.raises(ValueError, match="conflicting atom stereo groups"):
        from_rdmol(source.GetMol(), Molecule())


@pytest.mark.parametrize(
    ("cxsmiles", "expected_kind", "expected_id"),
    [
        ("F[C@](Cl)(Br)I |&12:1|", StereoGroupKind.AND, 12),
        ("F[C@](Cl)(Br)I |o7:1|", StereoGroupKind.OR, 7),
        ("F[C@](Cl)(Br)I |a:1|", StereoGroupKind.ABS, 0),
    ],
)
def test_enhanced_stereo_group_kind_membership_and_source_id_round_trip(
    cxsmiles,
    expected_kind,
    expected_id,
):
    source, molecule, exported = _round_trip(cxsmiles)
    source_group = source.GetStereoGroups()[0]
    group_ref = next(
        record.group
        for record in get_stereo_payload(molecule).records
        if record.group is not None
    )
    exported_group = exported.GetStereoGroups()[0]

    assert group_ref.kind is expected_kind
    assert group_ref.id == expected_id
    assert exported_group.GetGroupType() == source_group.GetGroupType()
    assert exported_group.GetReadId() == source_group.GetReadId()
    assert [atom.GetIdx() for atom in exported_group.GetAtoms()] == [
        atom.GetIdx() for atom in source_group.GetAtoms()
    ]


def test_independent_imports_allocate_distinct_enhanced_group_scopes():
    source = Chem.MolFromSmiles("F[C@](Cl)(Br)I |&3:1|")
    first = from_rdmol(source, Molecule())
    second = from_rdmol(source, Molecule())

    first_group = next(
        record.group
        for record in get_stereo_payload(first).records
        if record.group is not None
    )
    second_group = next(
        record.group
        for record in get_stereo_payload(second).records
        if record.group is not None
    )

    assert first_group.id == second_group.id == 3
    assert first_group.scope != second_group.scope


def test_isotope_radical_and_three_hydrogen_values_round_trip_exactly():
    source, molecule, exported = _round_trip("[13CH3][CH2]C")
    payload = get_stereo_payload(molecule)

    assert [_atom_state(atom) for atom in exported.GetAtoms()] == [
        _atom_state(atom) for atom in source.GetAtoms()
    ]
    assert [fact.current_hydrogens for fact in payload.atom_facts] == [
        HydrogenTransportState(
            atom.GetNumImplicitHs(),
            atom.GetNumExplicitHs(),
            atom.GetNoImplicit(),
        )
        for atom in source.GetAtoms()
    ]
    assert all(
        fact.source_hydrogens is fact.current_hydrogens
        for fact in payload.atom_facts
    )


def test_combined_level1_payload_round_trips_except_for_new_import_scope():
    source = Chem.MolFromSmiles(
        "F[C@](Cl)(Br)[C@](I)([13CH2])N/C=C/F |&12:1,4|"
    )
    first = from_rdmol(source, Molecule())
    exported = to_rdmol(first, kekulize=False)
    second = from_rdmol(exported, Molecule())

    assert _external_payload_state(get_stereo_payload(second)) == (
        _external_payload_state(get_stereo_payload(first))
    )


def test_export_uses_current_hydrogens_and_does_not_mutate_source_ledger():
    source, molecule, _ = _round_trip("CC")
    payload = get_stereo_payload(molecule)
    first = payload.atom_facts[0]
    impossible_source = HydrogenTransportState(99, 88, True)
    changed_payload = replace(
        payload,
        atom_facts=(
            replace(first, source_hydrogens=impossible_source),
            *payload.atom_facts[1:],
        ),
    )
    set_stereo_payload(molecule, changed_payload)

    exported = to_rdmol(molecule, kekulize=False)

    assert exported.GetAtomWithIdx(0).GetNumImplicitHs() == (
        source.GetAtomWithIdx(0).GetNumImplicitHs()
    )
    assert get_stereo_payload(molecule) is changed_payload
    assert changed_payload.atom_facts[0].source_hydrogens is impossible_source


def test_unrepresentable_current_implicit_hydrogen_count_is_explicitly_rejected():
    _, molecule, _ = _round_trip("CC")
    payload = get_stereo_payload(molecule)
    first = payload.atom_facts[0]
    changed_hydrogens = replace(
        first.current_hydrogens,
        implicit_hydrogens=99,
    )
    set_stereo_payload(
        molecule,
        replace(
            payload,
            atom_facts=(
                replace(first, current_hydrogens=changed_hydrogens),
                *payload.atom_facts[1:],
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="cannot reproduce current implicit hydrogen count",
    ):
        to_rdmol(molecule, kekulize=False)


def test_empty_rdkit_molecule_has_empty_level1_tables():
    molecule = from_rdmol(Chem.Mol(), Molecule())

    assert get_stereo_payload(molecule).records == ()
    assert get_stereo_payload(molecule).atom_facts == ()
    assert get_stereo_payload(molecule).bond_facts == ()
    assert to_rdmol(molecule, kekulize=False).GetNumAtoms() == 0


def test_unsupported_namespace_code_and_direction_are_not_silently_dropped():
    _, molecule, _ = _round_trip("CC")
    payload = get_stereo_payload(molecule)

    set_stereo_payload(
        molecule,
        replace(
            payload,
            records=(
                replace(payload.records[0], source_namespace="future-toolkit"),
                *payload.records[1:],
            ),
        ),
    )
    with pytest.raises(ValueError, match="stereo namespace: future-toolkit"):
        to_rdmol(molecule, kekulize=False)

    set_stereo_payload(
        molecule,
        replace(
            payload,
            atom_facts=(
                replace(
                    payload.atom_facts[0],
                    source_namespace="future-toolkit",
                ),
                *payload.atom_facts[1:],
            ),
        ),
    )
    with pytest.raises(
        ValueError,
        match="atom transport namespace: future-toolkit",
    ):
        to_rdmol(molecule, kekulize=False)

    set_stereo_payload(
        molecule,
        replace(
            payload,
            records=(
                replace(payload.records[0], source_code="CHI_FUTURE"),
                *payload.records[1:],
            ),
        ),
    )
    with pytest.raises(ValueError, match="chiral-tag code: CHI_FUTURE"):
        to_rdmol(molecule, kekulize=False)

    set_stereo_payload(
        molecule,
        replace(
            payload,
            bond_facts=(BondTransportRecord((0, 1), "FUTURE_DIR"),),
        ),
    )
    with pytest.raises(ValueError, match="bond-direction code: FUTURE_DIR"):
        to_rdmol(molecule, kekulize=False)


def test_import_export_never_calls_assign_stereochemistry(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("AssignStereochemistry must not be called by Level 1")

    monkeypatch.setattr(Chem, "AssignStereochemistry", forbidden)
    source = Chem.MolFromSmiles("F/C=C/F")

    molecule = from_rdmol(source, Molecule())
    exported = to_rdmol(molecule, kekulize=False, sanitize=True)

    assert exported.GetBondWithIdx(1).GetStereo() == Chem.BondStereo.STEREOE


def test_import_clones_and_does_not_kekulize_or_rewrite_the_source():
    source = Chem.MolFromSmiles("c1ccccc1")
    before = [
        (bond.GetBondType(), bond.GetIsAromatic()) for bond in source.GetBonds()
    ]

    from_rdmol(source, Molecule())

    assert [
        (bond.GetBondType(), bond.GetIsAromatic()) for bond in source.GetBonds()
    ] == before
