import pytest

from hotpot.cheminfo.structure.molecule import Molecule
from hotpot.cheminfo import stereo


def test_molecule_constructor_assembles_an_explicit_empty_payload():
    first = Molecule()
    second = Molecule()

    first_payload = stereo.get_stereo_payload(first)
    second_payload = stereo.get_stereo_payload(second)

    assert first_payload == stereo.StereoPayload(1, (), (), ())
    assert second_payload == first_payload
    assert "_stereo_payload" not in first.__dict__
    assert "_stereo_payload" not in second.__dict__


def test_missing_stereo_storage_is_not_a_supported_legacy_shape():
    mol = Molecule()
    del mol._stereo_source
    with pytest.raises(AttributeError, match="_stereo_source"):
        stereo.get_stereo_payload(mol)


def test_storage_reassembles_the_exact_payload_without_interpretation():
    mol = Molecule()
    payload = stereo.StereoPayload(
        schema_version=47,
        records=(
            stereo.StereoRecord(
                anchor=stereo.StereoAnchor(
                    stereo.StereoAnchorKind.ATOM,
                    (81,),
                ),
                carriers=(),
                descriptor="unsupported-future-descriptor",
                group=None,
                source_namespace="future-toolkit",
                source_code="opaque-code",
            ),
        ),
        atom_facts=(),
        bond_facts=(),
    )

    stereo.set_stereo_payload(mol, payload)

    assert stereo.get_stereo_payload(mol) == payload
