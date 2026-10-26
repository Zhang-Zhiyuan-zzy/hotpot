import pytest

from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo import stereo


class _MoleculeStub:
    pass


def test_molecule_constructor_owns_an_explicit_empty_payload():
    first = Molecule()
    second = Molecule()

    first_payload = stereo.get_stereo_payload(first)
    second_payload = stereo.get_stereo_payload(second)

    assert first_payload == stereo.StereoPayload(1, (), (), ())
    assert second_payload == first_payload
    assert second_payload is not first_payload


def test_missing_stereo_storage_is_not_a_supported_legacy_shape():
    with pytest.raises(AttributeError, match="_stereo_payload"):
        stereo.get_stereo_payload(_MoleculeStub())


def test_storage_keeps_the_exact_payload_without_interpretation_or_copying():
    mol = _MoleculeStub()
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

    assert stereo.get_stereo_payload(mol) is payload
