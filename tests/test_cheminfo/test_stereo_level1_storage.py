from hotpot.cheminfo import stereo


class _MoleculeStub:
    pass


def test_uninitialized_molecules_share_the_immutable_empty_payload():
    first = _MoleculeStub()
    second = _MoleculeStub()

    first_payload = stereo.get_stereo_payload(first)
    second_payload = stereo.get_stereo_payload(second)

    assert first_payload is second_payload
    assert first_payload is stereo._EMPTY_STEREO_PAYLOAD
    assert first_payload == stereo.StereoPayload(1, (), (), ())


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
