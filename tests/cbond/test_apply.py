from hotpot import read_mol
from hotpot.cheminfo.AImodels.cbond.apply import auto_build_cbond


def test_auto_build_cbond_never_materializes_the_metal_self_loop(runtime):
    result, probability = auto_build_cbond(
        read_mol("CN"),
        "Eu",
        runtime=runtime,
    )

    metal = result.metals[0]
    assert metal.neighbours
    assert all(bond.atom1 is not bond.atom2 for bond in result.bonds)
    assert 0.0 < probability <= 1.0
