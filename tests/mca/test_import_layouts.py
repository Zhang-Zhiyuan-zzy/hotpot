def test_public_symbols():
    import mca

    assert mca.MCACpuPredictor
    assert mca.MCAGpuPredictor
    assert not hasattr(mca, "MCAPredictor")
    assert not hasattr(mca, "MoleculeGraph")
