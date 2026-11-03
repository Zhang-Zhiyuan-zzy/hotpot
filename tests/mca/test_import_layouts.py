def test_public_symbols():
    import hotpot.cheminfo.operations.inference.mca as mca

    assert mca.MCACpuPredictor
    assert mca.MCAGpuPredictor
    assert mca.mca
    assert mca.AtomPrediction.__module__ == "hotpot.cheminfo.operations.inference.mca.results"
    assert mca.MoleculePrediction.__module__ == "hotpot.cheminfo.operations.inference.mca.results"
    assert not hasattr(mca, "MCAPredictor")
    assert not hasattr(mca, "MoleculeGraph")
