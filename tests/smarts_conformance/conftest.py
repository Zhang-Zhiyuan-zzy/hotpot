def pytest_configure(config):
    config.addinivalue_line(
        "markers", "smarts_smoke: fast SMARTS infrastructure and core behavior"
    )
    config.addinivalue_line(
        "markers", "smarts_core: deterministic offline SMARTS conformance"
    )
    config.addinivalue_line(
        "markers",
        "smarts_known_failure: strict contract assertion for a documented defect",
    )
    config.addinivalue_line(
        "markers", "smarts_heavy: differential, fuzz, or performance audit"
    )
