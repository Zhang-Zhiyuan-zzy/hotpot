from __future__ import annotations

import inspect

from hotpot.cheminfo import rdWrappers


def test_public_exports_are_explicit():
    assert rdWrappers.__all__ == (
        "ConformerBatch",
        "ConformerRecord",
        "CoordinationEmbeddingConstraints",
        "ETKDGOptions",
        "ETKDGSampler",
        "GenerationStatus",
        "RuntimeInfo",
        "generate_conformers",
        "runtime_info",
    )


def test_value_contracts_are_frozen():
    for contract in (
        rdWrappers.ETKDGOptions,
        rdWrappers.CoordinationEmbeddingConstraints,
        rdWrappers.ConformerRecord,
        rdWrappers.ConformerBatch,
        rdWrappers.RuntimeInfo,
    ):
        assert contract.__dataclass_params__.frozen is True


def test_etkdg_option_signature_is_keyword_only():
    parameters = inspect.signature(rdWrappers.ETKDGOptions).parameters
    assert tuple(parameters) == (
        "conformer_count",
        "seed",
        "prune_rms_threshold",
        "use_random_coordinates",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )
    assert tuple(parameter.default for parameter in parameters.values()) == (
        1,
        None,
        None,
        False,
    )


def test_sampler_signatures_are_stable():
    constructor = inspect.signature(rdWrappers.ETKDGSampler).parameters
    assert tuple(constructor) == ("options",)
    assert constructor["options"].default is None

    generate = inspect.signature(rdWrappers.ETKDGSampler.generate).parameters
    assert tuple(generate) == ("self", "mol", "constraints")
    assert generate["constraints"].kind is inspect.Parameter.KEYWORD_ONLY
    assert generate["constraints"].default is None

    function = inspect.signature(rdWrappers.generate_conformers).parameters
    assert tuple(function) == ("mol", "options", "constraints")
    assert function["options"].kind is inspect.Parameter.KEYWORD_ONLY
    assert function["options"].default is None
    assert function["constraints"].kind is inspect.Parameter.KEYWORD_ONLY
    assert function["constraints"].default is None


def test_sampler_default_is_the_frozen_default_contract():
    sampler = rdWrappers.ETKDGSampler()
    assert sampler.options == rdWrappers.ETKDGOptions()
