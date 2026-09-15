"""Determinism and provenance tests for the bounded fuzz generators."""

import random

import pytest

from hotpot.cheminfo.search.search import Substructure

from .fuzz.generators import (
    generate_structured_stress_smarts,
    generate_valid_smarts,
    mutate_valid_smarts,
)


pytestmark = pytest.mark.smarts_core


def test_invalid_generator_records_a_single_mutation_of_its_valid_seed():
    rng = random.Random(20260915)
    for _ in range(50):
        valid = generate_valid_smarts(rng)
        mutation, base, invalid, position, payload = mutate_valid_smarts(rng, valid)

        assert base == valid
        assert mutation
        assert invalid == base[:position] + payload + base[position:]


def test_generators_are_reproducible_from_seed():
    def sequence():
        rng = random.Random(41)
        mutations = tuple(
            mutate_valid_smarts(rng, generate_valid_smarts(rng)) for _ in range(20)
        )
        structured = tuple(
            generate_structured_stress_smarts(rng, max_depth=8) for _ in range(100)
        )
        return mutations, structured

    first = sequence()
    assert first == sequence()

    unit_counter = {
        "branch": lambda text: text.count("("),
        "recursive": lambda text: text.count("$("),
        "logic": lambda text: text.count("#6,#7"),
        "rings": lambda text: text.count("C1CCCCC1"),
    }
    structured = first[1]
    assert {kind for kind, _ in structured} == set(unit_counter)
    for kind, smarts in structured:
        assert 2 <= unit_counter[kind](smarts) <= 8
        Substructure.from_smarts(smarts)
