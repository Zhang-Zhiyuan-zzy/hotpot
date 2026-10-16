"""Public calculator entry points."""

from .base import Calculator
from .formal_charges import formal_charge, infer_charge
from .electronic_state.resolver import resolve_electronic_state
from .electronic_state.spin import infer_lowest_spin
from .contracts import (
    MolecularFacts,
    ScientificOperationRequirements,
    StatePolicy,
)
from .mca_inference import mca
from .molecular_charge import MolChargeCalculator
from .state_resolver import (
    ScientificStateBundle,
    StateRequest,
    extract_molecular_facts,
    resolve_scientific_state,
)

__all__ = [
    "Calculator",
    "MolChargeCalculator",
    "formal_charge",
    "infer_charge",
    "infer_lowest_spin",
    "mca",
    "resolve_electronic_state",
    "extract_molecular_facts",
    "resolve_scientific_state",
    "MolecularFacts",
    "StateRequest",
    "StatePolicy",
    "ScientificOperationRequirements",
    "ScientificStateBundle",
]

