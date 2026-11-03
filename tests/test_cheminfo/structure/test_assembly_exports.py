"""One assembly owner and its intentionally small public surface."""

import importlib.util
import inspect
from importlib.resources import files
import json

import hotpot
import hotpot.cheminfo as cheminfo
from hotpot.cheminfo.operations import assembly


CLASSES = (
    "Fragment", "EdgeShoulder", "AtomLink", "AlkylGraft", "BondAdding",
    "AtomReplace", "RingWedge", "AssembleFactory",
)
ACTIONS = (
    "alkyl_generator", "atom_link_atom_action", "shoulder_bond_action",
    "bond_order_add", "atom_replace", "ring_wedge",
)


def test_assembly_exports_share_their_one_canonical_definition():
    assert tuple(assembly.__all__) == CLASSES + ACTIONS
    for name in CLASSES:
        value = getattr(assembly, name)
        assert getattr(cheminfo, name) is getattr(hotpot, name) is value
        assert value.__module__.startswith("hotpot.cheminfo.operations.assembly.")
    for name in ACTIONS:
        assert callable(getattr(assembly, name))
        assert name not in cheminfo.__all__
        assert not hasattr(hotpot, name)
    for name in ("MolBatch", "ActionFuncTemplate", "DefaultAction", "ComplexStatistics"):
        assert not hasattr(hotpot, name)
        assert name not in cheminfo.__all__


def test_retired_assembly_and_incomplete_chemistry_statistics_have_no_runtime_owner():
    assert importlib.util.find_spec("hotpot.cheminfo.mol_assemble") is None
    assert importlib.util.find_spec("hotpot.cheminfo.mol_statistics") is None
    # The independent CCDC implementation is not part of this retirement.
    assert importlib.util.find_spec("hotpot.plugins.ccdc_api.statistics") is not None


def test_default_template_has_one_packaged_owner_and_its_original_definitions():
    resources = files(assembly)
    definitions = json.loads(resources.joinpath("data/fragment_template.json").read_text())
    assert len(definitions) == 26
    assert {row["method"] for row in definitions} == {
        "EdgeShoulder", "AtomLink", "AtomReplace", "BondAdding", "AlkylGraft",
    }
    for name in ("README.md", "Assemblers.svg", "Example-5step.svg"):
        assert resources.joinpath(name).is_file()


def test_advanced_actions_preserve_parameter_annotations_and_call_shape():
    for name in ACTIONS[1:]:
        action = getattr(assembly, name)
        parameters = inspect.signature(action).parameters
        assert tuple(parameters) == ("mol", "hit", "frag", "action_points")
        assert tuple(parameter.annotation for parameter in parameters.values()) == (
            cheminfo.Molecule, list[int], cheminfo.Molecule, list[int],
        )
        assert all(parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                   and parameter.default is inspect.Parameter.empty
                   for parameter in parameters.values())
        assert inspect.signature(action).return_annotation is inspect.Signature.empty
