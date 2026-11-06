"""Audit the maintained source-distribution contents."""

import argparse
from pathlib import Path
import tarfile


expected = {
    "hotpot/commands/models.py",
    "hotpot/commands/mca.py",
    "hotpot/commands/mca.md",
    "hotpot/commands/cbond.py",
    "hotpot/commands/cbond.md",
    "hotpot/cheminfo/operations/inference/mca/attachment.py",
    "hotpot/cheminfo/operations/inference/mca/results.py",
    "hotpot/cheminfo/operations/inference/mca/MODEL_CARD.md",
    "hotpot/cheminfo/operations/inference/cbond/runtime.py",
    "hotpot/cheminfo/rules/chemistry/mca_sites/__init__.py",
    "hotpot/cheminfo/adapters/inference/cbond.py",
    "hotpot/cheminfo/adapters/inference/runtime.py",
    "hotpot/cheminfo/adapters/rdkit/conformers.py",
    "hotpot/cheminfo/adapters/rdkit/sampling.py",
    "hotpot/cheminfo/operations/conformers/contracts.py",
    "hotpot/cheminfo/operations/conformers/openbabel.py",
    "hotpot/cheminfo/operations/conformers/rdkit.py",
    "tools/model_deployment/cbond/make_dynamic_onnx.py",
    "tools/model_deployment/cbond/models.py",
    "hotpot/cheminfo/operations/inference/artifacts/__init__.py",
    "hotpot/cheminfo/operations/inference/artifacts/resolver.py",
    "hotpot/cheminfo/adapters/inference/artifacts.py",
    "hotpot/cheminfo/operations/inference/artifacts/data/mca/manifest.json",
    "hotpot/cheminfo/operations/inference/artifacts/data/mca/README.md",
    "hotpot/cheminfo/operations/inference/artifacts/data/cbond/manifest.json",
    "hotpot/cheminfo/operations/inference/artifacts/data/cbond/README.md",
    "hotpot/_cli.py",
    "hotpot/pipeline/cli.py",
    "hotpot/pipeline/cli_doc.md",
    "hotpot/commands/ff.py",
    "hotpot/commands/ff.md",
    "hotpot/cheminfo/operations/forcefields/__init__.py",
    "hotpot/cheminfo/operations/forcefields/ff.py",
    "hotpot/cheminfo/operations/calculation/scientific_state/execution_selection.py",
    "hotpot/cheminfo/rules/chemistry/scientific_state/data/profiles/coordination_screening_v1.json",
    "hotpot/cheminfo/adapters/native/substructure/_search_native.pyi",
    "hotpot/cheminfo/rules/mathematics/graph/_graph_native.pyi",
    "hotpot/cheminfo/rules/mathematics/graph/_native/graph.cpp",
    "hotpot/cheminfo/rules/mathematics/graph/_native/graph.hpp",
    "hotpot/cheminfo/rules/mathematics/graph/_native/cycle_basis.cpp",
    "hotpot/cheminfo/rules/mathematics/graph/_native/cycle_basis.hpp",
    "hotpot/cheminfo/adapters/native/substructure/_native/bindings.cpp",
    "hotpot/cheminfo/rules/mathematics/graph_matching/_native/compatibility.hpp",
    "hotpot/cheminfo/rules/mathematics/graph_matching/_native/matcher.cpp",
    "hotpot/cheminfo/rules/mathematics/graph_matching/_native/matcher.hpp",
    "hotpot/cheminfo/rules/mathematics/graph_matching/_native/automorphism.cpp",
    "hotpot/cheminfo/rules/mathematics/graph_matching/_native/automorphism.hpp",
    "hotpot/cheminfo/rules/chemistry/substructure/_native/predicates.cpp",
    "hotpot/cheminfo/rules/chemistry/substructure/_native/predicates.hpp",
    "hotpot/cheminfo/adapters/native/substructure/_native/production.cpp",
    "hotpot/cheminfo/adapters/native/substructure/_native/production.hpp",
    "hotpot/plugins/xtb/cli.py",
    "hotpot/plugins/xtb/cli_doc.md",
    "hotpot/plugins/xtb/README.md",
    "hotpot/plugins/xtb/stage.py",
    "hotpot/plugins/xtb/workflow.py",
    "hotpot/pipeline/README.md",
    "hotpot/pipeline/state_recovery.py",
}

obsolete = {
    "hotpot/cheminfo/AImodels/artifacts/__init__.py",
    "hotpot/cheminfo/AImodels/artifacts/errors.py",
    "hotpot/cheminfo/AImodels/artifacts/manifest.py",
    "hotpot/cheminfo/AImodels/artifacts/registry.py",
    "hotpot/cheminfo/AImodels/artifacts/resolver.py",
    "hotpot/cheminfo/AImodels/artifacts/cli.py",
    "hotpot/cheminfo/AImodels/mca/models/manifest.json",
    "hotpot/cheminfo/AImodels/mca/models/README.md",
    "hotpot/cheminfo/AImodels/cbond/onnx/manifest.json",
    "hotpot/cheminfo/AImodels/cbond/onnx/README.md",
    "hotpot/cheminfo/forcefields/ff39.py",
    "hotpot/cheminfo/forcefields/utils.py",
    "hotpot/cheminfo/forcefields/utils39.py",
    "hotpot/cheminfo/_io/xtb.py",
    "hotpot/plugins/xtb/core.py",
    "hotpot/plugins/xtb/xtb_doc.md",
    "hotpot/cheminfo/search/_search_native.pyi",
    "hotpot/cheminfo/search/_native/bindings.cpp",
    "hotpot/cheminfo/search/_native/compatibility.hpp",
    "hotpot/cheminfo/search/_native/matcher.cpp",
    "hotpot/cheminfo/search/_native/matcher.hpp",
    "hotpot/cheminfo/search/_native/production.cpp",
    "hotpot/cheminfo/search/_native/production.hpp",
    "hotpot/cheminfo/search/_native/graph_data.cpp",
    "hotpot/cheminfo/search/_native/graph_data.hpp",
}


def audit_sdist(sdist: Path) -> None:
    with tarfile.open(sdist) as archive:
        members = {
            "/".join(Path(name).parts[1:])
            for name in archive.getnames()
        }
    assert expected <= members
    assert obsolete.isdisjoint(members)
    assert not any(name.startswith((
        "hotpot/cheminfo/AImodels/", "hotpot/cheminfo/calculator/",
        "hotpot/cheminfo/conformers/",
    )) for name in members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()
    audit_sdist(args.sdist)
    print(f"source distribution audit passed: {args.sdist}")


if __name__ == "__main__":
    main()
