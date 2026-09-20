"""Generate deterministic Relevant Cycle fixtures from the pinned RDL oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from corpus import golden_graph_cases
from oracle import RDLOracle

RDL_REPOSITORY = "https://github.com/rareylab/RingDecomposerLib.git"
RDL_COMMIT = "3a7ff93de0d9c4f6a5661508549c6063573f39c7"


def edge_digest(edges: Iterable[Sequence[int]]) -> str:
    """Return a stable digest of an ordered undirected edge collection."""
    normalized = sorted(
        (min(node1, node2), max(node1, node2)) for node1, node2 in edges
    )
    payload = ";".join(f"{node1},{node2}" for node1, node2 in normalized)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("golden_relevant_cycles.json"),
    )
    arguments = parser.parse_args()

    oracle = RDLOracle(arguments.library)
    cases = []
    for name, edges in golden_graph_cases().items():
        cycles = oracle.relevant_cycles(edges, max_cycles=None)
        cases.append(
            {
                "name": name,
                "edge_sha256": edge_digest(edges),
                "node_count": len({node for edge in edges for node in edge}),
                "edge_count": len(edges),
                "cycles": cycles,
            }
        )

    payload = {
        "schema_version": 1,
        "oracle": {
            "name": "RingDecomposerLib",
            "repository": RDL_REPOSITORY,
            "commit": RDL_COMMIT,
            "api": "RDL_getRCyclesIterator",
        },
        "cases": cases,
    }
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
