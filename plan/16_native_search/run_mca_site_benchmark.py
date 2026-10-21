"""Reproduce the Plan 16 one-worker MCA site-search performance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import hotpot as hp

from hotpot.cheminfo.AImodels.mca.site_detection import (
    find_nucleophilic_sites,
)


SMILES = (
    "COC",
    "CC(=O)C",
    "CC(=O)N",
    "C=C[O-]",
    "CC=O",
    "CC=N",
    "[NH-]C",
    "[CH2-]C",
    "C=[N+]([O-])[O-]",
    "CC(=O)OC",
    "CC(=O)O",
    "CCN",
    "C=C=[N-]",
    "CC#N",
    "[C-]#[N+]C",
    "Oc1ccccc1",
    "CO[Si](C)(C)C",
    "n1ccccc1",
    "[O-]",
    "C=C",
    "C=[N+](C)C",
    "C#C",
    "C#[S+]",
    "CS",
)


def _detect_corpus(molecules):
    return tuple(
        tuple(
            (site.atom_index, site.site_type)
            for site in find_nucleophilic_sites(molecule)
        )
        for molecule in molecules
    )


def _digest(value) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def run_benchmark(*, warmups: int, repeats: int) -> dict:
    molecules = tuple(hp.read_mol(smiles, "smi") for smiles in SMILES)
    for _ in range(warmups):
        _detect_corpus(molecules)

    samples = []
    results = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = _detect_corpus(molecules)
        samples.append(time.perf_counter() - started)
        results.append(result)

    digests = tuple(_digest(result) for result in results)
    if len(set(digests)) != 1:
        raise RuntimeError("MCA site detection was not deterministic")

    median = statistics.median(samples)
    return {
        "schema_version": 1,
        "scope": "24 prebuilt molecules; site search and symmetry filtering",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "method": {
            "clock": "time.perf_counter",
            "warmups": warmups,
            "repeats": repeats,
            "workers": 1,
        },
        "molecule_count": len(molecules),
        "samples_s": samples,
        "median_s": median,
        "molecules_per_s": len(molecules) / median,
        "result_sha256": digests[0],
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "sites": results[0],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = run_benchmark(warmups=args.warmups, repeats=args.repeats)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
