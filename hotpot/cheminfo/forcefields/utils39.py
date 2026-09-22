"""Open Babel 3.1 worker adapters used by the Python 3.9 façade."""

from __future__ import annotations

import ctypes
import os
from multiprocessing.connection import Connection
from typing import Optional, TYPE_CHECKING

from openbabel import openbabel as ob

from . import utils as _utils


if TYPE_CHECKING:
    from ..core import Molecule


def _seed_openbabel_random(seed: int) -> None:
    """Seed the legacy Open Babel 3.1 process-local random generators."""
    os.environ["OB_RANDOM_SEED"] = str(seed)
    probe = ob.vector3()
    probe.randomUnitVector()
    process_c_library = ctypes.CDLL(None)
    process_c_library.srand.argtypes = (ctypes.c_uint,)
    process_c_library.srand.restype = None
    process_c_library.srand(ctypes.c_uint(seed))


def _build_ligand_proxies_worker(
    mol: "Molecule",
    connection: Connection,
    candidate_count: Optional[int],
    max_attempts: int,
    candidate_warmup_steps: int,
    candidate_score_steps: int,
    best_candidate_refine_steps: int,
    effective_forcefield: str,
    seed: Optional[int],
    ligand_untangling_attempts: int = 20,
    perturb_sigma: float = 0.5,
) -> None:
    """Run the shared ligand-proxy worker with the Open Babel 3.1 RNG."""
    _utils._run_ligand_proxy_worker(
        mol,
        connection,
        candidate_count,
        max_attempts,
        candidate_warmup_steps,
        candidate_score_steps,
        best_candidate_refine_steps,
        effective_forcefield,
        seed,
        ligand_untangling_attempts,
        perturb_sigma,
        seed_initializer=_seed_openbabel_random,
    )


def _seeded_ob_build_worker(
    mol: "Molecule",
    connection: Connection,
    seed: int,
) -> None:
    """Run the shared OBBuilder worker with the Open Babel 3.1 RNG."""
    _utils._run_seeded_ob_build_worker(
        mol,
        connection,
        seed,
        seed_initializer=_seed_openbabel_random,
    )
