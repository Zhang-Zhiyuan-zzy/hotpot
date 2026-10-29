"""Concrete Plan20 benchmark tasks; fixture preparation is outside timing."""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict


def prepare(entry, phase):
    import numpy as np
    from hotpot import Molecule

    name = entry["operation"]
    params = entry.get("parameters", {})
    size = params.get("atoms", 64)
    loops = params.get("loops", 20)

    def molecule(count=size):
        mol = Molecule()
        for index in range(count):
            mol.create_atom(atomic_number=6, formal_charge=0, coordinates=(index * 1.5, 0.0, 0.0))
        return mol

    def repeat(function, count=loops):
        def run():
            value = None
            for _ in range(count):
                value = function()
            return {"completed_calls": count, "value": value}
        return run

    if name == "empty":
        return repeat(lambda: len(Molecule().atoms)), lambda: None
    if name == "construct":
        return repeat(lambda: len(molecule().atoms)), lambda: None
    if name == "io":
        smiles = params["smiles"]
        fmt = params["format"]
        source = smiles if fmt == "smi" else Molecule.read_one(smiles, "smi").write(fmt=fmt)
        return repeat(lambda: len(Molecule.read_one(source, fmt).atoms)), lambda: None

    if name in {"scalar_read", "scalar_write", "batch_facts", "coordinates", "insert_delete",
                "matrix", "native", "partial_projection", "mca_projection", "write_read",
                "clean_read", "unrelated_read", "conformers"}:
        mol = molecule()
        atoms = tuple(mol.atoms)
        coords = np.arange(size * 3, dtype=np.float64).reshape(size, 3) / 10.0
        if name == "scalar_read":
            return repeat(lambda: sum(atom.atomic_number + atom.formal_charge for atom in atoms)), lambda: None
        if name == "scalar_write":
            def update():
                for atom in atoms:
                    atom.formal_charge = 0
                return size
            return repeat(update), lambda: None
        if name == "batch_facts":
            def update():
                if phase == "baseline":
                    for atom in atoms:
                        atom.formal_charge = 0
                else:
                    mol.updata_atoms(formal_charge=0)
                return size
            return repeat(update), lambda: None
        if name == "coordinates":
            def transform():
                mol.coordinates = coords + (0.25, -0.5, 0.75)
                return {"rows": len(mol.coordinates), "returned_bytes": size * 3 * 8}
            return repeat(transform), lambda: None
        if name == "insert_delete":
            def edit():
                atom = mol.create_atom(atomic_number=8, coordinates=(0.0, 1.0, 2.0))
                mol.remove_atom(atom)
                return len(mol.atoms)
            return repeat(edit), lambda: None
        if name == "matrix":
            def export():
                if phase == "baseline":
                    matrix = np.asarray([(a.atomic_number, a.formal_charge, *a.coordinates) for a in atoms], dtype=np.float32)
                else:
                    with mol._mutation_lock:
                        matrix = mol._atom_table.select(
                            ("atomic_number", "formal_charge", "x", "y", "z"), dtype="float32"
                        )
                return {"shape": list(matrix.shape), "returned_bytes": matrix.nbytes, "dtype": str(matrix.dtype)}
            return repeat(export), lambda: None
        if name == "native":
            module = "hotpot.cheminfo._native.buffers"
            native = importlib.import_module(module)
            fields = frozenset(getattr(native.NativeMoleculeField, field) for field in params["fields"])
            def pack():
                result = native.pack_molecule(mol, fields)
                buffers = [getattr(result, field.value) for field in fields]
                return {"rows": result.atom_count, "returned_bytes": sum(v.nbytes for v in buffers), "buffer_count": len(buffers)}
            return repeat(pack), lambda: None
        if name in {"partial_projection", "mca_projection"}:
            # Synthetic values characterize accessor cost, not model predictions.
            if phase != "baseline":
                raise NotImplementedError("Bind this unchanged synthetic fixture to the Stage7C result owner before candidate measurement")
            for index, atom in enumerate(atoms):
                if name == "partial_projection":
                    atom.partial_charge = index / max(size, 1)
                else:
                    object.__setattr__(atom, "_mca", float(index))
            if name == "mca_projection":
                object.__setattr__(mol, "_mca_sites", {atom: float(index) for index, atom in enumerate(atoms)})
                return repeat(lambda: {"all_atom_sum": sum(a.mca for a in atoms), "site_count": len(mol.mca_sites)}), lambda: None
            return repeat(lambda: sum(a.partial_charge for a in atoms)), lambda: None
        if name in {"write_read", "clean_read", "unrelated_read"}:
            element = [6]
            def access():
                if name != "clean_read":
                    element[0] = 7 if element[0] == 6 else 6
                    for atom in atoms:
                        atom.atomic_number = element[0]
                if name == "unrelated_read":
                    return float(mol.coordinates.sum())
                return sum(atom.implicit_hydrogens for atom in atoms)
            return repeat(access), lambda: None
        if name == "conformers":
            def frames():
                mol.conformer_clear()
                mol.conformer_add(coords=coords)
                mol.conformer_add(coords=coords + 1.0)
                result = mol.conformer_get(0)
                mol.conformer_load(1)
                count = mol.conformers_number
                mol.conformer_clear()
                return {"frames": count, "rows": len(result["coordinates"])}
            return repeat(frames), lambda: None

    if name == "graph_geometry":
        mol = Molecule.read_one(params["smiles"], "smi")
        mol.coordinates = np.arange(len(mol.atoms) * 3).reshape(-1, 3) / 10.0
        def query():
            return {"rings": len(mol.rings), "path": len(mol.shortest_path(mol.atoms[0], mol.atoms[-1])),
                    "angles": len(mol.angles), "torsions": len(mol.torsions), "pairs": len(mol.pair_dist),
                    "hits": len(mol.search_substructure("[#6]-[#6]"))}
        return repeat(query), lambda: None
    if name == "geometry_screen":
        module = "hotpot.cheminfo.geometry" if phase == "baseline" else "hotpot.cheminfo.rules.mathematics.geometry"
        geometry = importlib.import_module(module)
        cycle = geometry.Cycle(tuple((float(np.cos(i * np.pi / 3)), float(np.sin(i * np.pi / 3)), 0.0) for i in range(6)))
        segments = [geometry.Segment((x, 0.0, -2.0), (x, 0.0, 2.0)) for x in (0.0, 3.0)]
        return repeat(lambda: [geometry.determine_segment_cycle_relation(segment, cycle).state.value for segment in segments]), lambda: None
    if name == "crystal":
        module = "hotpot.cheminfo.crystal"
        Crystal = importlib.import_module(module).Crystal
        matrix = np.asarray(params["matrix"], dtype=float)
        return repeat(lambda: vars(Crystal.from_matrix(matrix))["a"]), lambda: None
    if name == "bundle":
        from hotpot import MolBundle
        mols = [molecule(8) for _ in range(params["molecules"])]
        return repeat(lambda: len(MolBundle(mols).list_mol)), lambda: None
    if name == "convert":
        mol = Molecule.read_one(params["smiles"], "smi")
        def convert():
            obmol = mol.to_obmol()
            rdmol = mol.to_rdmol()
            return {"ob_atoms": obmol.NumAtoms(), "rd_atoms": rdmol.GetNumAtoms(), "calls": 2}
        return repeat(convert), lambda: None
    if name == "conformer":
        from rdkit import Chem
        module = "hotpot.cheminfo.AImodels.mca.conformer" if phase == "baseline" else "hotpot.cheminfo.adapters.rdkit.conformer"
        generate = importlib.import_module(module).ensure_3d_conformer
        source = Chem.MolFromSmiles(params["smiles"])
        def conformer():
            result = generate(source, seed=params["seed"])
            coords = result.GetConformer().GetPositions()
            return {"atoms": result.GetNumAtoms(), "frames": result.GetNumConformers(), "finite_coordinates": bool(np.isfinite(coords).all())}
        return repeat(conformer, 1), lambda: None
    if name == "forcefield":
        def optimize():
            mol = Molecule.read_one(params["smiles"], "smi")
            report = mol.build3d(seed=params["seed"], epochs=params["epochs"], steps_per_epoch=params["steps_per_epoch"], quality_level="off")
            return {"atoms": len(mol.atoms), "finite_coordinates": bool(np.isfinite(mol.coordinates).all()),
                    "report": json.loads(json.dumps(asdict(report), default=str))}
        return repeat(optimize, 1), lambda: None
    if name == "memory":
        retained = []
        def retain():
            retained[:] = [molecule() for _ in range(params["molecules"])]
            return {"molecules": len(retained), "atoms": sum(len(m.atoms) for m in retained)}
        return repeat(retain, 1), retained.clear
    if name == "mca":
        module = "hotpot.cheminfo.AImodels.mca" if phase == "baseline" else "hotpot.cheminfo.operations.inference.mca"
        mca = importlib.import_module(module)
        smiles = [params["smiles"][index % len(params["smiles"])] for index in range(params["molecules"])]
        if params["backend"] == "cpu":
            predictor = mca.MCACpuPredictor(cpu_count=params["cpu_count"], batch_size=64, conformer_seed=params["seed"])
        else:
            predictor = mca.MCAGpuPredictor(gpu_indices=params["gpu_indices"], site_cpu_per_gpu=params["site_cpu_per_gpu"],
                                          infer_cpu_per_gpu=params["infer_cpu_per_gpu"], gpu_batch_size=2048, conformer_seed=params["seed"])
        predictor.__enter__()
        def predict():
            predictions = list(predictor.iter_predict(smiles))
            return {"requested_molecules": len(smiles), "returned_molecules": len(predictions),
                    "sites": sum(len(p.sites) for p in predictions), "atom_predictions": sum(len(p.atom_predictions) for p in predictions),
                    "gpu_started": getattr(predictor, "_gpu_started", False),
                    "resource_plan": asdict(predictor.resource_plan) if getattr(predictor, "resource_plan", None) else None,
                    "cpu_ids": list(getattr(predictor, "cpu_ids", ())),
                    "prediction_sum": sum(a.mca_kj_mol for p in predictions for a in p.atom_predictions)}
        return repeat(predict, 1), lambda: predictor.__exit__(None, None, None)
    if name == "cbond":
        module = "hotpot.cheminfo.AImodels.cbond.runtime" if phase == "baseline" else "hotpot.cheminfo.adapters.inference.runtime"
        runtime = importlib.import_module(module).CBondRuntime(device=params["device"])
        atoms = np.asarray([6, 7, 8, 26], dtype=np.int32)
        edges = np.asarray([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=np.int64)
        def predict():
            xg = runtime.embed_graph(atoms, edges)
            values = runtime.predict(xg, xg[[0, 1, 2]].reshape(1, 3, 128), np.zeros((1, 3), dtype=bool),
                                     np.asarray([[3, 3], [0, 1]], dtype=np.int64))
            return {"predictions": len(values), "sum": float(values.sum()), "device": runtime.device}
        run = repeat(predict)
        run.runtime_allocation = lambda: {
            "device": runtime.device,
            "sessions": {name: {"providers": session.get_providers(), "provider_options": session.get_provider_options()}
                         for name, session in (("graph", runtime.graph_session), ("cbond", runtime.cbond_session))},
        }
        return run, lambda: None
    raise ValueError(f"Unknown workload operation: {name}")
