# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : core
 Created   : 2025/8/29 17:04
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""

import os
import re
import time
import shutil
import tempfile
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Union, Sequence, Tuple, Set, Any
from concurrent import futures
from itertools import product

import pandas as pd
import numpy as np


def _which(exe: str) -> Optional[str]:
    return shutil.which(exe)


def _run_subprocess(cmd: List[str], cwd: Optional[Union[str, Path]] = None, timeout: Optional[int] = None) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _ensure_out_dir(out_dir: Optional[Union[str, Path]]) -> Path:
    if out_dir is None:
        return Path(tempfile.mkdtemp(prefix="zeopp_out_"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _parse_key_value_pairs(line: str) -> Dict[str, float]:
    """
    Parse key:value numeric pairs on a line.
    Example: 'ASA_A^2: 60.7713 ASA_m^2/cm^3: 1976.4 ASA_m^2/g: 1218.21'
    """
    kv: Dict[str, float] = {}
    pattern = re.compile(r"([A-Za-z0-9_\^\/]+):\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")
    for k, v in pattern.findall(line):
        try:
            kv[k] = float(v)
        except ValueError:
            # skip non-numeric values gracefully
            pass
    return kv


def _extract_first_floats(line: str, count: Optional[int] = None) -> List[float]:
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?", line)
    vals = [float(x) for x in nums]
    return vals if count is None else vals[:count]


def _parse_res_file(path: Union[str, Path]) -> Dict[str, float]:
    """
    .res contains: 'filename Di Df Dif'
    Returns a dict labeled as {'GLD','LCD','PLD'} mapped to the first three numbers found.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip().splitlines()[0].split()
        assert len(content) == 4, f"Could not parse .res file: {path}"
        vals = list(map(float, content[1:]))
        return {"GCD": vals[0], "PLD": vals[1], "LCD": vals[2]}


def _parse_chan_file(path: Union[str, Path]) -> Tuple[Optional[int], List[Dict[str, float]]]:
    """
    .chan example:
      EDI.chan   1 channels identified of dimensionality 1
      Channel  0  4.89082  3.03868  4.89082
    Returns (dimensionality, list of channel dicts: {'channel_id', 'Di_A', 'Df_A', 'Dif_A'})
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    dimensionality: Optional[int] = None
    channels: List[Dict[str, float]] = []
    dim_match = re.search(r"dimensionality\s+(\d+)", lines[0]) if lines else None
    if dim_match:
        dimensionality = int(dim_match.group(1))
    for ln in lines:
        if ln.lower().startswith("channel"):
            parts = ln.split()
            if len(parts) >= 5:
                ch_id = int(parts[1])
                vals = _extract_first_floats(" ".join(parts[2:]))
                if len(vals) >= 3:
                    channels.append(
                        {
                            "channel_id": ch_id,
                            "Di_A": vals[0],
                            "Df_A": vals[1],
                            "Dif_A": vals[2],
                        }
                    )
    return dimensionality, channels


def _parse_sa_vol_like_file(path: Union[str, Path]) -> Dict[str, float]:
    """
    Parse files like .sa, .vol, .volpo with:
      - main summary line starting with '@ ' containing key:value pairs
      - optional extra lines about channels/pockets
    Returns a flattened dict of metrics.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    out: Dict[str, float] = {}
    for ln in lines:
        if ln.startswith("@"):
            out.update(_parse_key_value_pairs(ln))
            break
    for ln in lines:
        if "Number_of_channels" in ln:
            out.update(_parse_key_value_pairs(ln))
        elif "Channel_surface_area_A^2" in ln or "Channel_volume_A^3" in ln:
            out.update(_parse_key_value_pairs(ln))
        elif "Number_of_pockets" in ln:
            out.update(_parse_key_value_pairs(ln))
        elif "Pocket_surface_area_A^2" in ln or "Pocket_volume_A^3" in ln:
            out.update(_parse_key_value_pairs(ln))
    return out


def _parse_psd_file(path: Union[str, Path]) -> pd.DataFrame:
    """
    PSD histogram file: parse whitespace-separated numeric columns.
    Output columns:
      - If 4 columns: ['Bin(A)', 'Count', 'CumDist', 'DerivDist']
      - Otherwise: generic ['col1', 'col2', ...]
    Ignores comment lines starting with '#' or '@'.
    """
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.startswith("@"):
                continue
            parts = ln.split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            rows.append(vals)
    if not rows:
        return pd.DataFrame()
    ncol = max(len(r) for r in rows)
    if ncol == 4:
        cols = ["Bin(A)", "Count", "CumDist", "DerivDist"]
    else:
        warnings.warn(
            f"PSD file has {ncol} columns (expected 4); using generic column names.",
            RuntimeWarning,
            stacklevel=2,
        )
        cols = [f"col{i}" for i in range(1, ncol + 1)]
    padded = [r + [None] * (ncol - len(r)) for r in rows]
    return pd.DataFrame(padded, columns=cols)


# -------------- MpRun func ----------------
def _sum_wrapper(
    self: "ZeoppRunner",
    struct: Union[str, Path],
    pr: float,
    sa_samples_per_atom: int = 2000,
    vol_samples_per_uc: int = 50000,
    psd_samples_per_uc: Optional[int] = None,
    chan_radius_for_sa_vol: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}

    if "res" in self.enabled_tasks:
        result["res"] = self.run_res(struct, _apply_units=False)

    if "chan" in self.enabled_tasks:
        result["chan"] = self.run_chan(struct, pr, _apply_units=False)

    if "sa" in self.enabled_tasks:
        result["sa"] = self.run_sa(
            struct, pr, sa_samples_per_atom, chan_radius=chan_radius_for_sa_vol, _apply_units=False
        )

    if "vol" in self.enabled_tasks:
        result["vol"] = self.run_vol(
            struct, pr, vol_samples_per_uc, chan_radius=chan_radius_for_sa_vol, _apply_units=False
        )

    if "volpo" in self.enabled_tasks:
        result["volpo"] = self.run_volpo(
            struct, pr, vol_samples_per_uc, chan_radius=chan_radius_for_sa_vol, _apply_units=False
        )

    if "psd" in self.enabled_tasks:
        samples = psd_samples_per_uc if psd_samples_per_uc is not None else vol_samples_per_uc
        result["psd"] = self.run_psd(
            struct, pr, samples, chan_radius=chan_radius_for_sa_vol, _apply_units=False
        )

    return result


class ZeoppRunner:
    """
    A Python wrapper for Zeo++ 'network' binary to compute structural descriptors
    and parse results into pandas DataFrames.
    """

    def __init__(
        self,
        network_bin: str = "network",
        out_dir: Optional[Union[str, Path]] = None,
        ha: bool = True,
        nor: bool = False,
        radii_file: Optional[Union[str, Path]] = None,
        mass_file: Optional[Union[str, Path]] = None,
        strip_atom_names: bool = False,
        extra_args: Optional[Sequence[str]] = None,
        timeout: Optional[int] = None,
        enabled_tasks: Optional[Sequence[str]] = None,
        units_as_rows: bool = False,
        cleanup_outputs: bool = False,
    ):
        """
        Parameters
        ----------
        network_bin : path or name of 'network' binary
        out_dir : where to write outputs (defaults to a temp directory)
        ha : add '-ha' for high-accuracy as recommended
        nor : use '-nor' (point particle Voronoi)
        radii_file : custom radii file ('-r')
        mass_file : custom mass file ('-mass')
        strip_atom_names : use '-stripatomnames'
        extra_args : any extra flags to pass to all runs
        timeout : per-process timeout in seconds
        enabled_tasks : default set of tasks executed in summarize()
            Allowed: {'res','chan','sa','vol','volpo','psd'}.
            None → {'res','chan','sa','vol','volpo'} (PSD off by default).
        units_as_rows : if True, split variable/unit and insert two header rows as first two rows.
                        This resets the index to columns to show metadata rows.
        cleanup_outputs : if True, remove Zeo++ output files after successful parsing.
        """
        resolved = _which(network_bin)
        if resolved is None:
            raise FileNotFoundError(
                f"Could not find Zeo++ 'network' binary: {network_bin}. "
                f"Add it to PATH or provide absolute path."
            )
        self.network_bin = resolved
        self.out_dir = _ensure_out_dir(out_dir)
        self.ha = ha
        self.nor = nor
        self.radii_file = Path(radii_file) if radii_file else None
        self.mass_file = Path(mass_file) if mass_file else None
        self.strip_atom_names = strip_atom_names
        self.extra_args = list(extra_args) if extra_args else []
        self.timeout = timeout

        allowed = {"res", "chan", "sa", "vol", "volpo", "psd"}
        default_tasks = {"res", "chan", "sa", "vol", "volpo"}
        if enabled_tasks is None:
            self.enabled_tasks = set(default_tasks)
        else:
            tasks_set = {t.lower() for t in enabled_tasks}
            unknown = tasks_set - allowed
            if unknown:
                raise ValueError(f"Unknown task(s) in enabled_tasks: {sorted(unknown)}. Allowed: {sorted(allowed)}")
            self.enabled_tasks = tasks_set

        self.units_as_rows = bool(units_as_rows)
        self.cleanup_outputs = bool(cleanup_outputs)

    # -------------------- helpers --------------------

    def _base_args(self) -> List[str]:
        args = [self.network_bin]
        if self.ha:
            args.append("-ha")
        if self.strip_atom_names:
            args.append("-stripatomnames")
        if self.nor:
            args.append("-nor")
        if self.radii_file:
            args.extend(["-r", str(self.radii_file)])
        if self.mass_file:
            args.extend(["-mass", str(self.mass_file)])
        args.extend(self.extra_args)
        return args

    def _run(self, args: List[str], cwd: Optional[Union[str, Path]] = None) -> None:
        cmd = self._base_args() + args
        _run_subprocess(cmd, cwd=cwd, timeout=self.timeout)

    @staticmethod
    def _split_var_unit(name: str) -> Tuple[str, str]:
        """
        Heuristic: use the last '_' part as unit if it looks like one.
        Recognize A, A^2, A^3, m^2/g, m^2/cm^3, cm^3/g, g/cm^3, etc.
        """
        if not isinstance(name, str):
            return str(name), ""
        head, sep, tail = name.rpartition("_")
        if not sep:
            return name, ""
        units_whitelist = {
            "A",
            "A^2",
            "A^3",
            "m^2/g",
            "m^2/cm^3",
            "cm^3/g",
            "g/cm^3",
            "cm^3/cm^3",
        }
        if tail in units_whitelist or ("/" in tail) or ("^" in tail):
            return head, tail
        return name, ""

    def _apply_units_rows(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        If units_as_rows=True:
          - reset index to columns
          - insert two rows at the top: variables and units
        Otherwise return df unchanged.
        """
        if not self.units_as_rows or df is None or df.empty:
            return df
        df_reset = df.reset_index()
        cols = list(df_reset.columns)
        var_row: Dict[str, Any] = {}
        unit_row: Dict[str, Any] = {}
        for c in cols:
            var, unit = self._split_var_unit(str(c))
            var_row[c] = var
            unit_row[c] = unit
        meta = pd.DataFrame([var_row, unit_row], index=["__var__", "__unit__"])
        out = pd.concat([meta, df_reset], ignore_index=False)
        # Use a simple RangeIndex to keep order.
        out = out.reset_index(drop=True)
        return out

    def _cleanup(self, paths: Sequence[Union[str, Path]]) -> None:
        if not self.cleanup_outputs:
            return
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                # best-effort cleanup
                pass

    def _derive_base(self, structure: Union[str, Path]) -> str:
        st = Path(structure)
        return st.stem

    def _unique_outfile(self, base: str, probe_radius: Optional[float], ext: str) -> Path:
        """
        Generate a unique output file path to avoid collisions under parallel runs.
        """
        ts = time.time_ns()
        probe_part = "" if probe_radius is None else f"-{probe_radius}"
        return self.out_dir / f"{base}{probe_part}-{ts}.{ext}"

    # -------------- Calculations and parsers -----------------

    def run_res(self, structure: Union[str, Path], _apply_units: bool = True) -> pd.DataFrame:
        """
        Compute pore diameters (Di, Df, Dif) via -res
        """
        base = self._derive_base(structure)
        out_path = self._unique_outfile(base, None, "res")
        args = ["-res", str(out_path), str(structure)]
        self._run(args)
        data = _parse_res_file(out_path)
        df = pd.DataFrame([{**data, "structure": base}]).set_index("structure")
        self._cleanup([out_path])
        return self._apply_units_rows(df) if _apply_units else df

    def run_chan(self, structure: Union[str, Path], probe_radius: float, _apply_units: bool = True) -> pd.DataFrame:
        """
        Channel identification via -chan probe_radius
        """
        base = self._derive_base(structure)
        out_path = self._unique_outfile(base, probe_radius, "chan")
        args = ["-chan", str(probe_radius), str(out_path), str(structure)]
        self._run(args)
        dimensionality, channels = _parse_chan_file(out_path)
        self._cleanup([out_path])
        if not channels:
            df = (
                pd.DataFrame(
                    columns=[
                        "structure",
                        "probe_A",
                        "channel_id",
                        "dimensionality",
                        "Di_A",
                        "Df_A",
                        "Dif_A",
                    ]
                )
                .set_index(["structure", "probe_A", "channel_id"])
            )
            return self._apply_units_rows(df) if _apply_units else df
        for ch in channels:
            ch["structure"] = base
            ch["probe_A"] = float(probe_radius)
            ch["dimensionality"] = dimensionality
        df = pd.DataFrame(channels).set_index(["structure", "probe_A", "channel_id"])
        return self._apply_units_rows(df) if _apply_units else df

    def run_sa(
        self,
        structure: Union[str, Path],
        probe_radius: float,
        samples_per_atom: int,
        chan_radius: Optional[float] = None,
        _apply_units: bool = True,
    ) -> pd.DataFrame:
        """
        Accessible surface area via -sa chan_radius probe_radius num_samples
        """
        base = self._derive_base(structure)
        cr = probe_radius if chan_radius is None else chan_radius
        out_path = self._unique_outfile(base, probe_radius, "sa")
        args = ["-sa", str(cr), str(probe_radius), str(int(samples_per_atom)), str(out_path), str(structure)]
        self._run(args)
        data = _parse_sa_vol_like_file(out_path)
        row = {
            "structure": base,
            "chan_radius_A": float(cr),
            "probe_A": float(probe_radius),
            "samples_per_atom": int(samples_per_atom),
            **data,
        }
        df = pd.DataFrame([row]).set_index(["structure", "probe_A"])
        self._cleanup([out_path])
        return self._apply_units_rows(df) if _apply_units else df

    def run_vol(
        self,
        structure: Union[str, Path],
        probe_radius: float,
        samples_per_uc: int,
        chan_radius: Optional[float] = None,
        _apply_units: bool = True,
    ) -> pd.DataFrame:
        """
        Accessible volume via -vol chan_radius probe_radius num_samples
        """
        base = self._derive_base(structure)
        cr = probe_radius if chan_radius is None else chan_radius
        out_path = self._unique_outfile(base, probe_radius, "vol")
        args = ["-vol", str(cr), str(probe_radius), str(int(samples_per_uc)), str(out_path), str(structure)]
        self._run(args)
        data = _parse_sa_vol_like_file(out_path)
        row = {
            "structure": base,
            "chan_radius_A": float(cr),
            "probe_A": float(probe_radius),
            "samples_per_uc": int(samples_per_uc),
            **data,
        }
        df = pd.DataFrame([row]).set_index(["structure", "probe_A"])
        self._cleanup([out_path])
        return self._apply_units_rows(df) if _apply_units else df

    def run_volpo(
        self,
        structure: Union[str, Path],
        probe_radius: float,
        samples_per_uc: int,
        chan_radius: Optional[float] = None,
        _apply_units: bool = True,
    ) -> pd.DataFrame:
        """
        Probe-occupiable volume via -volpo chan_radius probe_radius num_samples
        """
        base = self._derive_base(structure)
        cr = probe_radius if chan_radius is None else chan_radius
        out_path = self._unique_outfile(base, probe_radius, "volpo")
        args = ["-volpo", str(cr), str(probe_radius), str(int(samples_per_uc)), str(out_path), str(structure)]
        self._run(args)
        data = _parse_sa_vol_like_file(out_path)
        row = {
            "structure": base,
            "chan_radius_A": float(cr),
            "probe_A": float(probe_radius),
            "samples_per_uc": int(samples_per_uc),
            **data,
        }
        df = pd.DataFrame([row]).set_index(["structure", "probe_A"])
        self._cleanup([out_path])
        return self._apply_units_rows(df) if _apply_units else df

    def run_psd(
        self,
        structure: Union[str, Path],
        probe_radius: float,
        samples_per_uc: int,
        chan_radius: Optional[float] = None,
        _apply_units: bool = True,
    ) -> pd.DataFrame:
        """
        Pore size distribution via -psd chan_radius probe_radius num_samples
        """
        base = self._derive_base(structure)
        cr = probe_radius if chan_radius is None else chan_radius
        out_path = self._unique_outfile(base, probe_radius, "psd")
        args = ["-psd", str(cr), str(probe_radius), str(int(samples_per_uc)), str(out_path), str(structure)]
        self._run(args)
        psd_file = out_path
        alt = self.out_dir / f"{base}.psd_histo"
        if not psd_file.exists() and alt.exists():
            psd_file = alt
        df = _parse_psd_file(psd_file)
        # cleanup both possible outputs
        self._cleanup([out_path, alt])
        if df.empty:
            return df if not _apply_units else self._apply_units_rows(df)
        df.insert(0, "structure", base)
        df.insert(1, "probe_A", float(probe_radius))
        df.insert(2, "samples_per_uc", int(samples_per_uc))
        df.insert(3, "chan_radius_A", float(cr))
        df = df.set_index(["structure", "probe_A"])
        return self._apply_units_rows(df) if _apply_units else df

    # -------------- Batch helpers -----------------

    def batch_res(self, structures: Sequence[Union[str, Path]]) -> pd.DataFrame:
        frames = [self.run_res(s, _apply_units=False) for s in structures]
        df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
        return self._apply_units_rows(df)

    def batch_chan(self, structures: Sequence[Union[str, Path]], probe_radii: Sequence[float]) -> pd.DataFrame:
        frames = []
        for s in structures:
            for pr in probe_radii:
                frames.append(self.run_chan(s, pr, _apply_units=False))
        df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
        return self._apply_units_rows(df)

    def batch_sa(
        self,
        structures: Sequence[Union[str, Path]],
        probe_radii: Sequence[float],
        samples_per_atom: int,
        chan_radius: Optional[float] = None,
    ) -> pd.DataFrame:
        frames = []
        for s in structures:
            for pr in probe_radii:
                frames.append(self.run_sa(s, pr, samples_per_atom, chan_radius=chan_radius, _apply_units=False))
        df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
        return self._apply_units_rows(df)

    def batch_vol(
        self,
        structures: Sequence[Union[str, Path]],
        probe_radii: Sequence[float],
        samples_per_uc: int,
        chan_radius: Optional[float] = None,
    ) -> pd.DataFrame:
        frames = []
        for s in structures:
            for pr in probe_radii:
                frames.append(self.run_vol(s, pr, samples_per_uc, chan_radius=chan_radius, _apply_units=False))
        df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
        return self._apply_units_rows(df)

    def batch_volpo(
        self,
        structures: Sequence[Union[str, Path]],
        probe_radii: Sequence[float],
        samples_per_uc: int,
        chan_radius: Optional[float] = None,
    ) -> pd.DataFrame:
        frames = []
        for s in structures:
            for pr in probe_radii:
                frames.append(self.run_volpo(s, pr, samples_per_uc, chan_radius=chan_radius, _apply_units=False))
        df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
        return self._apply_units_rows(df)

    def batch_psd(
        self,
        structures: Sequence[Union[str, Path]],
        probe_radii: Sequence[float],
        samples_per_uc: int,
        chan_radius: Optional[float] = None,
    ) -> pd.DataFrame:
        frames = []
        for s in structures:
            for pr in probe_radii:
                df = self.run_psd(s, pr, samples_per_uc, chan_radius=chan_radius, _apply_units=False)
                if not df.empty:
                    frames.append(df)
        df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
        return self._apply_units_rows(df)

    # -------------- Convenience: one-shot summary -----------------
    def concat_list_results(self, lst_res: List[Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        if not lst_res:
            return {}
        keys0 = set(lst_res[0].keys())
        assert all(set(r.keys()) == keys0 for r in lst_res), "All result dicts must have identical keys"

        dict_lst_df: Dict[str, List[pd.DataFrame]] = defaultdict(list)
        for r in lst_res:
            for k, df in r.items():
                dict_lst_df[k].append(df)

        cat_results: Dict[str, pd.DataFrame] = {}
        for ky, lst_df in dict_lst_df.items():
            df = pd.concat(lst_df).sort_index()
            cat_results[ky] = self._apply_units_rows(df) if self.units_as_rows else df

        return cat_results

    def _summarize_one(
        self,
        structures: Union[str, Path],
        probe_radius: float,
        sa_samples_per_atom: int = 2000,
        vol_samples_per_uc: int = 50000,
        psd_samples_per_uc: Optional[int] = None,
        chan_radius_for_sa_vol: Optional[float] = None,
    ) -> Dict[str, pd.DataFrame]:
        return _sum_wrapper(
            self,
            structures,
            probe_radius,
            sa_samples_per_atom,
            vol_samples_per_uc,
            psd_samples_per_uc,
            chan_radius_for_sa_vol,
        )

    def _prepare_iterator(
        self,
        structures: Sequence[Union[str, Path]],
        probe_radii: Sequence[float],
        sa_samples_per_atom: int = 2000,
        vol_samples_per_uc: int = 50000,
        psd_samples_per_uc: Optional[int] = None,
        chan_radius_for_sa_vol: Optional[float] = None,
        nproc: Optional[int] = 1,
    ) -> Tuple[int, Any]:
        if isinstance(probe_radii, float):
            probe_radii = [probe_radii]

        if nproc == 1 or nproc is None:
            arg_list = [
                [s, pr, sa_samples_per_atom, vol_samples_per_uc, psd_samples_per_uc, chan_radius_for_sa_vol]
                for s, pr in product(structures, probe_radii)
            ]
            _iterators = iter(arg_list)
        else:
            if not isinstance(nproc, int) and nproc is not None:
                raise TypeError("The nproc should be an integer or None")
            elif nproc is None or nproc < 1:
                nproc = os.cpu_count() or 1

            lst_struct: List[Union[str, Path]] = []
            lst_probe: List[float] = []
            for s, pr in product(structures, probe_radii):
                lst_struct.append(s)
                lst_probe.append(pr)
            sa_samples_per_atom_list = [sa_samples_per_atom] * len(lst_struct)
            vol_samples_per_uc_list = [vol_samples_per_uc] * len(lst_struct)
            psd_samples_per_uc_list = [psd_samples_per_uc] * len(lst_struct)
            chan_radius_list = [chan_radius_for_sa_vol] * len(lst_struct)
            _iterators = (
                [self] * len(lst_struct),
                lst_struct,
                lst_probe,
                sa_samples_per_atom_list,
                vol_samples_per_uc_list,
                psd_samples_per_uc_list,
                chan_radius_list,
            )

        return nproc or 1, _iterators

    def summarize(
        self,
        structures: Sequence[Union[str, Path]],
        probe_radii: Sequence[float],
        sa_samples_per_atom: int = 2000,
        vol_samples_per_uc: int = 50000,
        psd_samples_per_uc: Optional[int] = None,
        chan_radius_for_sa_vol: Optional[float] = None,
        nproc: Optional[int] = 1,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run selected analyses and return a dict of DataFrames. Keys are task names:
          - 'res', 'chan', 'sa', 'vol', 'volpo', optional 'psd'
        """
        nproc, iterators = self._prepare_iterator(
            structures, probe_radii, sa_samples_per_atom, vol_samples_per_uc, psd_samples_per_uc, chan_radius_for_sa_vol, nproc
        )

        if nproc == 1:
            lst_res = [self._summarize_one(*args) for args in iterators]
        else:
            # Note: Using process-based parallelism because work is external (subprocess),
            # and Python GIL isn't a bottleneck. Ensure this is used in a module context
            # safe for multiprocessing (Windows spawn semantics).
            with futures.ProcessPoolExecutor(max_workers=nproc) as executor:
                lst_res = list(executor.map(_sum_wrapper, *iterators))

        return self.concat_list_results(lst_res)


# Tasks that can be combined (PSD is intentionally excluded)
NON_PSD_TASKS = ("res", "chan", "sa", "vol", "volpo")

# Key columns for alignment
KEY_COLS = ("structure", "probe_A")

# Optional context (identifier) columns you may want to keep
CONTEXT_COLS = ("samples_per_atom", "samples_per_uc", "chan_radius_A", "dimensionality")


# ---------------------- Unit parsing and detection ----------------------

def _split_var_unit_from_name(name: str) -> Tuple[str, str]:
    """
    Heuristically split a column name into (metric, unit).
    Examples:
      'ASA_A^2' -> ('ASA','A^2'), 'Di_A'->('Di','A'), 'Number_of_channels'->('Number_of_channels','')
    """
    # Delegate to ZeoppRunner._split_var_unit for consistency
    return ZeoppRunner._split_var_unit(name)  # type: ignore[attr-defined]


def _detect_units_rows_and_strip(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """
    Detect and extract the top two 'variable/unit' rows if produced with units_as_rows=True.
    Returns (df_without_first_two_rows, col->metric map, col->unit map).
    If not detected, returns (original df, maps derived from column names).

    Detection rule: both first two rows have < 20% numeric values.
    """
    if df is None or df.empty:
        return df, {}, {}

    d = df.copy()
    if isinstance(d.index, pd.MultiIndex) or d.index.name is not None:
        d = d.reset_index()

    def frac_numeric(row: pd.Series) -> float:
        coerced = pd.to_numeric(row, errors="coerce")
        total = row.notna().sum()
        return float(coerced.notna().sum()) / total if total else 0.0

    has_units_rows = False
    if d.shape[0] >= 2:
        f0, f1 = frac_numeric(d.iloc[0]), frac_numeric(d.iloc[1])
        if f0 < 0.2 and f1 < 0.2:
            has_units_rows = True

    if has_units_rows:
        var_map = {c: str(d.iloc[0][c]) for c in d.columns}
        unit_map = {c: str(d.iloc[1][c]) for c in d.columns}
        # Normalize unit map values
        unit_map = {k: ("" if (v is None or str(v).lower() == "nan") else str(v)) for k, v in unit_map.items()}
        d = d.iloc[2:].reset_index(drop=True)
    else:
        var_map, unit_map = {}, {}
        for c in d.columns:
            v, u = _split_var_unit_from_name(str(c))
            var_map[c] = v
            unit_map[c] = u

    return d, var_map, unit_map


# ---------------------- Probe collection and name helpers ----------------------

def _collect_probes_per_structure(results: Dict[str, pd.DataFrame]) -> Dict[str, Set[float]]:
    """
    Build structure -> set(probe_A) from all selected non-res task tables (works with or without units_as_rows).
    This is used to expand 'res' rows across all probes seen for the same structure.
    """
    probes: Dict[str, Set[float]] = {}
    for t, df in results.items():
        if t not in NON_PSD_TASKS or t == "res" or df is None or df.empty:
            continue
        d, _, _ = _detect_units_rows_and_strip(df)
        if isinstance(d.index, pd.MultiIndex) or d.index.name is not None:
            d = d.reset_index()
        if "structure" in d.columns and "probe_A" in d.columns:
            for s, p in d[["structure", "probe_A"]].dropna().itertuples(index=False, name=None):
                probes.setdefault(str(s), set()).add(float(p))
    return probes


def _build_metric_col_name(metric: str, unit: str, keep_units: bool = True) -> str:
    """Create a display name for a metric with optional unit annotation."""
    metric = str(metric)
    unit = str(unit) if unit is not None else ""
    if keep_units and unit and unit.lower() != "nan":
        return f"{metric} ({unit})"
    return metric


def _resolve_collision(name: str, existing: Set[str], task: Optional[str] = None) -> str:
    """
    Ensure a column name is unique. If collision, append ' [task=...]', and if needed add counters.
    """
    base = name
    if name not in existing:
        return name
    if task:
        name = f"{base} [task={task}]"
    if name not in existing:
        return name
    # Add numbered suffix if still conflicting
    i = 2
    while True:
        cand = f"{name} #{i}"
        if cand not in existing:
            return cand
        i += 1


# ---------------------- Task standardization to wide ----------------------

def _prepare_task_wide(
    task: str,
    df: pd.DataFrame,
    probes_per_structure: Dict[str, Set[float]],
    keep_units: bool = True,
    chan_mode: str = "expand",   # "expand" or "aggregate"
    chan_agg: str = "max",       # used when chan_mode="aggregate": "max","mean","min","first","sum"
    include_context: bool = False,
) -> pd.DataFrame:
    """
    Standardize a single task table to a wide, one-row-per-structure-probe DataFrame.
    Returns only KEY_COLS plus metric columns (and optional context columns).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=list(KEY_COLS))

    d, var_map, unit_map = _detect_units_rows_and_strip(df)
    if isinstance(d.index, pd.MultiIndex) or d.index.name is not None:
        d = d.reset_index()

    if "structure" not in d.columns:
        raise ValueError(f"Task '{task}' DataFrame lacks 'structure' column after reset_index().")

    # Ensure probe_A
    if task == "res":
        # Expand across all probes seen for this structure in other tasks
        pairs = []
        for s in d["structure"].astype(str).unique():
            s_probes = sorted(probes_per_structure.get(str(s), []))
            if s_probes:
                for p in s_probes:
                    pairs.append((s, p))
            else:
                pairs.append((s, np.nan))  # no probe info available
        base = pd.DataFrame(pairs, columns=list(KEY_COLS))
        d = base.merge(d, on="structure", how="left")
    else:
        if "probe_A" not in d.columns:
            d["probe_A"] = np.nan

    # Identify which columns are metrics vs identifiers
    id_set: Set[str] = set(KEY_COLS) | {"channel_id"} | set(c for c in CONTEXT_COLS if c in d.columns)
    # Context columns should not be treated as metrics; they may be appended later if requested.
    metric_cols = [c for c in d.columns if c not in id_set]

    # Special handling for 'chan'
    if task == "chan":
        # Optional: keep dimensionality as a context column (per structure-probe)
        keep_cols = list(KEY_COLS)
        dim = None
        if include_context and "dimensionality" in d.columns:
            # Collapse to one value per structure-probe (if inconsistent, keep the first)
            dim = (
                d[["structure", "probe_A", "dimensionality"]]
                .dropna(subset=["structure"])
                .drop_duplicates(subset=["structure", "probe_A"])
            )

        if "channel_id" not in d.columns:
            d["channel_id"] = np.nan

        # Focus on numeric metrics
        metrics_numeric = []
        for c in metric_cols:
            # Only pivot numeric columns; skip text columns
            if pd.api.types.is_numeric_dtype(d[c]):
                metrics_numeric.append(c)

        if chan_mode not in ("expand", "aggregate"):
            raise ValueError("chan_mode must be 'expand' or 'aggregate'")
        if chan_mode == "aggregate":
            agg_map = {
                "max": "max",
                "mean": "mean",
                "min": "min",
                "first": "first",
                "sum": "sum",
            }
            if chan_agg not in agg_map:
                raise ValueError("chan_agg must be one of: 'max','mean','min','first','sum'")

            # Aggregate per structure-probe for each metric
            grp = d.groupby(list(KEY_COLS))
            agg_df = grp[metrics_numeric].agg(agg_map[chan_agg]).reset_index()

            # Rename metrics with unit (and aggregation marker)
            rename_map = {}
            for c in metrics_numeric:
                m = var_map.get(c, _split_var_unit_from_name(str(c))[0])
                u = unit_map.get(c, _split_var_unit_from_name(str(c))[1])
                new_name = _build_metric_col_name(m, u, keep_units=keep_units)
                new_name = f"{new_name} [chan-{chan_agg}]"
                rename_map[c] = new_name
            agg_df = agg_df.rename(columns=rename_map)

            out = agg_df
        else:
            # Expand per channel into separate columns for each metric
            out = d[list(KEY_COLS)].drop_duplicates().reset_index(drop=True)
            existing: Set[str] = set(out.columns)

            for c in metrics_numeric:
                m = var_map.get(c, _split_var_unit_from_name(str(c))[0])
                u = unit_map.get(c, _split_var_unit_from_name(str(c))[1])
                base_name = _build_metric_col_name(m, u, keep_units=keep_units)

                piv = d.pivot_table(index=list(KEY_COLS), columns="channel_id", values=c, aggfunc="first")
                if piv.empty:
                    continue
                piv = piv.reset_index()
                # Rename channel columns
                new_cols = {}
                for ch in [col for col in piv.columns if col not in KEY_COLS]:
                    col_label = f"{base_name} [chan={ch}]"
                    col_label = _resolve_collision(col_label, existing, task=task)
                    new_cols[ch] = col_label
                    existing.add(col_label)
                piv = piv.rename(columns=new_cols)

                out = out.merge(piv, on=list(KEY_COLS), how="left")

        # Merge dimensionality if kept
        if include_context and dim is not None and not dim.empty:
            out = out.merge(dim, on=list(KEY_COLS), how="left")

        return out

    # Non-chan tasks: rename metric columns with units, then keep one row per structure-probe
    rename_map = {}
    for c in metric_cols:
        m = var_map.get(c, _split_var_unit_from_name(str(c))[0])
        u = unit_map.get(c, _split_var_unit_from_name(str(c))[1])
        rename_map[c] = _build_metric_col_name(m, u, keep_units=keep_units)

    wide = d[list(KEY_COLS) + metric_cols].copy()
    wide = wide.rename(columns=rename_map)
    # Drop duplicates (if any) by keeping the first per structure-probe
    wide = wide.sort_values(list(KEY_COLS)).drop_duplicates(subset=list(KEY_COLS), keep="first")
    if include_context:
        # Append available context columns (coalesced later during global merge)
        ctx = [c for c in CONTEXT_COLS if c in d.columns]
        if ctx:
            wide = wide.merge(
                d[list(KEY_COLS) + ctx].drop_duplicates(subset=list(KEY_COLS)),
                on=list(KEY_COLS),
                how="left",
            )
    return wide


# ---------------------- Merging all tasks ----------------------

def _outer_merge_coalesce(left: pd.DataFrame, right: pd.DataFrame, on: Sequence[str]) -> pd.DataFrame:
    """
    Outer-merge two wide tables on 'on', and coalesce overlapping non-key columns with suffix '__dup'.
    Preference: keep left values, fill with right where missing.
    """
    if left is None or left.empty:
        return right.copy()
    if right is None or right.empty:
        return left.copy()

    merged = pd.merge(left, right, on=list(on), how="outer", suffixes=("", "__dup"))
    dup_cols = [c for c in merged.columns if c.endswith("__dup")]
    for dup in dup_cols:
        base = dup[:-5]
        if base in merged.columns:
            merged[base] = merged[base].combine_first(merged[dup])
            merged = merged.drop(columns=[dup])
        else:
            merged = merged.rename(columns={dup: base})
    return merged


def concat_results(
    results: Dict[str, pd.DataFrame],
    tasks: Optional[Sequence[str]] = None,
    keep_units: bool = True,
    chan_mode: str = "expand",   # "expand" -> per-channel columns; "aggregate" -> reduce over channels
    chan_agg: str = "max",       # when chan_mode="aggregate": "max","mean","min","first","sum"
    include_context: bool = False,
    add_task_suffix_on_collision: bool = True,
    sort_rows: bool = True,
) -> pd.DataFrame:
    """
    Concatenate non-PSD task tables into a single wide table:
      - One row per (structure, probe_A)
      - Columns are attributes; units included in column names when available
      - 'chan' can be expanded to per-channel columns or aggregated over channels
      - 'res' is expanded across all probes observed for the same structure in other tasks
    """
    if not isinstance(results, dict) or not results:
        return pd.DataFrame(columns=list(KEY_COLS))

    # Select tasks to include
    present = [t for t in NON_PSD_TASKS if t in results and isinstance(results[t], pd.DataFrame)]
    if tasks is None:
        selected = present
    else:
        wanted = [t.lower() for t in tasks]
        selected = [t for t in wanted if t in present]
    if not selected:
        return pd.DataFrame(columns=list(KEY_COLS))

    # Build structure->probes map for expanding 'res'
    probes_per_structure = _collect_probes_per_structure({k: results[k] for k in selected if k in results})

    # Prepare each task into a wide partial
    partials: Dict[str, pd.DataFrame] = {}
    for t in selected:
        df_t = results.get(t)
        if df_t is None or df_t.empty:
            continue
        part = _prepare_task_wide(
            t,
            df_t,
            probes_per_structure,
            keep_units=keep_units,
            chan_mode=chan_mode,
            chan_agg=chan_agg,
            include_context=include_context,
        )
        partials[t] = part

    if not partials:
        return pd.DataFrame(columns=list(KEY_COLS))

    # Merge all partials
    merged: Optional[pd.DataFrame] = None
    existing_cols: Set[str] = set(KEY_COLS)
    for t in selected:
        part = partials.get(t)
        if part is None or part.empty:
            continue

        # Resolve potential metric column name collisions across tasks
        if add_task_suffix_on_collision:
            rename_map = {}
            for c in part.columns:
                if c in KEY_COLS:
                    continue
                if c in existing_cols:
                    new_c = _resolve_collision(c, existing_cols, task=t)
                    if new_c != c:
                        rename_map[c] = new_c
                        existing_cols.add(new_c)
                else:
                    existing_cols.add(c)
            if rename_map:
                part = part.rename(columns=rename_map)

        merged = _outer_merge_coalesce(merged, part, on=KEY_COLS)

    # Final polish
    if merged is None or merged.empty:
        merged = pd.DataFrame(columns=list(KEY_COLS))

    if sort_rows and not merged.empty:
        sort_cols = [c for c in KEY_COLS if c in merged.columns]
        merged = merged.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)

    return merged
