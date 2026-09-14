# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : test_zeopp
 Created   : 2025/8/29 10:52
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
# test_zeoxx.py
import os.path as osp
import re
import tempfile
from textwrap import dedent
import warnings
import unittest as ut
import logging
from pathlib import Path
from glob import glob
from typing import List
from unittest.mock import patch

import pandas as pd
import numpy as np

from hotpot.plugins import zeoxx
from hotpot.utils.configs import setup_logging

# config the logging stdout format
setup_logging()

def make_mock_run(testcase):
    """
    Create a ZeoppRunner._run replacement that writes synthetic Zeo++ output files
    according to the args vector for each task:
      -res  out  structure
      -chan pr   out  structure
      -sa   cr   pr   samples  out  structure
      -vol  cr   pr   samples  out  structure
      -volpo cr  pr   samples  out  structure
      -psd  cr   pr   samples  out  structure
    """
    def _mock_run(self, args, cwd=None):
        tool = args[0]
        out_path = None
        content = None

        if tool == "-res":
            # args: ["-res", out_path, structure]
            out_path = Path(args[1])
            # single line with at least 3 floats
            content = "mock.cif 4.000 3.000 2.000\n"

        elif tool == "-chan":
            # args: ["-chan", probe_radius, out_path, structure]
            out_path = Path(args[2])
            content = (
                f"{out_path.name}   2 channels identified of dimensionality 1\n"
                "Channel  0  4.89082  3.03868  4.89082\n"
                "Channel  1  5.12000  3.25000  5.00000\n"
            )

        elif tool == "-sa":
            # args: ["-sa", cr, pr, samples_per_atom, out_path, structure]
            out_path = Path(args[4])
            content = (
                "@ ASA_A^2: 60.0 ASA_m^2/cm^3: 2000.0 ASA_m^2/g: 1200.0\n"
                "Number_of_channels: 2 Channel_surface_area_A^2: 42.0\n"
            )

        elif tool == "-vol":
            # args: ["-vol", cr, pr, samples_per_uc, out_path, structure]
            out_path = Path(args[4])
            content = (
                "@ AV_A^3: 100.0 AV_cm^3/g: 0.40\n"
                "Number_of_channels: 2 Channel_volume_A^3: 77.0\n"
            )

        elif tool == "-volpo":
            # args: ["-volpo", cr, pr, samples_per_uc, out_path, structure]
            out_path = Path(args[4])
            content = (
                "@ POV_A^3: 80.0\n"
                "Number_of_pockets: 2 Pocket_volume_A^3: 5.0\n"
            )

        elif tool == "-psd":
            # args: ["-psd", cr, pr, samples_per_uc, out_path, structure]
            out_path = Path(args[4])
            content = (
                "# histogram for PSD\n"
                "0.5  10  0.1  0.05\n"
                "1.0  20  0.3  0.10\n"
                "1.5  30  0.6  0.15\n"
            )
        else:
            raise AssertionError(f"Unexpected tool flag in args: {args}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        testcase.created_outputs.append(out_path)
    return _mock_run


class TestParsers(ut.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_parse_res_file(self):
        p = self.root / "x.res"
        p.write_text("file.cif 4.2 3.3 2.4\n", encoding="utf-8")
        out = zeoxx._parse_res_file(p)
        self.assertEqual(frozenset(out.keys()), frozenset(("GCD", "PLD", "LCD")))
        self.assertAlmostEqual(out["GCD"], 4.2)
        self.assertAlmostEqual(out["PLD"], 3.3)
        self.assertAlmostEqual(out["LCD"], 2.4)

    def test_parse_chan_file(self):
        p = self.root / "y.chan"
        p.write_text(
            "y.chan  2 channels identified of dimensionality 2\n"
            "Channel  0  4.89082  3.03868  4.89082\n"
            "Channel  1  5.10000  3.20000  5.00000\n",
            encoding="utf-8",
        )
        dim, channels = zeoxx._parse_chan_file(p)
        self.assertEqual(dim, 2)
        self.assertEqual(len(channels), 2)
        self.assertEqual(channels[0]["channel_id"], 0)
        self.assertIn("Di_A", channels[0])

    def test_parse_sa_vol_like_file(self):
        p = self.root / "z.sa"
        p.write_text(
            "@ ASA_A^2: 60.0 ASA_m^2/cm^3: 2000.0 ASA_m^2/g: 1200.0\n"
            "Number_of_channels: 1 Channel_surface_area_A^2: 10.0\n"
            "Number_of_pockets: 0 Pocket_volume_A^3: 0.0\n",
            encoding="utf-8",
        )
        out = zeoxx._parse_sa_vol_like_file(p)
        # Expect presence of keys parsed
        self.assertIn("ASA_A^2", out)
        self.assertIn("ASA_m^2/cm^3", out)
        self.assertIn("Number_of_channels", out)
        self.assertIn("Channel_surface_area_A^2", out)

    def test_parse_psd_file_standard(self):
        p = self.root / "psd.psd"
        p.write_text(
            "# psd\n"
            "0.5 1 0.1 0.01\n"
            "1.0 2 0.3 0.05\n",
            encoding="utf-8",
        )
        df = zeoxx._parse_psd_file(p)
        self.assertListEqual(list(df.columns), ["Bin(A)", "Count", "CumDist", "DerivDist"])
        self.assertEqual(df.shape, (2, 4))

    def test_parse_psd_file_generic(self):
        p = self.root / "psd_generic.psd"
        p.write_text(
            "# psd\n"
            "0.5 1 0.1\n"
            "1.0 2 0.3\n",
            encoding="utf-8",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = zeoxx._parse_psd_file(p)
            self.assertTrue(any("generic column names" in str(x.message) for x in w))
        self.assertListEqual(list(df.columns), ["col1", "col2", "col3"])
        self.assertEqual(df.shape, (2, 3))


class TestRunnerSingleProcess(ut.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmpdir.name) / "out"
        self.in_dir = Path(self.tmpdir.name) / "in"
        self.in_dir.mkdir(parents=True, exist_ok=True)
        # Create two dummy CIF files
        (self.in_dir / "A.cif").write_text("data_A", encoding="utf-8")
        (self.in_dir / "B.cif").write_text("data_B", encoding="utf-8")

        self.created_outputs: List[Path] = []

        # Patch which to avoid FileNotFoundError
        self.p_which = patch.object(zeoxx, "_which", return_value="/usr/bin/network")
        self.p_which.start()

        # Patch _run to write mock content
        self.p_run = patch.object(zeoxx.ZeoppRunner, "_run", new=make_mock_run(self))
        self.p_run.start()

    def tearDown(self):
        self.p_run.stop()
        self.p_which.stop()
        self.tmpdir.cleanup()

    def test_run_methods(self):
        runner = zeoxx.ZeoppRunner(out_dir=self.out_dir, cleanup_outputs=False)
        struct = str(self.in_dir / "A.cif")

        # res
        df_res = runner.run_res(struct)
        self.assertIn("GCD", df_res.columns)
        self.assertIn("PLD", df_res.columns)
        self.assertIn("LCD", df_res.columns)

        # chan
        df_chan = runner.run_chan(struct, probe_radius=1.2)
        self.assertTrue(isinstance(df_chan.index, pd.MultiIndex))
        self.assertIn("Di_A", df_chan.columns)
        self.assertIn("Df_A", df_chan.columns)
        self.assertIn("Dif_A", df_chan.columns)

        # sa
        df_sa = runner.run_sa(struct, probe_radius=1.2, samples_per_atom=1000)
        self.assertIn("ASA_A^2", df_sa.columns)
        self.assertIn("ASA_m^2/cm^3", df_sa.columns)

        # vol
        df_vol = runner.run_vol(struct, probe_radius=1.2, samples_per_uc=2000)
        self.assertIn("AV_A^3", df_vol.columns)

        # volpo
        df_volpo = runner.run_volpo(struct, probe_radius=1.2, samples_per_uc=2000)
        self.assertIn("POV_A^3", df_volpo.columns)

        # psd
        df_psd = runner.run_psd(struct, probe_radius=1.2, samples_per_uc=2000)
        self.assertIn("Bin(A)", df_psd.columns)
        self.assertIn("Count", df_psd.columns)

        # Ensure output files exist (cleanup not enabled)
        self.assertTrue(any(p.exists() for p in self.created_outputs))

    def test_summarize_and_concat_expand(self):
        runner = zeoxx.ZeoppRunner(out_dir=self.out_dir, cleanup_outputs=False)
        structs = sorted(glob(str(self.in_dir / "*.cif")))
        results = runner.summarize(
            structures=structs,
            probe_radii=[1.2, 1.5],
            sa_samples_per_atom=1000,
            vol_samples_per_uc=2000,
            nproc=1,
        )
        # Expected task keys (PSD excluded by default)
        for key in ("res", "chan", "sa", "vol", "volpo"):
            self.assertIn(key, results)
            self.assertIsInstance(results[key], pd.DataFrame)
            self.assertFalse(results[key].empty)

        # Concatenate (expand channels)
        merged = zeoxx.concat_results(
            results,
            keep_units=True,
            chan_mode="expand",
            include_context=False,
            add_task_suffix_on_collision=True,
            sort_rows=True,
        )
        self.assertIn("structure", merged.columns)
        self.assertIn("probe_A", merged.columns)


        chan_di_cols = [c for c in merged.columns if re.match(
                                                        dedent(r"""^Di $A$
                                                                        $$
                                                                        chan=\d+
                                                                        $$
                                                                        $"""), c)]

        # Ensure res got expanded to both probes if other tasks had both
        subset = merged[merged["structure"].isin(["A", "B"])]
        probes_seen = set(subset["probe_A"].dropna().unique().tolist())
        self.assertTrue({1.2, 1.5}.issubset(probes_seen))

    def test_concat_aggregate_and_context(self):
        runner = zeoxx.ZeoppRunner(out_dir=self.out_dir, cleanup_outputs=False)
        structs = [str(self.in_dir / "A.cif")]
        results = runner.summarize(
            structures=structs,
            probe_radii=[1.2],
            sa_samples_per_atom=1000,
            vol_samples_per_uc=2000,
            nproc=1,
        )
        merged = zeoxx.concat_results(
            results,
            keep_units=False,
            chan_mode="aggregate",
            chan_agg="max",
            include_context=True,
            add_task_suffix_on_collision=True,
        )
        # Aggregated chan metric name
        agg_cols = [c for c in merged.columns if c.startswith("Di [chan-max]")]
        self.assertTrue(len(agg_cols) == 1)
        # Context columns should be present when requested
        self.assertIn("dimensionality", merged.columns)
        self.assertIn("samples_per_atom", merged.columns)
        self.assertIn("samples_per_uc", merged.columns)
        self.assertIn("chan_radius_A", merged.columns)

    def test_units_as_rows_roundtrip(self):
        # Enable units_as_rows to ensure detection/stripping in concat_results works
        runner = zeoxx.ZeoppRunner(out_dir=self.out_dir, units_as_rows=True, cleanup_outputs=False)
        structs = [str(self.in_dir / "A.cif"), str(self.in_dir / "B.cif")]
        results = runner.summarize(
            structures=structs,
            probe_radii=[1.2, 1.5],
            sa_samples_per_atom=1000,
            vol_samples_per_uc=2000,
            nproc=1,
        )
        # Each df now has two meta rows at top; concat_results should handle them
        merged = zeoxx.concat_results(results, keep_units=True, chan_mode="expand", include_context=False)
        self.assertIn("structure", merged.columns)
        self.assertGreater(len(merged), 0)

    def test_cleanup_outputs(self):
        runner = zeoxx.ZeoppRunner(out_dir=self.out_dir, cleanup_outputs=True)
        struct = str(self.in_dir / "A.cif")
        # Run one task
        _ = runner.run_sa(struct, probe_radius=1.2, samples_per_atom=100)
        # Files should have been deleted by cleanup
        self.assertTrue(all(not p.exists() for p in self.created_outputs))


class TestUtilities(ut.TestCase):
    def test_split_var_unit_from_name(self):
        # Uses fixed/final implementation (no __func__ usage)
        self.assertEqual(zeoxx._split_var_unit_from_name("ASA_A^2"), ("ASA", "A^2"))
        self.assertEqual(zeoxx._split_var_unit_from_name("Di_A"), ("Di", "A"))
        self.assertEqual(zeoxx._split_var_unit_from_name("Number_of_channels"), ("Number_of_channels", ""))
        self.assertEqual(zeoxx._split_var_unit_from_name(123), ("123", ""))

    def test_outer_merge_coalesce(self):
        left = pd.DataFrame(
            {"structure": ["A"], "probe_A": [1.2], "X": [1.0]}
        )
        right = pd.DataFrame(
            {"structure": ["A"], "probe_A": [1.2], "X": [np.nan], "Y": [2.0]}
        )
        out = zeoxx._outer_merge_coalesce(left, right, on=("structure", "probe_A"))
        self.assertIn("X", out.columns)
        self.assertIn("Y", out.columns)
        self.assertAlmostEqual(out.loc[0, "X"], 1.0)
        self.assertAlmostEqual(out.loc[0, "Y"], 2.0)


class TestZeoXX(ut.TestCase):
    def test_zeoXX(self):
        # Example usage:
        runner = zeoxx.ZeoppRunner(
            network_bin="network",  # or "/path/to/network"
            # out_dir="zeopp_outputs",
            ha=True,
            nor=False,
            radii_file=None,
            mass_file=None,
            strip_atom_names=False,
            extra_args=[],
            timeout=600,
            # enabled_tasks=["res", "sa", "vol"],  # default tasks for summarize
            units_as_rows=True,  # show variable/unit as first two rows
            # cleanup_outputs=True,  # delete Zeo++ outputs after parsing
        )

        logging.info(f"runner out_dir: {runner.out_dir}")

        inputs_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', 'input'))
        structs = glob(osp.join(inputs_dir, '*.cif'))
        probe = 1.2

        # # Direct calls (units rows applied per DataFrame)
        # df_res = runner.run_res(struct1)
        # print("RES:\n", df_res, "\n")

        # Batch/summarize calls (units rows applied once after concat)
        results = runner.summarize(
            structures=structs,
            probe_radii=[probe, 1.5],
            sa_samples_per_atom=2000,
            vol_samples_per_uc=50000,
            nproc=12,
            # tasks=["res", "sa", "vol"],
        )
        for k, v in results.items():
            print(f"{k.upper()}:\n", v.head(6), "\n")

        cres = zeoxx.concat_results(results)

        print(cres)


if __name__ == '__main__':
    # Example usage:
    runner = zeoxx.ZeoppRunner(
        network_bin="network",  # or "/path/to/network"
        # out_dir="zeopp_outputs",
        ha=True,
        nor=False,
        radii_file=None,
        mass_file=None,
        strip_atom_names=False,
        extra_args=[],
        timeout=600,
        # enabled_tasks=["res", "sa", "vol"],  # default tasks for summarize
        units_as_rows=True,  # show variable/unit as first two rows
        # cleanup_outputs=True,  # delete Zeo++ outputs after parsing
    )

    inputs_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', 'input'))
    structs = glob(osp.join(inputs_dir, '*.cif'))
    probe = 1.2

    # # Direct calls (units rows applied per DataFrame)
    # df_res = runner.run_res(struct1)
    # print("RES:\n", df_res, "\n")

    # Batch/summarize calls (units rows applied once after concat)
    results = runner.summarize(
        structures=structs,
        probe_radii=[probe, 1.5],
        sa_samples_per_atom=2000,
        vol_samples_per_uc=50000,
        nproc=12,
        # tasks=["res", "sa", "vol"],
    )
    for k, v in results.items():
        print(f"{k.upper()}:\n", v.head(6), "\n")

    cres = zeoxx.concat_results(results)
