# ZeoppRunner - Python Wrapper for Zeo++ Network Binary

## Overview

ZeoppRunner is a comprehensive Python wrapper for the Zeo++ `network` binary that computes porous material descriptors and parses the results into pandas DataFrames. It provides high-level interfaces for batch processing, automatic unit handling, and result concatenation for materials science applications.

### Key Features

* **Automated Zeo++ Integration**: Seamless wrapper around the `network` binary with automatic output parsing
* **Multiple Analysis Types**: Support for pore diameters (res), channels (chan), surface areas (sa), volumes (vol/volpo), and pore size distributions (psd)
* **Batch Processing**: Process multiple structures and probe sizes efficiently
* **Unit-Aware Output**: Automatic extraction and preservation of physical units
* **Result Concatenation**: Combine multi-task outputs into analysis-ready wide tables
* **Flexible Configuration**: Fine-grained control over tasks, accuracy settings, and output management

## Installation

### Requirements

* Python ≥ 3.9
* Zeo++ software installed with `network` binary accessible
* Python packages:
  ```bash
  pip install pandas numpy  
  ```

## Setup

1. Install Zeo++ from http://www.zeoplusplus.org/
2. Ensure the network binary is in your PATH or note its absolute path
3. Install the Python module:

```pycon
# Save the module as zeopp_runner.py
from zeopp_runner import ZeoppRunner, concat_results
```

## Quick Start Tutorial

### Basic Usage

```pycon
from zeopp_runner import ZeoppRunner

# Initialize the runner
runner = ZeoppRunner(
    network_bin="network",       # or "/path/to/network"
    out_dir="zeopp_outputs",     # output directory
    ha=True,                     # high-accuracy mode
    units_as_rows=True,          # include unit information
    cleanup_outputs=True         # auto-cleanup temporary files
)

# Run a single analysis
df_res = runner.run_res("MOF-5.cif")
print(df_res)

# Run surface area calculation
df_sa = runner.run_sa("MOF-5.cif", probe_radius=1.2, samples_per_atom=2000)
print(df_sa)
```

### Batch Processing Multiple Structures

```pycon
# Process multiple structures with different probe sizes
structures = ["MOF-5.cif", "IRMOF-1.cif", "ZIF-8.cif"]
probe_radii = [1.2, 1.5, 1.65]  # Common probe sizes for N2, Ar, CO2

# Run comprehensive analysis
results = runner.summarize(
    structures=structures,
    probe_radii=probe_radii,
    sa_samples_per_atom=2000,
    vol_samples_per_uc=50000,
    tasks=["res", "chan", "sa", "vol", "volpo"],  # select tasks
    nproc=8  # running work with 8 processes, -1 to use all cores
)

# Access individual results
print("Pore diameters:")
print(results["res"])
print("\nSurface areas:")
print(results["sa"])
```

### Concatenating Results into a Single Table

```pycon
from zeopp_runner import concat_results

# Merge all results into one wide table
wide_table = concat_results(
    results,
    keep_units=True,           # include units in column names
    chan_mode="aggregate",     # aggregate channel data
    chan_agg="max",           # use maximum values for channels
    include_context=False     # exclude sampling parameters
)

print(wide_table)
# Output: One row per structure-probe pair with all metrics as columns

# Export to CSV
wide_table.to_csv("porous_materials_analysis.csv")
```

### Advanced: Custom Task Selection

```pycon
# Configure default tasks for the runner
runner = ZeoppRunner(
    network_bin="network",
    enabled_tasks=["res", "sa", "vol"],  # default tasks
    units_as_rows=True,
    cleanup_outputs=True
)

# Override for specific analysis
results = runner.summarize(
    structures=["structure.cif"],
    probe_radii=[1.2],
    tasks=["res", "chan", "sa", "vol", "volpo", "psd"]  # include PSD
)
```

## API Reference

### Class: ZeoppRunner

Main class for interfacing with Zeo++ network binary.

#### Constructor Parameters


| Parameter          | Type          | Default                           | Description                          |
|--------------------|---------------|-----------------------------------|--------------------------------------|
| `network_bin`      | str           | "network"                         | Path or name of network binary       |
| `out_dir`          | str/Path      | None                              | Output directory (temp if None)      |
| `ha`               | bool          | True                              | Use high-accuracy mode (-ha)         |
| `nor`              | bool          | False                             | Use point particle Voronoi (-nor)    |
| `radii_file`       | str/Path      | None                              | Custom atomic radii file             |
| `mass_file`        | str/Path      | None                              | Custom atomic mass file              |
| `strip_atom_names` | bool          | False                             | Strip atom names flag                |
| `extra_args`       | Sequence[str] | None                              | Additional arguments for network     |
| `timeout`          | int           | None                              | Process timeout in seconds           |
| `enabled_tasks`    | Sequence[str] | ["res","chan","sa","vol","volpo"] | Default tasks for summarize()        |
| `units_as_rows`    | bool          | False                             | Add variable/unit rows to DataFrames |
| `cleanup_outputs`  | bool          | False                             | Auto-delete Zeo++ output files       |


#### Core Methods

- `run_res(structure)` → DataFrame — Pore diameters (GLD, PLD, LCD)
- `run_chan(structure, probe_radius)` → DataFrame — Channel analysis
- `run_sa(structure, probe_radius, samples_per_atom)` → DataFrame — Accessible surface area
- `run_vol(structure, probe_radius, samples_per_uc)` → DataFrame — Accessible volume
- `run_volpo(structure, probe_radius, samples_per_uc)` → DataFrame — Probe-occupiable volume
- `run_psd(structure, probe_radius, samples_per_uc)` → DataFrame — Pore size distribution

#### Batch Methods
The `Batch Methods` just an easy batch implementation of Core Methods
- `batch_res(structures)`
- `batch_chan(structures, probe_radii)`
- `batch_sa(structures, probe_radii, samples_per_atom)`
- `batch_vol(structures, probe_radii, samples_per_uc)`
- `batch_volpo(structures, probe_radii, samples_per_uc)`
- `batch_psd(structures, probe_radii, samples_per_uc)`

#### Example of Core Methods
Compute pore limiting diameters (GLD, PLD, LCD): `run_res(structure, _apply_units=True) -> DataFrame`.
```pycon
df = runner.run_res("material.cif")
# Returns: DataFrame with columns [GLD, PLD, LCD]
```

Calculate accessible surface area: `run_sa(structure, probe_radius, samples_per_atom, chan_radius=None, _apply_units=True) -> DataFrame`
```pycon
df = runner.run_sa("material.cif", probe_radius=1.2, samples_per_atom=2000)
# Returns: DataFrame with columns [ASA_A^2, ASA_m^2/cm^3, ASA_m^2/g, ...]
```

Calculate accessible volume: run_vol(structure, `probe_radius, samples_per_uc, chan_radius=None, _apply_units=True) → DataFrame`
```pycon
df = runner.run_vol("material.cif", probe_radius=1.2, samples_per_uc=50000)
# Returns: DataFrame with columns [AV_A^3, AV_cm^3/g, ...]
```


Calculate probe-occupiable volume: `run_volpo(structure, probe_radius, samples_per_uc, chan_radius=None, _apply_units=True) → DataFrame`
```pycon
df = runner.run_volpo("material.cif", probe_radius=1.2, samples_per_uc=50000)
# Returns: DataFrame with columns [POAV_A^3, POAVF, ...]
```

Calculate pore size distribution: `run_psd(structure, probe_radius, samples_per_uc, chan_radius=None, _apply_units=True) → DataFrame`
```pycon
df = runner.run_psd("material.cif", probe_radius=1.2, samples_per_uc=50000)
# Returns: DataFrame with columns [Bin(A), Count, CumDist, DerivDist]
```

#### High-Level (high-throughput) Method

> **summarize(structures, probe_radii, ...) → Dict[str, DataFrame]**

Run multiple analyses and return results dictionary.

##### Parameters:

- structures: List of structure files
- probe_radii: List of probe radii in Angstroms
- sa_samples_per_atom: Sampling density for surface area (default: 2000)
- vol_samples_per_uc: Sampling density for volume (default: 50000)
- psd_samples_per_uc: Sampling density for PSD (default: same as vol)
- chan_radius_for_sa_vol: Channel radius for SA/Vol calculations
- tasks: List of tasks to run (overrides enabled_tasks)
- do_psd: Deprecated, use tasks parameter
- nproc(Optional[int]): How many processing to run 

##### Returns:

Dictionary with keys as task names and DataFrames as values.

### Function: concat_results

Concatenate multiple task results into a single wide table.

```pycon
def concat_results(
    results: Dict[str, DataFrame],
    tasks: Optional[Sequence[str]] = None,
    keep_units: bool = True,
    chan_mode: str = "expand",
    chan_agg: str = "max",
    include_context: bool = False,
    add_task_suffix_on_collision: bool = True,
    sort_rows: bool = True
) -> DataFrame
```

- **Parameters:**

  | Parameter                      | Type                 | Default  | Description                                              |
  |--------------------------------|----------------------|----------|----------------------------------------------------------|
  | `results`                      | Dict[str, DataFrame] | -        | Output from summarize()                                  |
  | `tasks`                        | Sequence[str]        | None     | Tasks to include (default: all non-PSD)                  |
  | `keep_units`                   | bool                 | True     | Include units in column names                            |
  | `chan_mode`                    | str                  | "expand" | "expand" for per-channel columns, "aggregate" to reduce  |
  | `chan_agg`                     | str                  | "max"    | Aggregation method: "max", "mean", "min", "first", "sum" |
  | `include_context`              | bool                 | False    | Include sampling parameters as columns                   |
  | `add_task_suffix_on_collision` | bool                 | True     | Add task suffix to resolve name conflicts                |
  | `sort_rows`                    | bool                 | True     | Sort output by structure and probe                       |
- **Return:**
  Wide DataFrame with one row per (structure, probe_A) pair.

## Examples

### Example 1: High-Throughput Screening

```pycon
import glob
from zeopp_runner import ZeoppRunner, concat_results

# Find all CIF files
structures = glob.glob("structures/*.cif")

# Initialize runner with specific settings
runner = ZeoppRunner(
    network_bin="network",
    ha=True,
    units_as_rows=True,
    cleanup_outputs=True,
    timeout=300  # 5 minutes per structure
)

# Run analysis for gas separation (N2 and CO2)
results = runner.summarize(
    structures=structures,
    probe_radii=[1.2, 1.65],  # N2 and CO2
    sa_samples_per_atom=2000,
    vol_samples_per_uc=50000,
    tasks=["res", "sa", "vol", "volpo"]
)

# Create feature table
features = concat_results(
    results,
    keep_units=True,
    include_context=False
)

# Export for machine learning
features.to_csv("mof_features.csv", index=False)
```

### Example 2: Detailed Channel Analysis

```pycon
# Analyze channel characteristics
runner = ZeoppRunner(network_bin="network", ha=True)

# Get detailed channel information
df_chan = runner.run_chan("zeolite.cif", probe_radius=1.4)

# Process multiple probe sizes
results = runner.summarize(
    structures=["zeolite.cif"],
    probe_radii=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    tasks=["chan"]
)

# Expand channels into separate columns
wide = concat_results(
    results,
    chan_mode="expand",  # Each channel gets its own columns
    keep_units=True
)

print(wide.columns.tolist())
# Output: ['structure', 'probe_A', 'Di (A) [chan=0]', 'Df (A) [chan=0]', ...]
```

### Example 3: Comparing Materials

```pycon
import pandas as pd
import matplotlib.pyplot as plt

# Compare MOFs
mofs = ["MOF-5.cif", "HKUST-1.cif", "ZIF-8.cif", "MIL-101.cif"]

runner = ZeoppRunner(network_bin="network", ha=True, cleanup_outputs=True)

# Analyze at standard probe size
results = runner.summarize(
    structures=mofs,
    probe_radii=[1.2],
    sa_samples_per_atom=2000,
    vol_samples_per_uc=50000,
    tasks=["res", "sa", "vol", "volpo"]
)

# Get comparison table
comparison = concat_results(results, keep_units=True)

# Visualize key metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Surface area comparison
axes[0, 0].bar(comparison['structure'], comparison['ASA (A^2)'])
axes[0, 0].set_ylabel('Accessible Surface Area (Å²)')
axes[0, 0].set_title('Surface Area Comparison')

# Volume comparison  
axes[0, 1].bar(comparison['structure'], comparison['AV (A^3)'])
axes[0, 1].set_ylabel('Accessible Volume (Å³)')
axes[0, 1].set_title('Volume Comparison')

# Pore size comparison
pore_data = results['res']
axes[1, 0].bar(pore_data.index, pore_data['LCD'])
axes[1, 0].set_ylabel('Largest Cavity Diameter (Å)')
axes[1, 0].set_title('Pore Size Comparison')

plt.tight_layout()
plt.show()
```

### Output Format Details

#### Units as Rows Format

When units_as_rows=True, DataFrames include metadata rows:

```ccs
        structure  probe_A   ASA    AV
0       __var__    probe    ASA    AV
1       __unit__   A        A^2    A^3
2       MOF-5      1.2      3500   12000
3       MOF-5      1.5      3200   11500
```

#### Concatenated Wide Format

The concat_results function produces:

```ccs
structure  probe_A  GLD (A)  PLD (A)  LCD (A)  ASA (A^2)  AV (A^3)  POAV (A^3)
MOF-5      1.2      6.8      8.4      15.1     3500       12000     10500
MOF-5      1.5      6.8      8.4      15.1     3200       11500     9800
ZIF-8      1.2      3.4      4.2      11.6     1240       5420      4850
```

### Troubleshooting

#### Common Issues

1. Network binary not found

```pycon
# Specify absolute path
runner = ZeoppRunner(network_bin="/usr/local/bin/network")
```

2. Timeout errors

```pycon
# Specify absolute path
runner = ZeoppRunner(network_bin="/usr/local/bin/network")
```

3. Memory issues with PSD

```pycon
# Reduce sampling density
results = runner.run_psd("structure.cif", probe_radius=1.2, samples_per_uc=10000)
```

4. Parsing errors

```pycon
# Keep output files for debugging
runner = ZeoppRunner(cleanup_outputs=False)
```
