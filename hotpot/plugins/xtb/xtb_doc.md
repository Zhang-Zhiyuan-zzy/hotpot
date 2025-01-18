from ase.test.calculator.octopus.test_big import calculate

# Tutorial with Examples

This module offers a python wrapper `XtbCalculator` to run the xtb software and a convenient
function `xtb_batch_run()` to high-throughputly run the xtb

To use this module, you should install xtb software in linux machine. see https://xtb-docs.readthedocs.io/en/latest/setup.html

Check the installation position and the xtb executable file path `$xtb_root/bin/xtb`

## Utilization of `XtbCalculator`
```python
import hotpot as hp
from hotpot.plugins.xtb import XtbCalculator

mol = next(hp.MolReader('c1ccccc1C(=O)O[Sr]'))
calculator = XtbCalculator(
      work_dir='path/to/an/empty/directory',
      xtb_executable='path/to/xtb/executable'
)

calculator.mol = mol  # add hotpot.Molecule object

# Set the molecule charges or number of unpair electrons
calculator.charge = 1
calculator.unpair = 0

# Or, you can assign a default charge and unpair value by calling:
calculator.set_mol_charge_unpairEs()

# Performing XTB calculation
res = calculator.run()
print(res.stdout)  # print results
```
The default task is `single point`. to specify your own tanks, adding the xtb command flag 
(see https://xtb-docs.readthedocs.io/en/latest/commandline.html) For example, if your want 
to perform the structure optimization:
```python
calculator.clear_options()
calculator.options.append('--opt')
calculate.run()
```

## Utilization of `xtb_batch_run()`
Before implementing high-throughputly xtb calculation, you should prepare a bundle of structure
files putting at a directory `struct_dir`. Then calling `xtb_batch_run`:
```python
import hotpot as hp
from hotpot.plugins.xtb import xtb_batch_run

struct_dir = ...
xtb_batch_run(
    mol_file_dir=struct_dir,
    res_file_dir=...,  # An empty dir
    mol_file_pattern='*.mol2',  # This is the default structure pattern
    options=['--ohess']  # add the xtb commandline flag, this the default value if a `None` passes into.
)
```

