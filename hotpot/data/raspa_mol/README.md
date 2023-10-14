## Illustration
All files defined under this dir are RASPA mol.def file.

When Hotpot is invoking RASPA2 software to simulate guest molecules where they are given by guest names, 
the Hotpot module will find the guest.def file from the following dirs sequentially:

1) `raspa_root`/share/molecules/Hotpot
2) `raspa_root`/share/molecules/ExampleDefinitions
3) `hotpot_root`/data/raspa_mol

If all above file are missed, an error raise!