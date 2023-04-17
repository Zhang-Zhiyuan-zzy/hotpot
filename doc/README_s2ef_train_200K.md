# README_s2ef_train_200K


This readme describes the file details for the S2EF task for the 200K split. The data here corresponds to the “200K” training set, with a total of 200K frames.
 
The `s2ef_train_200K/` folder contains:

* 40 compressed trajectory files and are of the format `.extxyz.xz`
* 40 compressed text files and are of the format `.txt.xz`

Files are named as follows:  `<num>.<extxyz/txt>.xz` where `<num>` is a 0-indexed counter.

The `.extxyz.xz` files are LZMA compressed `.extxyz` trajectory files, which contain 5000 structures per file. Each of the structures is from a different adsorbate+catalyst system. Information about the `.extxyz` trajectory file format may be found at https://wiki.fysik.dtu.dk/ase/dev/_modules/ase/io/extxyz.html .

The uncompressed `.txt` files (one per corresponding `.extxyz` file) have information about each structure in the `.extxyz` files with the following format:
`system_id,frame_number,reference_energy`

where:

* `system_id `- Internal random ID corresponding to an adsorbate+catalyst system.
* `frame_number` - Index along `system_id's` relaxation trajectory for which snapshot was taken. Can be ignored.
* `reference_energy` - Energy used to reference system energies to bare catalyst+gas reference energies. Used for adsorption energy calculations.


The number of lines in a `.txt` file is guaranteed to be equal to the number of structures in the corresponding `.extxyz` file. The details on `system_id` / `frame_number` / `reference_energy` have a one-to-one correspondence with the respective structure in the `.extxyz` file.

For example: `line #10` in `123.txt` will have details about index #10 in `123.extxyz`


Please refer to the `preprocessing` section in `README.md` to check out more details about how to preprocess these files and use them for training your models.



