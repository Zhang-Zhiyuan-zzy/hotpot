This readme describe the file details for the BADER data.

This directory contains BADER charge analysis data for all train and validation systems in OC20 (for both S2EF and IS2RE/IS2RS tasks). BADER charge analysis is available for the relaxed, or final, frame of each system. Once extracted, the data structure is as follows:

bader/
    | - randomXXXXX
        | - ACF.dat 
        | - AVF.dat 
        | - BCF.dat 
        | - bader_stdout

Where XXXXX corresponds to the system identifier consistent with OC20 (i.e. `sid` in the LMDB objects)
For details on the contents of the `dat` files and Bader calculations, visit http://theory.cm.utexas.edu/henkelman/code/bader/. The standard output for these calculations is also provided as `bader_stdout`.
