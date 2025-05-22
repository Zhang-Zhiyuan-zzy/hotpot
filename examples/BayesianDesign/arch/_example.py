# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : _example
 Created   : 2025/5/16 15:06
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""

from architector import build_complex
from architector import view_structures

inputDict = {
    ################ Core (metal) structure and optional definitions #####################
    # Requires input for what metal and what type of coordination environments to sample #

    "core": {
        "metal": 'Am',
        # "coordList" OR "coreType" OR "coreCN" (Suggested!)
        'coordList': None,
        # Handles user-defined list of core coordination vectors e.g.
        # [
        #     [2., 0.0, 0.0],
        #     [0.0, 2., 0.0],
        #     [0.0, 0.0, 2.],
        #     [-2., 0.0, 0.0],
        #     [0.0, -2., 0.0],
        #     [0.0, 0.0, -2.]
        # ] -> gets defined as 'user_geometry'
        "coreType": None,
        # e.g. 'octahedral' ....
        # or list of coreTypes - e.g. ['octahedral','trigonal_prismatic','tetrahedral']
        # "coreCN": 6  # (SUGGETED!)
        # Core coordination number (CN) (int)
        # Will calculate all possible geometries with the given coreCN
        # Tends to sample the metal space better than other options.
        # OR list of CNs [4,6] -> Will calculate all possible geometries with these CNs.
        # NOTE that if nothing is passed, a list of common coreCNs will be used to attempt structure generation.
    },
    ############## Ligands  list and optional definitions ####################
    # Requires either smiles and metal-coordinating site definitions or default ligand names  #

    "ligands": [
        {"smiles": "CCOP(=O)(c1ccc2c(n1)c1nc(ccc1cc2)P(=O)(OCC)OCC)OCC",
         # Smiles required. Can also be generated and drawn using avogadro molecular editor.
         "coordList": [2, 4, 10, 12, 21, 22, 25, 28],
         # Coordination sitrresponding to the SMILES atom connecting to the metal
         # Can be determined/assigned manually using utils/ligand_viewing_coordinating_atom_selecting.ipynb
         # Alternatively [[0,1],[11,2]], In this case it forces it to be map to the user-defined core coordinating sites.
         # 'ligType': 'bi_cis'
         # Optional, but desirable - if no`t-specified will assign the best ligType guess using a brute force assignment that can be slow.
         },
    ],
    # NOTE - multiple ligands should be added to fill out structure if desired.

    ############## Additional Parameters for the structural generation  ####################
    # Here, metal oxdiation state and spin state, methods for evaluating complexes during construction, #
    # And many other options are defined, but are often handled automatically by Architector in the background #

    "parameters": {
        ######## Electronic parameters #########
        "metal_ox": None,  # Oxidation State
        "metal_spin": None,  # Spin State
        "full_spin": None,  # Assign spin to the full complex (overrides metal_spin)
        "full_charge": None,  # Assign charge to the complex (overrides ligand charges and metal_ox)!

        # Method parameters.
        "full_method": "GFN2-xTB",  # Which  method to use for final cleaning/evaulating conformers.
        "assemble_method": "GFN2-xTB",  # Which method to use for assembling conformers.
        # For very large speedup - use "GFN-FF", though this is much less stable (especially for Lanthanides)
        # Additionaly, it is possible to use "UFF" - which is extremely fast. Though it is recommend to perform an XTB-level optimization
        # for the "full_method", or turn "relaxation" off.
        "xtb_solvent": 'none',  # Add any named XTB solvent!
        "xtb_accuracy": 1.0,  # Numerical Accuracy for XTB calculations
        "xtb_electronic_temperature": 300,  # In K -> fermi smearing - increase for convergence on harder systems
        "xtb_max_iterations": 250,  # Max iterations for xtb SCF.
        "force_generation": False,
        # Whether to force the construction to proceed without xtb energies - defaults to UFF evaluation
        # in cases of XTB outright failure. Will still enforce sanity checks on output structures.
        "ff_preopt": False,
        # Whether to force forcefield (FF) pre-optimization of fully-built complexes before final xtb evalulation.
        # FF preoptimization helps especially in cases where longer carbon chains are present that tend to overlap.
        # This option will also decrease the maximum force tolerance for xtb relaxation to 0.2 and set assemble_method='GFN-FF'
        # By default for acceleration.

        # Covalent radii and vdw radii of the metal if nonstandard radii requested.
        # "vdwrad_metal": vdwrad_metal,
        # "covrad_metal": covrad_metal,

        ####### Conformer parameters and information stored ########
        "n_conformers": 1,  # Number of metal-core symmetries at each core to save / relax
        "return_only_1": False,  # Only return single relaxed conformer (do not test multiple conformations)
        "n_symmetries": 10,  # Total metal-center symmetrys to build, NSymmetries should be >= n_conformers
        "relax": True,  # Perform final geomtetry relaxation of assembled complexes
        "save_init_geos": False,  # Save initial geometries before relaxations.
        "crest_sampling": False,  # Perform CREST sampling on lowest-energy conformer(s)?
        "crest_sampling_n_conformers": 1,
        # Number of lowest-energy Architector conformers on which to perform crest sampling.
        "crest_options": "--gfn2//gfnff --noreftopo --nocross --quick",  # Crest Additional commandline options
        # Note that Charge/Spin/Solvent should NOT be added to crest_options
        # they will be used from the generated complexes and xtb_solvent flags above.
        "return_timings": True,  # Return all intermediate and final timings.
        "skip_duplicate_tests": False,  # Skip the duplicate tests (return all generated/relaxed configurations)
        "return_full_complex_class": False,
        # Return the complex class containing all ligand geometry and core information.
        # "uid": u_id,  # Unique ID (generated by default, but can be assigned)
        "seed": None,  # If a seed is passed (int/float) use it to initialize np.random.seed for reproducability.
        # If you want to replicate whole workflows - set np.random.seed() at the beginning of your workflow.
        # Right not openbabel will still introduce randomness into generations - so it is often valuable
        # To run multiple searches if something is failing.

        # Dump all possible intermediate xtb calculations to separate ASE database
        "dump_ase_atoms": False,  # or True
        "ase_atoms_db_name": 'architector_ase_db_{uid}.json',  # Possible to name the databse filename
        # Will default to a "uid" included name.
        "temp_prefix": "/tmp/",  # Default here - for MPI running on HPC suggested /scratch/$USER/

        ####### Ligand parameters #########
        # Ligand to finish filling out coordination environment if underspecified.
        "fill_ligand": "water",
        # Secondary fill ligand will be a monodentate ligand to fill out coordination environment
        # in case the fill_ligand and specified ligands list cannot fully map to the coordination environment.
        "secondary_fill_ligand": "water",
        # or integer index in reference to the ligand list!!
        "force_trans_oxos": True,  # Force trans configurations for oxos (Useful for actinyls)
        # Will only be activated when actinides are present - otherwise will not force trans oxos.
        "override_oxo_opt": True,  # Override no relaxation of oxo groups (not generally suggested)
        "lig_assignment": 'bruteforce',  # or "similarity" - How to automatically assign ligand types.

        ######### Sanity check parameters ########
        "assemble_sanity_checks": True,  # Turn on/off assembly sanity checks.
        "assemble_graph_sanity_cutoff": 1.8,
        # Graph Sanity cutoff for imposed molecular graph represents the maximum elongation of bonds
        # rcov1*full_graph_sanity_cutoff is the maximum value for the bond lengths.
        "assemble_smallest_dist_cutoff": 0.3,
        # Smallest dist cutoff screens if any bonds are less than smallest_dist_cutoff*sum of cov radii
        # Will not be evaluated by XTB if they are lower.
        "assemble_min_dist_cutoff": 4,
        # Smallest min dist cutoff screens if any atoms are at minimum min_dist_cutoff*sum of cov radii
        # away from ANY other atom (indicating blown-up structure)
        # - will not be evaluated by XTB if they are lower.
        "full_sanity_checks": True,  # Turn on/off final sanity checks.
        "full_graph_sanity_cutoff": 1.7,
        # full_graph_sanity_cutoff can be tightened to weed out distorted geometries (e.g. 1.5 for non-group1-metals)
        "full_smallest_dist_cutoff": 0.55,
        "full_min_dist_cutoff": 3.5,
    }
}


if __name__ == "__main__":
    out = build_complex(inputDict)
    view_structures(out)


