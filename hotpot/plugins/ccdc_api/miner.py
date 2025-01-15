"""
python v3.9.0
@Project: hotpot0.5.0
@File   : miner
@Auther : Zhiyuan Zhang
@Data   : 2024/6/4
@Time   : 21:00
Note: This module requires CCDC Python API, see https://www.ccdc.cam.ac.uk/
"""
import os
from copy import copy
import time
from os.path import join as opj
import csv
from operator import attrgetter
import multiprocessing as mp

from tqdm import tqdm
import ccdc
from ccdc import io
from ccdc.molecule import Atom, Molecule
from ccdc.crystal import Crystal


def _has_metal(mol: Molecule) -> bool:
    """
    Determines whether the given molecule contains any metallic atoms.

    This function iterates through all atoms in the provided molecule's
    structure and checks if any of those atoms are metallic. It returns
    a boolean result indicating the presence of metallic atoms.

    :param mol: A molecule object, expected to be an instance of a class
                with an attribute `atoms` containing a list of atom objects.
    :type mol: Molecule
    :return: True if the molecule contains at least one atom classified
             as a metal, otherwise False.
    :rtype: bool
    """
    return any(a.is_metal for a in mol.atoms)


def _remove_component(mol, component):
    """
    Removes a specified component from the molecule by identifying and deleting
    its associated atoms. The atoms are identified through their labels, and
    the method ensures that all atoms in the specified component are accurately
    removed from the molecule.

    :param mol: The molecule object from which the component is to be removed.
    :type mol: Molecule
    :param component: The component object containing the atoms that need to
        be removed from the molecule.
    :type component: Component
    :return: None
    """
    mol.remove_atoms([mol.atom(a.label) for a in component.atoms])


def _remove_isolated_atoms(mol):
    """
    Removes isolated atoms from a molecule.

    This method iterates through the atoms in a molecule and identifies
    those that have no neighbors. It collects these isolated atoms in a
    list and subsequently removes them from the molecule.

    Parameters:
    mol : Any
        The molecule object which holds the atoms to be checked and
        modified.

    Returns:
    None

    Raises:
    None
    """
    to_remove = []
    for atom in mol.atoms:
        if not atom.neighbours:
            to_remove.append(atom)

    mol.remove_atoms(to_remove)


def _process_complexes_with_polymeric_bonds(complexes, crystal, cluster_metal):
    """
    Processes complexes that include polymeric bonds within a crystal structure. This function modifies
    the provided complexes by performing operations like polymer expansion based on a given cluster
    metal, removing excess atoms, and normalizing labels. Components without any metallic atoms are
    also removed during the process.

    Args:
        complexes (Molecule): The molecular object representing the complexes to be processed.
        crystal (Crystal): The crystal object containing the structural data for polymer expansion.
        cluster_metal (list[Atom]): A list of metal atoms used as a reference during polymer expansion
            and modifications.

    Raises:
        None explicitly raised by this function.

    Returns:
        None. The provided complexes are modified in place.
    """
    crystal.molecule = complexes
    complexes = crystal.polymer_expansion(cluster_metal)
    complexes = Molecule(complexes.identifier, complexes._molecule.create_editable_molecule())
    complexes.remove_atoms([a for a in complexes.atoms if a.is_metal][len(cluster_metal):])

    complexes.normalise_labels()
    components_remove = [c for c in complexes.components if not any(a.is_metal for a in c.atoms)]
    remove_atoms = [complexes.atom(a.label) for c in components_remove for a in c.atoms]
    complexes.remove_atoms(remove_atoms)


def _process_molecule(mol: Molecule, crystal: Crystal, is_polymeric: bool):
    """
    Processes a molecule to extract complexes.

    This function normalizes the labels of the input molecule, identifies metal atoms and their
    neighbors, and processes cloned versions of the molecule to create a list of complexes.
    Each complex is based on a specific metal and its connected components, ensuring no
    non-neighboring components are included. All processed complexes are assigned unique identifiers.

    :param mol: Molecule object to be processed.
                Must contain normalized atoms with labels and information
                about metal atoms and their neighbors.
    :type mol: Molecule
    :return: A list of molecule complexes, each associated with a specific metal atom.
    :rtype: list[Molecule]
    """
    identifier = mol.identifier
    mol.normalise_labels()

    clusters = _identify_metal_cluster(mol)

    list_complexes = []
    for cluster in clusters:
        complexes = mol.copy()
        cluster_metal = [complexes.atom(m.label) for m in cluster]

        metal_remove = [a for a in complexes.atoms if a.is_metal and a not in cluster_metal]
        complexes.remove_atoms(metal_remove)

        components_remove = [c for c in complexes.components if any(m not in c.atoms for m in cluster_metal)]
        for c in components_remove:
            _remove_component(complexes, c)

        if complexes.is_polymeric:
            _process_complexes_with_polymeric_bonds(complexes, crystal, cluster_metal)

        complexes.assign_bond_types('unknown')
        complexes.add_hydrogens('missing')
        complexes.identifier = f"{identifier}_"+ '.'.join([f"{m.label}" for m in cluster])
        list_complexes.append(complexes)

    return list_complexes


def _clear_complex_metals(complexes: Molecule, central_metal: Atom):
    """
    Remove unwanted atoms and metals from a molecular complex.

    This function modifies a molecular complex by removing isolated atoms,
    unnecessary metallic atoms that are not centrally bonded to a specified
    central metal atom, or are not part of nearby atomic rings. It ensures
    the molecular structure retains only relevant atoms associated with the
    central metal atom and its nearby structures.

    Parameters:
        complexes (Molecule): The molecular complex in which atoms and metals
                              will be analyzed and removed if necessary.
        central_metal (Atom): The central metal atom around which validation
                              of related atoms and metals is performed.

    Raises:
        None

    Returns:
        None
    """
    # Remove all isolate atoms
    complexes.remove_atoms([a for a in complexes.atoms if (not a.neighbours)])

    # Remove metals
    other_metals = [a for a in complexes.atoms if a.is_metal and a.label != central_metal.label]
    rings_atoms = {a for r in central_metal.rings for a in r.atoms}
    to_remove = []
    for other_metal in other_metals:
        if other_metal not in rings_atoms:
            to_remove.append(other_metal)

        elif not (0 < complexes.shortest_path(central_metal, other_metal) <=4):
            to_remove.append(other_metal)

    complexes.remove_atoms(to_remove)


def _identify_metal_cluster(mol: Molecule):
    """
    Identify metal clusters within a given molecule.

    This function analyzes the given molecule and identifies groups of connected
    metal atoms, referred to as "metal clusters". A metal atom is defined by the
    `is_metal` attribute of the molecule's atoms. Clusters are determined based on
    a combination of ring membership and path length between metals.

    Parameters:
        mol (Molecule):
            The molecule to analyze for metal clusters. This object must include
            attributes like `atoms`, `rings`, and methods like `shortest_path()`.

    Returns:
        list[list[Atom]]:
            A list of metal clusters, where each cluster is a list of `Atom`
            objects representing connected metal atoms.

    Raises:
        None
    """
    metals = [a for a in mol.atoms if a.is_metal]
    clusters = []

    while metals:
        cluster = [metals.pop()]

        while True:
            to_add = []
            for other_metal in metals:
                if any(
                        any(other_metal in r for r in m.rings) and 0 < mol.shortest_path(m, other_metal) < 4
                        for m in cluster
                ):
                    to_add.append(other_metal)

            if to_add:
                cluster.extend(to_add)
                for m in to_add:
                    metals.remove(m)
            else:
                break

        clusters.append(cluster)

    return clusters


class EntryReader:
    def __init__(self, start: int = 0):
        self._reader = io.EntryReader('CSD')
        self.start = start
        self.index = start

    def __len__(self):
        return len(self._reader) - self.start

    def __next__(self):
        if self.index >= len(self._reader):
            raise StopIteration

        index = self.index
        self.index = self.index + 1
        return self._reader[index]


    def __iter__(self):
        reader = io.EntryReader('CSD')
        for i in range(self.start, len(reader)):
            yield reader[i]

    def refresh(self):
        self.index = self.start


def extract_metal_complexes(
        output_dir: str,
        fmt: str = 'mol2',
        nproc: int = None,
        timeout: float = 100,
):
    """
    Extracts metal complexes from a given dataset and writes them to a specified output directory
    in the specified format. This function supports parallel processing using multiprocessing
    and manages the lifecycle of multiple processes to ensure optimal execution within a given
    timeout period.

    Arguments:
        output_dir (str): The directory where extracted metal complexes will be saved.
        fmt (str): The file format to use for saving metal complexes. Defaults to 'mol2'.
        nproc (int, optional): Number of parallel processes to use. Defaults to the number of
            available CPU cores if not specified.
        timeout (float): Maximum time allowed for a process to run in seconds. Defaults to 100.

    Raises:
        RuntimeWarning: Raised when a process exceeds the timeout.
        UserWarning: Raised if a RuntimeError occurs while iterating over entries in the dataset.
    """
    if nproc is None:
        nproc = mp.cpu_count()

    reader = iter(tqdm(io.EntryReader('CSD'), desc='Extracting metal complexes'))
    count = 0
    while count < 47018:
        _ = next(reader)
        count += 1

    if nproc == 1:
        for entry in reader:
            _extract_metal_complexes(entry, output_dir, fmt)
        return

    processes = {}
    previous_time = time.time()
    while True:
        while len(processes) >= nproc:
            to_remove = []
            for process, t0 in processes.items():
                if not process.is_alive():
                    to_remove.append(process)
                elif time.time() - t0 > timeout:
                    to_remove.append(process)
                    print(RuntimeWarning(f"{process} timeout: {time.time() - t0} seconds"))

            if to_remove:
                for rm_process in to_remove:
                    rm_process.terminate()
                    del processes[rm_process]
            else:
                time.sleep(0.01)

        try:
            entry = next(reader)
        except StopIteration:
            break
        except RuntimeError:
            print(UserWarning('RuntimeError happened in reader Entry'))
            continue


        # Waiting
        while time.time() - previous_time < 0.025:
            time.sleep(0.001)

        process = mp.Process(
            target=_extract_metal_complexes,
            args=(entry, output_dir, fmt)
        )
        process.start()
        previous_time = processes[process] = time.time()

    for process in processes:
        process.join()
        process.terminate()


def extract_metal_complexes_(
        output_dir: str,
        fmt: str = 'mol2',
        nproc: int = None,
        timeout: float = 20,
):
    """
    Extracts metal complexes from a given dataset and writes them to a specified output directory
    in the specified format. This function supports parallel processing using multiprocessing
    and manages the lifecycle of multiple processes to ensure optimal execution within a given
    timeout period.

    Arguments:
        output_dir (str): The directory where extracted metal complexes will be saved.
        fmt (str): The file format to use for saving metal complexes. Defaults to 'mol2'.
        nproc (int, optional): Number of parallel processes to use. Defaults to the number of
            available CPU cores if not specified.
        timeout (float): Maximum time allowed for a process to run in seconds. Defaults to 100.

    Raises:
        RuntimeWarning: Raised when a process exceeds the timeout.
        UserWarning: Raised if a RuntimeError occurs while iterating over entries in the dataset.
    """
    if nproc is None:
        nproc = mp.cpu_count()

    # reader = iter(tqdm(io.EntryReader('CSD'), desc='Extracting metal complexes'))
    # if nproc == 1:
    #     for entry in reader:
    #         _extract_metal_complexes(entry, output_dir, fmt)
    #     return

    reader = io.EntryReader('CSD')
    index = 69296
    render_length = len(reader)
    p_bar = tqdm(total=render_length, desc='Extracting metal complexes', initial=index)

    processes = {}
    previous_time = time.time()
    while index < render_length:
        if len(processes) < nproc:
            try:
                entry = reader[index]
                index += 1
                p_bar.update(1)
            except RuntimeError:
                index += 1
                p_bar.update(1)

            process = mp.Process(
                target=_extract_metal_complexes,
                args=(entry, output_dir, fmt)
            )
            process.start()
            previous_time = processes[process] = time.time()

        to_remove = []
        for process, t0 in processes.items():
            if not process.is_alive():
                to_remove.append(process)
            elif time.time() - t0 > timeout:
                process.terminate()
                to_remove.append(process)
                print(RuntimeWarning(f"{process} timeout: {time.time() - t0} seconds"))

        if to_remove:
            for process in to_remove:
                del processes[process]
        elif len(processes) > nproc:
            if time.time() - previous_time > 20:
                print(f'Block time: {time.time() - previous_time}')
            time.sleep(0.1)

    for process in processes:
        process.join()
        process.terminate()


def _extract_metal_complexes(
        entry,
        output_dir: str,
        fmt: str = 'mol2'
) -> None:
    """
    Extracts metal complexes from a given molecular entry and writes them to a specified
    output directory in the desired format. The function checks for the presence of a
    3D structure and ensures that the molecule does not contain disorders before
    processing. If the molecule contains metal atoms, it processes the molecule to
    identify metal complexes and writes them using an appropriate molecule writer.

    Args:
        entry: The molecular entry to extract metal complexes from.
        output_dir: str
            The directory where the extracted metal complexes will be saved.
        fmt: str, optional
            The format in which to save the extracted metal complexes
            (default is 'mol2').

    Returns:
        None
    """
    try:
        if not entry.has_3d_structure:
            return

        if entry.has_disorder:
            return

        mol = entry.molecule
        if not _has_metal(mol):
            return

        list_complexes = _process_molecule(mol, entry.crystal, entry.is_polymeric)

        for complexes in list_complexes:
            with io.MoleculeWriter(opj(output_dir, f"{complexes.identifier}.{fmt}")) as writer:
                writer.write(complexes)

    except RuntimeError:
        return

    return


def extract_entry_info():
    entry_attrs = [
        'identifier',
        'heat_capacity',
        'heat_of_fusion',
        'melting_point',
        'r_factor'
    ]

    crystal_attrs = [
        'calculated_density',
        'cell_volume',
        'crystal_system'
        "spacegroup_symbol"
    ]

    entry_getter = attrgetter(*entry_attrs)
    crystal_getter = attrgetter(*crystal_attrs)


if __name__ == '__main__':
    extract_metal_complexes_('/home/zzy/mol2', 'mol2', nproc=1)
    # _extract_metal_complexes(
    #     io.EntryReader('CSD'),
    #     '/home/zzy/mol2',
    #     'mol2'
    # )
