"""
python v3.9.0
@Project: hotpot
@File   : __init__.py
@Auther : Zhiyuan Zhang
@Data   : 2024/12/24
@Time   : 15:33
Notes: This script manages TmQm dataset processing and provides utilities to parse and generate molecular data 
from various file formats (XYZ, CSV, Q, BO). It includes classes to handle downloading, file setup, property extraction,  
and molecule creation for further computational tasks.
"""
import os
from os import path
from os.path import join as opj
import multiprocessing as mp

from . import download
from hotpot import Molecule
import csv

_module_root = path.dirname(__file__)
_data_dir = opj(_module_root, 'data')

class DataGenerator:
    """
    Handles the process of generating and iterating over molecular data extracted from multiple file formats.

    The DataGenerator class is responsible for reading molecular data stored in various file formats such as XYZ, 
    CSV, Q, and BO files. It utilizes generator methods to provide data sequentially by molecule, facilitating 
    efficient handling of large datasets. The class aims to support creating molecules with associated properties 
    and bonds based on the parsed data.

    Attributes:
        XYZ_FILE: str
            The filename for the XYZ molecular data.
        CSV_FILE: str
            The filename for the CSV molecular data.
        Q_FILE: str
            The filename for the Q molecular data.
        BO_FILE: str
            The filename for the BO molecular data.
        data_dir: str
            Directory containing the required molecular data files.

    Methods:
        __init__(data_dir: str)
            Initializes the DataGenerator instance with a specified data directory and prepares file generators.
        _initialize_generators()
            Internal method to initialize all file generators for data streaming.
        get_xyz_generator()
            Provides molecular data sequentially from an XYZ file.
        get_csv_generator()
            Provides molecular properties sequentially from a CSV file.
        get_q_generator()
            Provides molecular charge data sequentially from a Q file.
        get_bo_generator()
            Provides bond information data sequentially from a BO file.
        next_mol()
            Combines parsed data from various files to construct and return a molecule object.
        iter_mol()
            Iterates over all molecules by repeatedly combining data from file generators.
        __iter__()
            Returns an iterator for molecule data.
        __next__()
            Returns the next molecule data from the iterator.
    """
    XYZ_FILE = "tmQm.xyz"
    CSV_FILE = "tmQm.csv"
    Q_FILE = "tmQm.q"
    BO_FILE = "tmQm.BO"

    mol_attrs = (
        'identifier', 'energy', 'dispersion', 'dipole', 'metal_q',
        'Hl', 'HOMO', 'LUMO', 'polarizability'
    )

    def __init__(self, data_dir, nproc=1):
        self.data_dir = data_dir
        self._initialize_generators()
        self.nproc = nproc
        self.text2mol_queues = [mp.Queue() for _ in range(nproc)]
        self.result_queues = [mp.Queue() for _ in range(nproc)]

    def _initialize_generators(self):
        """初始化所有生成器"""
        self._xyz_generator = self.get_xyz_generator()
        self._csv_generator = self.get_csv_generator()
        self._q_generator = self.get_q_generator()
        self._bo_generator = self.get_bo_generator()

    def get_xyz_generator(self):
        """按分子为单位读取XYZ文件"""
        xyz_path = os.path.join(self.data_dir, self.XYZ_FILE)
        with open(xyz_path, 'r') as file:
            while True:
                line = file.readline().strip()
                if not line:
                    break
                atom_count = int(line)
                file.readline()  # 跳过注释行
                atoms = [
                    tuple(atom_line.split()[0:4]) for atom_line in
                    (file.readline().strip() for _ in range(atom_count))
                ]
                file.readline()  # 跳过空行
                yield atom_count, atoms

    def get_csv_generator(self):
        """按分子为单位读取CSV文件"""
        csv_path = os.path.join(self.data_dir, self.CSV_FILE)
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 跳过表头
            for row in reader:
                yield [item.strip() for item in row[0].split(';')][:-1]

    def get_q_generator(self):
        """按分子为单位读取Q文件"""
        q_path = os.path.join(self.data_dir, self.Q_FILE)
        try:
            with open(q_path, 'r', encoding='utf-8') as file:
                current_unit = []
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('CSD'):
                        if current_unit:
                            yield current_unit
                        current_unit = []
                    elif line.startswith('Total'):
                        current_unit.append(line)
                        yield current_unit
                        current_unit = []
                    else:
                        self._process_q_line(line, current_unit)
                if current_unit:
                    yield current_unit
        except FileNotFoundError:
            print(f"Error: The file {q_path} does not exist.")
        except IOError as e:
            print(f"Error reading the file {q_path}: {e}")

    @staticmethod
    def _process_q_line(line, current_unit):
        """处理Q文件中的单行数据"""
        parts = line.split()
        if len(parts) == 2:
            try:
                float_value = float(parts[1])
                current_unit.append(float_value)
            except ValueError:
                pass

    def get_bo_generator(self):
        """按分子为单位读取BO文件"""
        bo_path = os.path.join(self.data_dir, self.BO_FILE)
        with open(bo_path, 'rb') as file:
            prev_csd_line = None
            lines_between = []
            empty_line_count = 0
            for line in file:
                line = line.decode('utf-8', errors='ignore').strip()
                if not line:
                    empty_line_count += 1
                    if empty_line_count == 2:
                        break
                    continue
                empty_line_count = 0
                if line.startswith("CSD"):
                    if prev_csd_line is not None:
                        yield self._process_bo_lines(lines_between)
                    prev_csd_line = line
                    lines_between = []
                else:
                    lines_between.append(line)
            if prev_csd_line is not None and lines_between:
                yield self._process_bo_lines(lines_between)

    @staticmethod
    def _process_bo_lines(lines):
        """优化BO文件处理"""
        result = []
        for line in lines:
            parts = line.split()
            for i in range(0, len(parts), 3):
                result.append(tuple(parts[i:i + 3]))
        return result

    @staticmethod
    def _is_target_in_array(arr, target):
        """检查目标元素在数组中是否重复"""
        for i, elem in enumerate(arr):
            if elem is None or elem == set():
                arr[i] = target
                return False
            if elem == target:
                return True
        return False

    def _get_wiberg_val_bo(self, mol: Molecule, bo_res):
        atoms = mol.atoms
        atom_pairs = mol.atom_pairs
        atom1 = None
        for id_sym, sym_id, val_bo in bo_res:
            try:
                a1idx = int(id_sym) - 1
                a1sym = sym_id
                a1val = float(val_bo)

                atom1 = atoms[a1idx]
                assert atom1.symbol == a1sym
                atom1.frac_valence = a1val

            except ValueError:
                a2sym = id_sym
                a2idx = int(sym_id) - 1
                wiberg_bo = float(val_bo)

                atom2 = atoms[a2idx]
                assert atom2.symbol == a2sym
                atom_pairs.getdefault((atom1, atom2)).wiberg_bond_order = wiberg_bo

    def _mp_read_test(self):
        print("Reading test data start...")
        for i, test in enumerate(zip(
                self._xyz_generator,
                self._csv_generator,
                self._q_generator,
                self._bo_generator
        )):
            self.text2mol_queues[i % self.nproc].put(test)

        print("Tests read Done!!")
        for i in range(self.nproc):
            self.text2mol_queues[i].put(None)

    def _mp_mol_construct(self, i: int):
        while True:
            text = self.text2mol_queues[i].get(timeout=20)
            if text is None:
                break
            else:
                (atom_counts, xyz), csv_res, q_res, bo_res = text

            mol = Molecule()
            for (sym, x, y, z), q in zip(xyz, q_res):
                mol._create_atom(
                    symbol=sym,
                    coordinates=(x, y, z),
                    partial_charge=q,
                )

            self._get_wiberg_val_bo(mol, bo_res)

            # .csv分子数据
            mol.identifier = csv_res[0]
            mol.energy = float(csv_res[1])
            mol.dispersion = float(csv_res[2])
            mol.dipole = float(csv_res[3])
            mol.metal_q = float(csv_res[4])
            mol.Hl = float(csv_res[5])
            mol.HOMO = float(csv_res[6])
            mol.LUMO = float(csv_res[7])
            mol.polarizability = float(csv_res[8])

            mol.link_atoms()

            self.result_queues[i].put(mol)

        self.result_queues[i].put(None)

    def mp_iter(self):
        """"""
        p_text = mp.Process(target=self._mp_read_test)
        mol_ps = [mp.Process(target=self._mp_mol_construct, args=(i,)) for i in range(self.nproc)]

        p_text.start()
        for mol_p in mol_ps:
            mol_p.start()

        list_queues = [q for q in self.result_queues]
        while True:
            to_remove = []
            for q in list_queues:
                mol = q.get(timeout=50)
                if mol is None:
                    to_remove.append(q)
                else:
                    yield mol

            for q in to_remove:
                list_queues.remove(q)

            if not list_queues:
                break

    def next_mol(self):
        atom_counts, xyz = next(self._xyz_generator)
        csv_res = next(self._csv_generator)
        q_res = next(self._q_generator)
        bo_res = next(self._bo_generator)

        # .xyz .q 文件的获取
        mol = Molecule()
        for (sym, x, y, z), q in zip(xyz, q_res):
            mol._create_atom(
                symbol=sym,
                coordinates=(x, y, z),
                partial_charge=q,
            )

        self._get_wiberg_val_bo(mol, bo_res)

        # .csv分子数据
        mol.identifier = csv_res[0]
        mol.energy = float(csv_res[1])
        mol.dispersion = float(csv_res[2])
        mol.dipole = float(csv_res[3])
        mol.metal_q = float(csv_res[4])
        mol.Hl = float(csv_res[5])
        mol.HOMO = float(csv_res[6])
        mol.LUMO = float(csv_res[7])
        mol.polarizability = float(csv_res[8])

        mol.link_atoms()
        return mol

    def iter_mol(self):
        self._initialize_generators()
        while True:
            try:
                yield self.next_mol()
            except StopIteration:
                break

    def __iter__(self):
        if self.nproc == 1:
            return iter(self.iter_mol())
        else:
            return iter(self.mp_iter())

    def __next__(self):
        return self.next_mol()


# 读取tmQm的分子信息，输出Molecule.
class TmQmDataset:
    """
    Represents a dataset loader and iterator for the TmQm dataset.

    Handles the initialization, downloading, uncompressing, and management of
    data files for the TmQm dataset. The class also provides iteration capabilities
    for efficiently fetching Molecule objects.

    Attributes:
        _generator: Instance of DataGenerator that provides data iterators.

    Methods:
        __init__: Initializes the TmQmDataset by setting up required files and
        loading the data generator.
        __iter__: Yields an iterator for TmQmDataset.
        __next__: Returns the next Molecule object from the dataset.
    """
    mol_attrs = DataGenerator.mol_attrs
    def __init__(self, nproc=None):
        """"""
        # 如果数据目录不存在，则创建目录并下载文件
        if not path.exists(_data_dir):
            os.mkdir(_data_dir)
            download.download_files()
        else:
            list_data = os.listdir(_data_dir)
            for ext, url in download.urls.items():
                if f'tmQm.{ext}' not in list_data:
                    file_path = opj(_data_dir, f'tmQm.{ext}.gz')
                    download.download_file(file_path, url)
                    download.uncompress_all_gz(file_path)
                    os.remove(file_path)  # 删除压缩包文件
                else:
                    print(f"tmQm.{ext} already exist !!")

        if nproc is None:
            self.nproc = os.cpu_count()
        else:
            self.nproc = nproc

        self.queue = mp.Queue()
        self._generator = DataGenerator(_data_dir, nproc=self.nproc)

    def __iter__(self):
        """使 TmQmDataset 类可迭代，返回 _DataGenerator 的迭代器"""
        return iter(self._generator)

    def __next__(self):
        """返回下一个 Molecule 对象"""
        return next(self._generator)



