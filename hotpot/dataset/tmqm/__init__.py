"""
python v3.9.0
@Project: hotpot
@File   : __init__.py
@Auther : Zhiyuan Zhang
@Data   : 2024/12/24
@Time   : 15:33
"""
import os
from os import path
from os.path import join as opj

from . import download
from hotpot import Molecule
import csv

_module_root = path.dirname(__file__)
_data_dir = opj(_module_root, 'data')


# define a randon integer generator
# class Gen:
#
#     def __iter__(self):
#         return range(10)
#
# gen = Gen()
# for i in gen:
#     print(i)

#
class _DataGenerator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._xyz_gen = self.get_xyz_generator()
        self._csv_gen = self.get_csv_generator()
        self._q_gen = self.get_q_generator()
        self._BO_gen = self.get_BO_generator()

    def _init_generators(self):
        self._xyz_gen = self.get_xyz_generator()
        self._csv_gen = self.get_csv_generator()
        self._q_gen = self.get_q_generator()
        self._BO_gen = self.get_BO_generator()

    def get_xyz_generator(self):
        """按分子为单位读取XYZ文件"""
        data_dir = opj(self.data_dir, 'tmQM.xyz')
        with open(data_dir, 'r') as file:
            while True:
                # 读取原子数
                line = file.readline().strip()
                if not line:
                    break  # 如果文件结束，则退出循环
                atom_count = int(line)  # 第一行是原子数

                file.readline()  # 跳过第二行注释行

                # 读取原子数据
                atoms = []
                for _ in range(atom_count):
                    atom_line = file.readline().strip()
                    parts = atom_line.split()
                    atom = (parts[0], float(parts[1]), float(parts[2]), float(parts[3]))
                    atoms.append(atom)

                # 跳过可能的空行
                file.readline()  # 读取并忽略空行

                # 使用yield返回一个分子的所有数据
                yield atom_count, atoms


    def get_csv_generator(self):
        """按分子为单位读取csv文件"""
        data_dir = opj(self.data_dir, 'tmQM.csv')
        with open(data_dir, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 跳过第一行（表头）

            for row in reader:
                # 按分号拆分数据，并去掉最后一个元素
                yield [item.strip() for item in row[0].split(';')][:-1]  # 去掉最后一个元素


    def get_q_generator(self):
        """按分子为单位读取q文件"""
        data_dir = opj(self.data_dir, 'tmQm.q')
        try:
            with open(data_dir, 'r', encoding='utf-8') as file:
                current_unit = []  # 存储当前迭代单位的行
                for line in file:
                    line = line.strip()  # 去掉行尾的空白字符

                    # 跳过空行
                    if not line:
                        continue

                    # 如果这一行以 'CSD' 开头，开始一个新的迭代单位，但跳过 'CSD' 行
                    if line.startswith('CSD'):
                        if current_unit:
                            yield current_unit  # 生成之前的一个迭代单位
                        current_unit = []  # 清空当前单位，准备接收新的数据
                        continue  # 跳过这行，不加入 current_unit

                    # 如果这一行以 'Total' 开头，结束当前迭代单位
                    elif line.startswith('Total'):
                        current_unit.append(line)  # 加入 'Total' 行
                        yield current_unit  # 生成当前迭代单位
                        current_unit = []  # 清空当前单位，准备开始下一个
                    else:
                        # 提取每行中的浮动数值（假设每行是由一个字符串和一个浮动数值组成）
                        parts = line.split()
                        if len(parts) == 2:  # 确保每行有两个部分
                            try:
                                float_value = float(parts[1])  # 取第二部分作为浮动数值
                                current_unit.append(float_value)  # 加入浮动数值
                            except ValueError:
                                # 如果第二部分无法转换为浮动数值，跳过该行
                                continue

                # 如果文件结束时仍然有未处理的迭代单位，返回它
                if current_unit:
                    yield current_unit
        except FileNotFoundError:
            print(f"Error: The file {data_dir} does not exist.")
        except IOError as e:
            print(f"Error reading the file {data_dir}: {e}")

    def process_lines(self,lines):
        """将BO文件读取结果进行优化"""
        result = []
        for line in lines:
            # 以空格拆分每行数据
            parts = line.split()
            # 每三个数据作为一个元组
            for i in range(0, len(parts), 3):
                result.append(tuple(parts[i:i + 3]))
        return result

    def get_BO_generator(self):
        """按分子为单位读取BO文件"""
        data_dir = opj(self.data_dir, 'tmQM.BO')
        with open(data_dir, 'rb') as file:
            prev_csd_line = None  # 用于存储上一个前三个字母为 CSD 的行
            lines_between = []  # 用于存储当前 CSD 行和上一行之间的行
            empty_line_count = 0  # 用于记录连续空行的数量

            # 逐行读取文件
            for line in file:
                line = line.decode('utf-8', errors='ignore').strip()  # 假设文件是以 UTF-8 编码，去掉两端空白字符

                if not line:  # 如果是空行
                    empty_line_count += 1
                    if empty_line_count == 2:
                        break  # 如果连续两行为空，停止读取
                    continue  # 如果是空行，继续读取下一行

                empty_line_count = 0  # 重置连续空行计数

                if line[:3] == "CSD":  # 检查当前行的前三个字母是否是 CSD
                    if prev_csd_line is not None:
                        # 如果有上一个 CSD 行，将其与当前行之间的所有行作为一个迭代单位返回
                        yield self.process_lines(lines_between)  # 处理并返回当前 CSD 行之间的所有行
                    # 更新 prev_csd_line，并清空 lines_between
                    prev_csd_line = line
                    lines_between = []  # 重置为当前 CSD 行后的行
                else:
                    # 如果当前行不是 CSD，则将其加入到 lines_between 列表中
                    if prev_csd_line is not None:
                        lines_between.append(line)

            # 文件读取完毕时，处理最后一个 CSD 行与文件末尾的所有行
            if prev_csd_line is not None and lines_between:
                yield self.process_lines(lines_between)  # 处理最后一个 CSD 行后的所有行

    #用于.BO文件获取时防止重复产生键
    @staticmethod
    def _is_target_in_array(arr, target):
        for i, elem in enumerate(arr):
            # 如果当前元素是 None 或空集合，插入目标集合到这个位置
            if elem is None or elem == set():
                arr[i] = target  # 将目标集合插入当前位置
                return False  # 插入后返回 False，因为没有找到匹配

            # 如果目标集合与当前元素匹配，则返回 True
            if elem == target:
                return True

        # 如果没有找到空位置且没有匹配，则返回 False
        return False

    def _get_BO(self, mol: Molecule, bo_res):
        # .BO文件处理
        a1idx = int(bo_res[0][0]) - 1
        set_array = [set() for _ in range(1000)]
        for i in range(1, len(bo_res) - 1):
            if bo_res[i][0].isalpha():
                a2idx = int(bo_res[i][1]) - 1
                target = {a1idx, a2idx}

                if self._is_target_in_array(set_array, target):
                    continue

                else:
                    try:
                        bond = mol._add_bond(int(a1idx), int(a2idx), bond_order=round(float(bo_res[i][2])))
                    except IndexError as e:
                        print(len(mol.atoms))
                        print(int(a1idx), int(a2idx))
                        raise e

                    bond.wiberg_orber = float(bo_res[i][2])

                    for j in range(len(bo_res)):
                        if bo_res[j - 1][0] == a1idx:
                            mol.atoms[int(a1idx)].valence = round(float(bo_res[j - 1][2]))
                            mol.atoms[int(a1idx)].partial_valence = float(bo_res[j - 1][2])

            else:
                a1idx = int(bo_res[i][0]) - 1
                continue



    def next_mol(self):
        atom_counts, xyz = next(self._xyz_gen)
        csv_res = next(self._csv_gen)
        q_res = next(self._q_gen)
        bo_res = next(self._BO_gen)

        # .xyz .q 文件的获取
        mol = Molecule()
        for (sym, x, y, z), q in zip(xyz, q_res):
            mol._create_atom(
                symbol=sym,
                coordinates=(x, y, z),
                partial_charge=q,
            )

        #self._get_BO(mol, bo_res)

        # .csv分子数据
        mol.identifier = csv_res[0]
        mol.energy = float(csv_res[1])
        mol.dispersion = float(csv_res[2])
        mol.Dipole = float(csv_res[3])
        mol.Metal_q = float(csv_res[4])
        mol.Hl = float(csv_res[5])
        mol.HOMO = float(csv_res[6])
        mol.LUMO = float(csv_res[7])
        mol.Polarizability = float(csv_res[8])

        mol.link_atoms()
        return mol

    def iter_mol(self):
        self._init_generators()
        while True:
            yield self.next_mol()

    def __iter__(self):
        return iter(self.iter_mol())

    def __next__(self):
        return self.next_mol()


# 读取tmQm的分子信息，输出Molecule.
class TmQmDataset:
    _dataset_files = ('tmQm.xyz', 'tmQm.q', 'tmQm.csv', 'tmQm.BO')

    def __init__(self):
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

        # 创建 _DataGenerator 实例
        self._generator = _DataGenerator(_data_dir)

    def __iter__(self):
        """使 TmQmDataset 类可迭代，返回 _DataGenerator 的迭代器"""
        return iter(self._generator)

    def __next__(self):
        """返回下一个 Molecule 对象"""
        return next(self._generator)



