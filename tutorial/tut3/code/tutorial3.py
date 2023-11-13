"""
python v3.9.0
@Project: hotpot
@File   : tutorial3
@Auther : Zhiyuan Zhang
@Data   : 2023/11/13
@Time   : 11:24
"""
import hotpot as hp


if __name__ == '__main__':
    mol1 = hp.Molecule.read_from('../input_files/IRMOF-1.cif')
    mol2 = hp.Molecule.read_from('../input_files/MIL-101(Cr).cif')

    cryst1 = mol1.crystal()
    cryst2 = mol2.crystal()

    pm1 = cryst1.pack_molecule

    pm2 = cryst2.pack_molecule

    # pmol.remove_metals()
    # print(pmol.atoms)
    # print(pmol.coordinates)
    #
    # print(c1, c2)
