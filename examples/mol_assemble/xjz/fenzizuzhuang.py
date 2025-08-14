import os
import hotpot as hp
from hotpot.cheminfo.mol_assemble import AssembleFactory
import pandas as pd

output_dir = "/ihepfs/raq/user/xiaojiaze/JQXX/Environment/Hotpot/output"
os.makedirs(output_dir, exist_ok=True)

assemblers = AssembleFactory.load_assembler_file("Template.json")

frames = [
    hp.read_mol('C1C(CCCC1)C(=NC)N', fmt='smi'),  
    hp.read_mol('C1CCCCC1', fmt='smi'),    
    hp.read_mol('C1=CC=C(C=C1)C(=NO)N', fmt='smi'),
    hp.read_mol('C1=CC=CC=C1', fmt='smi'),
]

factory = AssembleFactory(
    assembler=assemblers,
    iter_step=2,
    catch_path=None,        # 不用 JSON 保存
    save_per_step=10000     # 每 10000 个分子保存一次
)

file_index = 1
batch_data = []

results = factory.make(frames)

for count, mol in enumerate(results, start=1):
    smi = str(mol)  
    batch_data.append([smi, str(mol)])
    
    if count % 10000 == 0:
        df = pd.DataFrame(batch_data, columns=["SMILES", "Molecule"])
        file_path = os.path.join(output_dir, f"{file_index}.csv")
        df.to_csv(file_path, index=False)
        print(f"已保存: {file_path}")
        batch_data = []
        file_index += 1

if batch_data:
    df = pd.DataFrame(batch_data, columns=["SMILES", "Molecule"])
    file_path = os.path.join(output_dir, f"{file_index}.csv")
    df.to_csv(file_path, index=False)
    print(f"已保存: {file_path}")

print("所有分子结果已按批次保存为 CSV 文件。")
