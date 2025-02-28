import os
import torch

os.chdir("/mnt/d/1-hnh")

# 金属元素的原子序号列表：1到118号的所有金属元素
metal_elements = [
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26,27, 28, 29, 30, 31, 37, 38,
    39, 40, 41,42, 43, 44,45,46, 47, 48, 49, 50,55, 56, 57,58,59,60, 61,62, 63,
    64, 65, 66, 67, 68, 69, 70,71,72,73,74,75,76, 77, 78, 79, 80,81, 82, 83,84,
    87,88, 89,90,91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,102,103,104,105,
    106,107,108,109,110,111,112, 113, 114, 115
]

# 非金属元素的原子序号（O, N, S, P）
non_metal_elements = [8, 7, 16, 15]



def load_data_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            file_path = os.path.join(folder_path, filename)
            data = torch.load(file_path)
            yield data, filename


def process_data(data):

    data_x = data.x[:, 0]

    pair_index = data.pair_index
    edge_index = data.edge_index

    cbond_pair_index = []

    for idx in range(pair_index.shape[1]):
        node1 = pair_index[0, idx].item()
        node2 = pair_index[1, idx].item()

        atom1 = data_x[node1].item() if isinstance(data_x[node1], torch.Tensor) else data_x[node1]
        atom2 = data_x[node2].item() if isinstance(data_x[node2], torch.Tensor) else data_x[node2]

        if (atom1 in metal_elements and atom2 in non_metal_elements) or (
                atom2 in metal_elements and atom1 in non_metal_elements):
            cbond_pair_index.append(idx)

    is_cbond = []
    for idx in cbond_pair_index:
        node1 = pair_index[0, idx].item()
        node2 = pair_index[1, idx].item()

        if torch.any((edge_index[0] == node1) & (edge_index[1] == node2)) or torch.any(
                (edge_index[0] == node2) & (edge_index[1] == node1)):
            is_cbond.append(1)
        else:
            is_cbond.append(0)

    data.cbond_pair_index = torch.tensor(cbond_pair_index)
    data.is_cbond = torch.tensor(is_cbond)

    return data


input_folder = 'mono_data'
output_folder = 'mono_data_cbond'


def process_and_save_data():
    for data, filename in load_data_from_folder(input_folder):
        processed_data = process_data(data)
        output_file_path = os.path.join(output_folder, filename)
        torch.save(processed_data, output_file_path)
        print(f"Processed {filename} and saved to {output_file_path}")


process_and_save_data()
