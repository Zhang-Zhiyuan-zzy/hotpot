import os
import torch

def process_data(data):
    """处理数据，生成cbond_index并删除cbond_pair_index"""
    pair_index = data['pair_index']
    cbond_pair_index = data['cbond_pair_index']

    cbond_pair_index = cbond_pair_index.long()

    cbond_index = pair_index[:, cbond_pair_index]

    data['cbond_index'] = cbond_index

    del data['cbond_pair_index']

    return data


def file_generator(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.pt'):
            file_path = os.path.join(input_folder, filename)
            data = torch.load(file_path,weights_only=False)
            yield filename, data


def process_files(input_folder, output_folder):
    for filename, data in file_generator(input_folder):
        try:
            processed_data = process_data(data)

            output_path = os.path.join(output_folder, filename)
            torch.save(processed_data, output_path)
            print(f"Processed and saved: {filename}")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

input_folder = '/home/hnh/proj/datasets/mono/train_false2'
output_folder = '/home/hnh/proj/datasets/mono/train'

process_files(input_folder, output_folder)
