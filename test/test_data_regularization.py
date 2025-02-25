import os
import torch
import math

os.chdir("/mnt/d/1-hnh")

# 生成器函数：逐个加载文件
def process_pt_files(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".pt"):
            file_path = os.path.join(input_folder, filename)
            try:
                # 加载 .pt 文件
                data = torch.load(file_path)
                yield filename, data
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# 生成器函数：修改 y 的值
def modify_y_values(input_folder, output_folder):
    for filename, data in process_pt_files(input_folder):
        if 'y' in data:
            y = data['y']

            if y.ndimension() == 2 and y.shape[1] >= 8:
                y[0, 0] = (y[0, 0] + 29008.53125) / 28713.2876892

                y[0, 1] = (y[0, 1] + 0.637394011) / 0.632188011

                y[0, 2] = math.log(1 + ((y[0, 2] / 106.4645996) * (math.e - 1)))

                y[0, 3] = (y[0, 3] + 3.08336997) / 5.41350007

                y[0, 4] = (y[0, 4] - 0.00157) / 0.305849986

                y[0, 5] = (y[0, 5] + 0.442030013) / 0.508570016

                y[0, 6] = (y[0, 6] + 0.371989995) / 0.570179999

                y[0, 7] = (y[0, 7] - 51.24996185) / 1505.72025315

                output_path = os.path.join(output_folder, filename)
                torch.save(data, output_path)
                print(f"Saved modified {filename} to {output_path}")
            else:
                print(f"Skipping {filename} because 'y' shape is incorrect.")
        else:
            print(f"No 'y' found in {filename}")


def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    modify_y_values(input_folder, output_folder)


input_folder = 'tmqm_data0207'  # 输入文件夹路径
output_folder = 'tmqm_data0224_minmax'  # 输出文件夹路径
main(input_folder, output_folder)
