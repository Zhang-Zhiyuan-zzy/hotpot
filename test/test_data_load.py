import os
import torch
import pandas as pd

os.chdir("/mnt/d/1-hnh")

# 生成器函数，用于逐个处理 .pt 文件
def process_pt_files(data_folder):
    for filename in os.listdir(data_folder):
        if filename.endswith(".pt"):
            file_path = os.path.join(data_folder, filename)

            try:
                data = torch.load(file_path, map_location=torch.device('cpu'))  # 加载到CPU
                print(f"Loaded {filename}: {data}")

                print(f"Keys in {filename}: {list(data.keys())}")

                if 'y' in data:
                    y = data['y']
                    print(f"Found 'y' in {filename}: {y.shape}")

                    y_values = [y[0, i].item() for i in range(8)]  # 提取 y 的所有元素
                    print(f"Y values in '{filename}': {y_values}")

                    yield filename, y_values
                else:
                    print(f"No 'y' found in {filename}")

            except Exception as e:
                print(f"Error loading {filename}: {e}")


# 主处理函数，使用生成器逐步写入 Excel
def save_y_values_to_excel(data_folder, output_file):
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 创建一个空的 DataFrame 来初始化 Excel
        df_empty = pd.DataFrame(columns=['Filename', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8'])
        df_empty.to_excel(writer, index=False, sheet_name='Y Values')

        data_list = []
        last_row = 1

        for filename, y_values in process_pt_files(data_folder):
            data_list.append([filename] + y_values)

            if len(data_list) >= 1000:
                df = pd.DataFrame(data_list, columns=['Filename', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8'])
                df.to_excel(writer, index=False, sheet_name='Y Values', header=False,
                            startrow=last_row)
                last_row = writer.sheets['Y Values'].max_row  # 更新最后一行
                data_list = []

        if data_list:
            df = pd.DataFrame(data_list, columns=['Filename', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8'])
            df.to_excel(writer, index=False, sheet_name='Y Values', header=False,
                        startrow=last_row)

    print(f"Excel 文件已保存：{output_file}")


data_folder = 'tmqm_data0207_test'  # 数据文件夹
output_file = 'y_values.xlsx'  # 输出文件
save_y_values_to_excel(data_folder, output_file)
