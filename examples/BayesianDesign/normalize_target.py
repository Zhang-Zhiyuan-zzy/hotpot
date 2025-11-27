"""
@File Name:        normalize_target
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/25 17:20
@Project:          Hotpot
"""
from machines_config import dir_datasets
from hotpot.plugins.ComplexFormer.data_process import normalize_data


if __name__ == '__main__':
    normalizer = normalize_data.DatasetAnalyzer(dir_datasets)
    normalizer.analyze_datasets()

