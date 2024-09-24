"""
python v3.9.0
@Project: hotpot
@File   : ml_train
@Auther : Zhiyuan Zhang
@Data   : 2024/8/29
@Time   : 9:30

Define and implement the procedure of training the Machine Learning (ML) model based on dividing treatment methods
"""
import os
import json
import argparse
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from xgboost import XGBRegressor

from hotpot.main.optimize import read_excel
from hotpot.plugins.ml.wf import MachineLearning_

_models = {
    'Regression': {
        'logic': LogisticRegression,
        'LeastSquares': LinearRegression,
        'Ridge': Ridge,
        'SVM': SVR,
        'GBDT': GradientBoostingRegressor,
        'RF': RandomForestRegressor,
        'AdaBoost': AdaBoostRegressor,
        'XGBoost': XGBRegressor
    },

}


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('data_file', type=str, help='The data to train the ML model, given by excel file')
    parser.add_argument(
        '-f', '--features',
        type=str,
        help='A list of features name or feature indices (with -i flag)'
             'the features are separated by comma ",", such as feature1,feature2,feature3 '
             'By default, all columns excluding the last would be treat as the features.'
    )

    parser.add_argument(
        '-ef', '--exclude-features',
        type=str,
        help='A list of feature names or feature indices, excluding from selected list '
             'the features are separated by comma ",", such as feature1,feature2,feature3 '
    )

    parser.add_argument(
        '-t', '--target',
        type=str,
        help='The target column name or feature indices in the data excel file'
    )

    parser.add_argument(
        '-i', '--by-index',
        help='Specify the params by their index in the input excel',
        action='store_true'
    )

    parser.add_argument(
        '-a', '--algorithm', type=str,
        help='Which algorithm to implement to build the model, defaults to XGBoost',
        choices=['SVM', 'RF', 'Ridge', 'GBDT', 'XGBoost', 'DecisionTree', 'LeastSquares', 'KNN', 'AdaBoost'],
        default='XGBoost'
    )

    parser.add_argument(
        '-d', '--work-dir', type=str,
        help='The working directory to save the results, defaults to CURRENT_DIR/work_dir'
    )

    parser.add_argument(
        '-m', '--method', type=str,
        help='Which method to train the model, "Regression(R)", "Classification(R)", "Cluster", or ""',
        default='Regression'
    )

    parser.add_argument(
        '--hyper', type=str,
        help='Path to a json file with hyperparameters'
    )

    parser.add_argument(
        '--example', type=str,
        help="Run the examples",
        choices=['logK_LnAn', 'SF_LnAn'],
    )


def train(args: argparse.Namespace):
    """ Perform the training of the ML model """
    # Assign the work dir
    work_dir = Path(args.work_dir or os.path.join(os.getcwd(), 'work_dir'))
    if not work_dir.exists():
        work_dir.mkdir()

    if args.example == 'logK_LnAn':
        _example_of_stability_constant(work_dir, args)
    elif args.example == 'SF_LnAn':
        _example_of_selective_factors(work_dir, args)
    else:
        features, target = read_excel(args.data_file, args)

        model_class = _models[args.method][args.algorithm]
        if args.hyper is not None:
            hyper = json.load(open(args.hyper, 'r'))
        else:
            hyper = {}
        print(hyper)

        ml = MachineLearning_(
            work_dir=work_dir,
            data=pd.concat([features, target], axis=1),
            features=features.columns.tolist(),
            target=target.columns.tolist() if isinstance(target, pd.DataFrame) else target.name,
            estimator=model_class(**hyper)
        )

        print(features, target)

        ml.work()
        print(ml.features)
        print(ml.valid_score)


def _example_of_stability_constant(work_dir, args):
    """ Run the example to predict the stability constant of metal complexes by building an ML model """
    src_data = Path(__file__).parents[1].joinpath('dataset', 'data', 'examples', 'logβ.xlsx')
    excel_file = pd.ExcelFile(src_data)
    data = excel_file.parse(sheet_name='data', index_col=0)
    test_indices = excel_file.parse(sheet_name='test_indices', index_col=0).values.flatten().tolist()
    essential_features = ['SMR_VSA1', 'BCUT2D_MRHI', 'groups', 'dG', 'Med(MP)', 'BCUT2D_LOGPHI']

    ml = MachineLearning_(
        work_dir=work_dir,
        data=data,
        features=data.columns.tolist()[:-1],
        essential_features=essential_features,
        target=data.columns[-1],
        estimator=GradientBoostingRegressor(),
        test_indices=test_indices,
        recur_addi_prev_threshold=0.1,
        recur_addi_threshold=-0.01
    )

    ml.work()
    print(ml.features)


def _example_of_selective_factors(work_dir, args):
    pass


if __name__ == '__main__':
    # os.system('python ../ ml_train ../dataset/data/logβ1.xlsx --example logK_LnAn')
    _example_of_stability_constant('/home/zz1/wf', 'aa')
