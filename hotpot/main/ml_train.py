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
from typing import Literal
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor, sklearn

from hotpot.main.optimize import read_excel
from hotpot.plugins.ml.wf import MachineLearning_, LinearAddLightGBM

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
    parser.add_argument('data_file', type=str, help='The ChemData to train the ML model, given by excel file')
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
        help='The target column name or feature indices in the ChemData excel file'
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
        '--examples', type=str,
        help="Run the examples",
        choices=['logK_LnAn', 'SF_LnAn'],
    )


def train(args: argparse.Namespace):
    """ Perform the training of the ML model """
    # Assign the work dir
    work_dir = Path(args.work_dir or os.path.join(os.getcwd(), 'work_dir'))
    if not work_dir.exists():
        work_dir.mkdir()

    if args.examples == 'logK_LnAn':
        _example_of_stability_constant(work_dir, args)
    elif args.examples == 'SF_LnAn':
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


def _data_for_LnAn_example(
        which: Literal['All', 'LnAn'] = 'All',
        medium: list[str] = None,
        ligand_charge: list[int] = None,
):
    src_data = Path(__file__).parents[1].joinpath('dataset', 'ChemData', 'examples', 'logβ.xlsx')
    excel_file = pd.ExcelFile(src_data)
    data_ = excel_file.parse(sheet_name='ChemData', index_col=0)
    test_indices_ = excel_file.parse(sheet_name='test_indices', index_col=0).values.flatten().tolist()
    LnAn_indices_ = pd.read_excel(src_data, sheet_name='LnAn_only', index_col=0).values.flatten()
    MA_indices_ = pd.read_excel(src_data, sheet_name='MA_only', index_col=0).values.flatten()
    other_info = pd.read_excel(src_data, sheet_name='other_info', index_col=0)

    if which == 'LnAn':
        if isinstance(medium, list):
            media = np.where(np.isin(other_info['Media'].values, medium))[0]
            LnAn_indices_ = np.intersect1d(LnAn_indices_, media)

        if isinstance(ligand_charge, list):
            charges = np.where(np.isin(np.int_(data_['Ligand_Charges'].values), ligand_charge))[0]
            LnAn_indices_ = np.intersect1d(LnAn_indices_, charges)


        data_ = data_.loc[LnAn_indices_, :]
        test_indices_ = np.where(np.isin(LnAn_indices_, test_indices_))[0]
        MA_indices_ = np.where(np.isin(LnAn_indices_, MA_indices_))[0]

    return data_, test_indices_, MA_indices_


def _coef_stability_analysis(which: Literal['All', 'LnAn'] = 'LnAn'):
    from hotpot.plugins.ml.wf import linear_leave_one_out_analysis
    data, test_indices, MA_indices = _data_for_LnAn_example(
        which,
        medium=['NaClO4', 'Et4NNO3'],
        ligand_charge=[0]
    )
    essential_features = ['dG', 'c1', 'SMR_VSA1']
    X = data[essential_features].values
    y = data.iloc[:, -1].values.flatten()

    coff, intercept = linear_leave_one_out_analysis(X, y)

    mean_coef = np.mean(coff, axis=0)
    std_coef = np.std(coff, axis=0)

    mean_intercept = np.mean(intercept)
    std_intercept = np.std(intercept)

    print('Mean Coefficients: ', mean_coef)
    print('Std Coefficients: ', std_coef)
    print('Mean Intercept: ', mean_intercept)
    print('Std Intercept: ', std_intercept)

    with pd.ExcelWriter('/home/zz1/wf/Linear/coef_intercept.xlsx') as writer:
        pd.DataFrame(coff, columns=essential_features).to_excel(writer, sheet_name='coef')
        pd.Series(intercept, name='intercept').to_excel(writer, sheet_name='intercept')


def _train_LnAn_logK_by_linear_lightgbm(which: Literal['All', 'LnAn'] = 'LnAn'):
    # from hotpot.plots import R2Regression, SciPlotter
    import shap
    from hotpot.plugins.plots import SciPlotter, R2Regression, Hist, FeatureImportance

    data, test_indices, MA_indices = _data_for_LnAn_example(
        which,
        medium=['NaClO4', 'Et4NNO3'],
        # ligand_charge=[0]
    )

    essential_features = ['ΔG', 'c1', 'SMR_VSA1', 'Med(MP)', 'Med(c)']
    min_offset = np.array([1., 0., -0.0, 0., 0.])
    max_offset = np.array([1., 0., -0.5, 0., 0.])
    linear_index = [0, 1, 2]
    # linear_index = [0]

    X = data[essential_features].values
    y = data.iloc[:, -1].values.flatten()

    # Directly fitting
    gbdt_LnAn = GradientBoostingRegressor()
    gbdt_LnAn.fit(X, y)

    linear_gbm_model = LinearAddLightGBM()

    cv_pred = cross_val_predict(linear_gbm_model, X, y, cv=KFold(100, shuffle=True))
    r2 = r2_score(y, cv_pred)
    mae = mean_absolute_error(y, cv_pred)
    rmse = np.sqrt(mean_squared_error(y, cv_pred))

    plotter = SciPlotter(R2Regression([y, cv_pred], show_mae=True, show_rmse=True, to_sparse=True))
    fig, ax = plotter()
    fig.savefig('/home/zz1/wf/Linear/cv_LnMA_LinearGBM.png')

    print('R2 score: ', r2)
    print('MAE: ', mae)
    print('RMSE: ', rmse)

    linear_gbm_model.fit(X, y, linear_feature_index=linear_index)
    print(linear_gbm_model.linear_coeff_)
    print(linear_gbm_model.linear.intercept_)

    train_pred = linear_gbm_model.predict(X)
    train_r2 = r2_score(y, train_pred)
    train_mae = mean_absolute_error(y, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y, train_pred))

    print(f"Train R2: {train_r2}")
    print(f"Train MAE: {train_mae}")
    print(f"Train RMSE: {train_rmse}")

    explainer, shap_value = MachineLearning_.shap_analysis_(
        linear_gbm_model, X, y,
        feature_names=essential_features,
        # sample_size=200,
        test_size=200,
        # X_test=X,
        explainer_cls=shap.Explainer,
        shap_values_save_path='/home/zz1/wf/Linear/gbdt_LnAn_shap.xlsx',
        min_offset=min_offset,
        max_offset=max_offset
    )
    MachineLearning_.make_shap_bar_beeswarm(shap_value, savefig_path='/home/zz1/wf/Linear/gbdt_LnAn_shap.png')


    imp = linear_gbm_model.lightgbm.feature_importances_
    plotter = SciPlotter(FeatureImportance(essential_features, imp))
    fig, ax = plotter()
    fig.savefig('/home/zz1/wf/Linear/feat_imp_nonlinear.png')
    imp = pd.Series(imp, index=essential_features)
    imp.to_excel('/home/zz1/wf/Linear/feat_imp.xlsx')

    pred_lin = linear_gbm_model.linear.predict(X[:, 0:3])
    error_lin = y - pred_lin

    error = y - cv_pred
    plotter = SciPlotter(np.array([[Hist(error_lin, 'Linear Error', bins=45)], [Hist(error, 'Prediction Error', bins=45)]]))
    fig, ax = plotter()
    fig.savefig('/home/zz1/wf/Linear/error_hist.png')

    explainer, shap_value = MachineLearning_.shap_analysis_(
        linear_gbm_model.lightgbm, X, linear_gbm_model.delta_y,
        feature_names=essential_features,
        test_size=200,
        gen_X_train=True,
        # sample_size=200,
        min_offset = np.array([-0.25, -0.25, 0.0, 0., 0.]),
        max_offset = np.array([-0.25, 0., 0.0, 0., 0.]),
        X_test=X,
        shap_values_save_path='/home/zz1/wf/Linear/shap.xlsx'
    )

    MachineLearning_.make_shap_bar_beeswarm(shap_value, savefig_path='/home/zz1/wf/Linear/shap.png')

    error = pd.Series(error)
    error.to_excel('/home/zz1/wf/Linear/error.xlsx')


def _example_of_stability_constant(work_dir, args):
    """ Run the example to predict the stability constant of metal complexes by building an ML model """

    # src_data = Path(__file__).parents[1].joinpath('dataset', 'ChemData', 'examples', 'logβ.xlsx')
    # excel_file = pd.ExcelFile(src_data)
    # ChemData = excel_file.parse(sheet_name='ChemData', index_col=0)
    # test_indices = excel_file.parse(sheet_name='test_indices', index_col=0).values.flatten().tolist()
    # essential_features = ['SMR_VSA1', 'BCUT2D_MRHI', 'groups', 'dG', 'Med(MP)', 'BCUT2D_LOGPHI']
    # essential_features = ['SMR_VSA1', 'c1', 'Med(MP)', 'Med(c)', 'dG']
    # essential_features = ['SMR_VSA1', 'c1', 'Med(MP)', 'Med(c)', 'dG', 'Ions Radius']
    # essential_features = ['SMR_VSA1', 'c1', 'Med(MP)', 'Med(c)', 'dG', 'groups']
    # essential_features = ['SMR_VSA1', 'c1', 'Med(MP)', 'Med(c)', 'groups', 'Ions Radius']
    essential_features = ['SMR_VSA1', 'c1', 'Med(MP)', 'Med(c)', 'dG', 'groups', 'Ions Radius']
    # essential_features = ['SMR_VSA1', 'c1', 'Med(MP)', 'Med(c)', 'dG', 'groups', 'Ions Radius', 'Charges']
    # essential_features = ['SMR_VSA1', 'c1', 'Med(MP)', 'Med(c)', 'dG', 'Ions Radius', 'Charges']
    # essential_features = ['SMR_VSA1', 'BCUT2D_MRHI', 'BCUT2D_MRHI', 'Med(MP)', 'Med(c)']

    data, test_indices, MA_indices = _data_for_LnAn_example('All')
    # ChemData, test_indices, MA_indices = _data_for_LnAn_example('LnAn')

    # ml = MachineLearning_(
    #     work_dir=work_dir,
    #     ChemData=ChemData,
    #     features = ChemData.columns.tolist()[:-1],
    #     target = ChemData.columns[-1],
    #     estimator = GradientBoostingRegressor(),
    #     test_indices = test_indices,
    #     recur_addi_prev_threshold = 0.1,
    #     recur_addi_threshold = -0.01,
    # )
    # ml.feature_engineering()

    # ChemData, test_indices, MA_indices = _example_of_stability_constant_only_LnAn(ChemData, src_data)

    ml = MachineLearning_(
        work_dir=work_dir,
        data=data,
        features=data.columns.tolist()[:-1],
        essential_features=essential_features,
        target=data.columns[-1],
        estimator=GradientBoostingRegressor(),
        # estimator=Ridge(),
        test_indices=test_indices,
        recur_addi_prev_threshold=0.1,
        recur_addi_threshold=-0.01,
        dir_exists=True,
        highlight_sample_indices=MA_indices
    )

    # ml.essential_features = essential_features
    ml.kwargs['dir_exists'] = True

    ml.work()
    print(ml.features)


def _example_of_selective_factors(work_dir, args):
    pass


if __name__ == '__main__':
    # os.system('python ../ ml_train ../dataset/ChemData/logβ1.xlsx --example logK_LnAn')
    # _example_of_stability_constant('/home/zz1/wf', 'aa')
    _train_LnAn_logK_by_linear_lightgbm()
    # _coef_stability_analysis()
