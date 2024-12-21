# Machine Learning workflow

## API Documentation

<h3><font color=#888888>Discarded ↓↓↓↓↓↓↓↓ </font></h3>
<h4><font color=#b8860s>**class**</font> MachineLearning(work_dir: Pathlike, 
data: pandas.DataFrame, features: Union[Sequence[str], numpy.ndarray[str]], 
target: Union[str, list[str]], estimator: Optional=GradientBoostingRegressor, 
**kwargs)</h4>
A convenient interface to perform a Machine learning (based on the divide and conquer method) procedure.
- **work_dir(Path|str)**: the working directory to save result from the running procedure.
- **data(pd.DataFrame)**: the total dataset, including features, target, and other correlation information.
- **features(ArrayLike[str], Sequence[str])**: feature names in the given data DataFrame.
- **target(str, Sequence[str])**: target name(s) in data DataFrame.
- **estimator(BaseEstimator)**: algorithm instance with method: fit(), predict(), and score()

Keyword Arguments:

- **xscaler**(sklean.preprocessing.BaseScaler):
- **yscaler**(sklean.preprocessing.BaseScaler):
- **test_size**(float|int): the size of test set.
- **test_indices**(float|int): specify which sample should be assigned to the test set. If specified, the process
  of dataset split will skip. This argument is useful to reproduce the previous works.
- **data_shuffle**(bool): whether to shuffle the `data` when split the dataset.
- **data_stratify**(bool): Whether to perform stratified sampling based on the predicted target.
- **stratify_log**: Does stratified sampling involve log transformation of the target?
- **feat_imp_measurer**:
- **feat_imp_threshold**:
- **feat_cluster_threshold**:
- **recur_addi_threshold**: the least metric ($R^2$) to add a feature into the essential feature set.
- **recur_addi_prev_threshold**: the threshold of feature importance, which is evaluated by a given
  measurer ML algorithm, like RF, where a feature with an importance larger than this threshold will
  be added into the initial essential feature collection.
- **essential_features**: Directly specify the necessary feature indices and skip all feature engineering steps.

Attributes:

- data: a DataFrame, store all data used in the work.
- sample_num: the total number of samples, in `MachineLearning.data`.
- features: list of feature names.
- target: the target name or list of target name.
- algorithm: the algorithm instance with method: fit(), predict(), and score().
- X: a DataFrame, the feature matrix.
- y: a Series or a DataFrame.
- other_info: other relevant data in the data, excluded in `X` or `y`.
- scaled_X: a numpy.Array, the feature matrix after scaling.
- scaled_y: a numpy.Array, the target vector or matrix after scaling.
- test_indices: the indices of sample in test set
- train_indices: the indices of sample in train set
- test_size: the ratio or count of test samples
- data_shuffle: Whether to shuffle the dataset in split the whole data to train and test set.
- data_stratify: Whether to perform stratified sampling based on the predicted target.
- stratify_log: If true, performing logarithmic preprocessing for the target before the stratification.
- explainer: applied the SHAP explainer.
- shap_value: the result of SHAP analysis
- scaled_X: a Numpy array, the X after scaling preprocessing
- scaled_y: a Numpy array, the y after scaling preprocessing
- sXtr: scaled X in train set
- sYtr: scaled y in train set
- sXte: scaled X in test set
- sYte: scaled y in test set
- ntri_feature: feature names (a list) after removing those trivial features, with ignorable feature importance
- ntri_X: X matrix after removing trivial features
- ntri_Xtr: X matrix after removing trivial features in train set
- ntri_Xte: X matrix after removing trivial features in test set
- ntri_sX: scaled X matrix after removing trivial features
- ntri_sXtr: scaled X matrix after removing trivial features in train set
- ntri_sXte: scaled X matrix after removing trivial features in test set
- clustered_feat_names: a dict, giving the mapping from clustered feature name to their own component feature names
- reduced_features: the name of features after removing the trivial and eliminating redundancy by merging similar ones
- reduced_sX: scaled reduced X (see `reduced_features`).
- reduced_sXtr: scaled reduced X (see `reduced_features`) in train set.
- reduced_sXte: scaled reduced X (see `reduced_features`) in test set.
- essential_features: the final features after feature engineering and to train the ML model, actually.
- essential_sX: scaled essential X (see\ `essential_features`).
- essential_sXtr: scaled essential X (see `essential_features`) in train set.
- essential_sXte: scaled essential X (see `essential_features`) in test set.

Methods:

> work()

Run the default machine learning workflow

> preprocess():

> train_test_split():

> feature_engineering():

> quickly_feature_selection():

> *staticmethod* calc_pearson_coef(X: Numpy.Array):

- **Args:**

  - X(Numpy.Array): shape of [sample_number, feature_number]
- **Return:**

  - Pearson_matrix: a matrix of Pearson coefficient for all cross-feature pairs

> calc_pearson_matrix():

calculate Pearson matrix for all non-trivial features

> make_pearson_matrix_plot():

draw Pearson matrix plot and save it to `work_dir/picture_dir`

> *staticmethod* make_hierarchical_tree(clustering: AgglomerativeClustering, threshold: float):

given a fitted sklearn.AgglomerativeClustering object and the clustering threshold, Draw the hierarchical
tree and return the Figure and Axes objects

> feat_hiera_clustering():

Constructing a Hierarchical tree for all non-trivial features, where leaf of the tree is all non-trivial
features and leaf will gradually merge with others from closer to further. The distances between features
are determined from Pearson matrix, where each of rows or columns is regarded as the coordinates of
corresponding features in Euclidean space.

The results are a dict stored in attribute `ntri_feat_cluster`, recording the maps from clustered features
to their original feature.

> pac_dimension_reduction():

Merging similar feature to a new clustering feature by principal component analysis (PCA) method, the
similarity or distance determined by Pearson matrix and hierarchical tree.

> recursive_feature_addition():

Carrying out Recursion Feature Addition (RFA) for the reduced_features, the non-trivial feature after
merging similar features to new ones by PCA. Recursively adding feature to the `essential feature`.

> recursive_feature_elimination()

Carrying out Recursion Feature Elimination (RFE) for the reduced_features, the non-trivial feature
after merging similar features to new ones by PCA. Recursively adding most feature to the `essential feature`.

> recursive_feature_selection():

Carrying out both Recursive Feature Addition (RFA) and Recursive Feature Elimination (RFE). the essential
features would be determined based on the model performance in cross validation (CV) and the feature numbers.

In general, the features collection gives a better performance in CV will be assigned as the essential ones.
However, if the performances are on par with each other, the collection with fewer features would be the
essential ones.

> *staticmethod* permutation_importance(estimator, *args, **kwargs):

Determining feature importance by value permutation

- **Args:**

  - estimator: an instance with method fit() and predict()
  - args and kwargs: arguments for `sklean.inspection.permutation_importance` function
- **Return:**

  - importance: a Numpy.Array, the importance with same order with the given X

> *staticmethod* gini_importance(estimator, *args, **kwargs):

Determine feature importance by gini method. Specially, train treelike model and the model-determined
feature importance.

- **Args:**

  - estimator: an instance with method fit() and predict()
  - X: feature matrix
  - y: the target
- **Return:**

  - importance: a Numpy.Array, the importance with same order with the given X

> determine_importance(feature_type: Literal['essential', 'reduced', 'scaled'] = 'essential',
> sample_type: Literal['train', 'test', 'all'] = 'train'):

Determine feature importance by all pre-defined methods. See the results by
attribute `MachineLearning.feature_importance`

- **Args:**
  - feature_type: select from 'essential', 'reduced', 'scaled', default 'essential'
  - sample_type: select from 'train', 'test', 'all', default 'train'

> cross_validation(feature_type: Literal['essential', 'reduced', 'scaled'] = 'essential',
> sample_type: Literal['train', 'test', 'all'] = 'train')

Performing cross validation under specified feature set and dataset. plot the resulted CV plot

- **Args:**

  - feature_type: select from 'essential', 'reduced', 'scaled', default 'essential'
  - sample_type: select from 'train', 'test', 'all', default 'train'
- **Return:**

  - importance: a Numpy.Array, the importance with same order with the given X

> train_model():

train the model by all train dataset, under essential feature and specified algorithm

> generate_test_X(
> self,
> template_X: np.ndarray = None,
> independent: bool = False,
> norm_uniform: Literal['norm', 'uniform'] = 'uniform',
> min_offset: Union[float, np.ndarray] = 0.,
> max_offset: Union[float, np.ndarray] = 0.,
> sample_num: int = 1000,
> seed_: int = None
> ):

Generates a hypothetical test X with similar covariance of features as template X.

- Args:

  - template_X: template X to define covariance (matrix). if not given, Essential_sX will be applied.
  - independent: whether to suppose each feature is independent with the other, defaults to False.
  - norm_uniform: generate from a uniform distribution or a normal distribution, defaults to 'uniform'
  - min_offset: the proportion between the offset lower than min value and the diff from min value to max value,
    the value should be a float or numpy array. if a float is provided, the same offset value will
    act on all feature; if a numpy array is provided, the length of array should be equal to the
    X.shape[1], in this case, different offset will assign to different features.
  - max_offset: the proportion between the offset higher than max value and the diff from min value to max value
    the value should be a float or numpy array. if a float is provided, the same offset value will
    act on all feature; if a numpy array is provided, the length of array should be equal to the
    X.shape[1], in this case, different offset will assign to different features.
  - sample_num: number of samples to be generated, i.e., X.shape[0] == sample_num.
  - seed_: random state for distribute generation
- Returns:
  generated test X with identical covariance of features as template X.

> calc_shap_values(
> self, estimator, X, y,
> feature_names: Union[Sequence, np.ndarray] = None,
> sample_size: int = 1000,
> X_test: np.ndarray = None,
> test_size: int = 1000,
> explainer_cls: shap.Explainer = shap.TreeExplainer,
> shap_values_save_path: Union[str, os.PathLike] = None,
> **kwargs
> ):

calculate shap values.

- Args:

  - estimator: estimator to be explained by SHAP analysis.
  - X: input data to train estimator
  - y: target data to train estimator
  - feature_names: feature names for each column of input data
  - sample_size: how many samples sample from input data to carry out SHAP analysis
  - X_test: data to be explained by SHAP explainer. if not given, test data is generated by
    workflow.generate_test_X method with a 'test_size' size
  - test_size: when the X_test is None, this arg is used to control the number of generated test samples
  - explainer_cls: class of SHAP explainer, defaults to shap.TreeExplainer.
  - shap_values_save_path: path to save the calculated SHAP results, should be an EXCEL file.
  - **kwargs: keyword arguments for workflow.generate_test_X method.
- Returns:

  - SHAP explainer, SHAP values.

> make_shap_bar_beeswarm(
> shap_values: shap.Explanation,
> max_display: int = 15,
> savefig_path: Union[str, os.PathLike] = None,
> **kwargs
> ):

Draw a bar and a beeswarm plots to overview the given shap values.

- Args:
  - shap_values: analyzed SHAP values
  - max_display: the max features to be shown
  - savefig_path: the path to save the resulted plot
  - **kwargs: keyword arguments for `hotpot.plot.SciPlotter` object

> shap_analysis():

Calculate SHAP value in given `estimator`, `essential features`, and `explainer`, where `estimator`
and `explainer` are specified in the initialization of `MachineLearning` instance.

<h3><font color=#888888>Discarded ↑↑↑↑↑↑↑↑ </font></h3>

## Overview

The `MachineLearning_` class is a Python interface designed to streamline machine learning workflows using scikit-learn and other methods. It simplifies tasks such as data preprocessing, model training, evaluation, prediction, and hyperparameter tuning.

<h3><font color=#b8860s><b>class</b></font> MachineLearning(work_dir: Pathlike,data: pandas.DataFrame, features: Union[Sequence[str], numpy.ndarray[str]],target: Union[str, list[str]], estimator: Optional=GradientBoostingRegressor, **kwargs)</h3>

- **work_dir(Path|str)**: the working directory to save result from the running procedure.
- **data(pd.DataFrame)**: the total dataset, including features, target, and other correlation information.
- **features(ArrayLike[str], Sequence[str])**: feature names in the given data DataFrame.
- **target(str, Sequence[str])**: target name(s) in data DataFrame.
- **estimator(BaseEstimator)**: algorithm instance with method: fit(), predict(), and score()

##### Keyword Arguments:

- **xscaler**(sklean.preprocessing.BaseScaler): Scaler for the input features (X).
- **yscaler**(sklean.preprocessing.BaseScaler): Scaler for the output target (y).
- **test_size**(float|int): the size of test set.
- **test_indices**(float|int): specify which sample should be assigned to the test set. If specified, the process
  of dataset split will skip. This argument is useful to reproduce the previous works.
- **data_shuffle**(bool): whether to shuffle the `data` when split the dataset.
- **data_stratify**(bool): Whether to perform stratified sampling based on the predicted target.
- **stratify_log**: Does stratified sampling involve log transformation of the target?
- **feat_imp_measurer**: An object with methods `fit()` and one of `coef_` or `feature_importance_`, which is applied to measure the feature importance in feature engineering.
- **feat_imp_cutoff**(float): A feature importance cutoff to drop out the feature with less importance from feature collection.
- **feat_cluster_threshold**(float): the threshold distance that features less than it would be put into a same cluster.
- **recur_addi_threshold**: the least metric ($R^2$ or other) to add a feature into the essential feature set.
- **recur_addi_prev_threshold**: the threshold of feature importance, which is evaluated by a given
  measurer ML algorithm, like RF, where a feature with an importance larger than this threshold will
  be added into the initial essential feature collection.
- **essential_features**: Directly specify the essential feature indices and skip all feature engineering steps.
- **skip_feature_reduce**(bool): Whether to skip feature reduction procedure.

##### Attributes:

- **work_dir**: Directory where all results are stored.
- **data**(pandas.DataFrame): Store all data used in the work.
- **target**: name(s) of target.
- **features**: names of feature.
- **estimator**: An instance of the model algorithm, which should implement methods such as `fit()`, `predict()`, and `score()`.
- **sample_num**: number of samples in the Dataset.
- **other_info**: Additional information within `data`, excluding features and target.
- **xscaler**: Scaler object for input features.
- **yscaler**: Scaler object for the output target.
- **test_indice**(numpy.ndarry, Sequence[int]): The indices of samples in the test set.
- **train_indice**(numpy.ndarry, Sequence[int]): The indices of samples in the train set.
- **test_size**(float): the proportion of test samples.
- **data_shuffle**: Whether to shuffle the dataset in split the whole data to train and test set.
- **data_stratify**: Whether to perform stratified sampling based on the predicted target.
- **stratify_log**: Whether to perform logarithmic preprocessing for the target before the stratification.
- **feat_imp_measurer**: The object with methods `fit()` and one of `coef_` or `feature_importance_`, which is applied to measure the feature importance in feature engineering.
- **feat_imp_cutoff**: A feature importance cutoff to drop out the feature with less importance from feature collection.
- **feat_cluster_threshold**: The threshold distance that features less than it would be put into a same cluster.
- **recur_addi_threshold**: The least metric ($R^2$ or other) to add a feature into the essential feature set.
- **recur_addi_pre_threshold**: The threshold of feature importance, which is evaluated by a given measurer ML algorithm, like RF, where a feature with an importance larger than this threshold will be added into the initial essential feature collection.
- **pearson_mat_dict**: A dict storing Pearson matrices for each stage.
- **clustered_maps**: A dict mapping each clustered feature to their original features before clustering.
- **clustering**: The return from `sklearn.AgglomerativeClustering`.
- **cv_best_metric**: The best cross-validation selection in feature selection operation.
- **how_to_essential**: How the essential features is selected, `Recursive elimination` or `Recursive Addition`.
- **feature_importance**: A dict storing feature importances determined by variable method, such as `perm` for permutate method, `gini` for gini method, `shap` for SHAP method.
- **valid_score**: The mean score in cross-validation.
- **valid_pred**: The predictive values in cross validation.
- **valid_true**: The true values in cross validation.
- **explainer**: The SHAP explainer.
- **shap_value**: SHAP values.
- **kwargs**: Received keyword arguments
- **X**: Current input matrix.
- **y**: the target vector or matrix.
- **X_train**: input matrix for train samples.
- **y_train**: y of train samples.
- **X_test**: X of test samples.
- **y_test**: y of test samples.
- **features**: Feature names in current working stage.
- **features_type**: Which type the current feature is, `original`, `scaled`, `non-trivial`, `reduced`, or `essential`.
- **pearson_mat**: The matrix of Pearson correlation coefficient between pairwise features.

##### Methods:

> work()

Run the default machine learning workflow

> preprocess():

Scaling X and y

> train_test_split():

> feature_engineering():

Performing feature engineering, including to drop out trivial features, clustering and merging similar features, recursive feature selection.

> quickly_feature_selection():

Drop out trivial features.

> *staticmethod* calc_pearson_matrix_(X: Numpy.Array):

- **Args:**

  - X(Numpy.Array): shape of [sample_number, feature_number].
- **Return:**

  - Pearson_matrix: a matrix of Pearson coefficient for all cross-feature pairs.

> calc_pearson_matrix():

Calculate Pearson matrix for all non-trivial features.

> make_pearson_matrix_plot():

Draw Pearson matrix plot and save it to `work_dir/picture_dir`.

> *staticmethod* make_hierarchical_tree(clustering: AgglomerativeClustering, threshold: float):

given a fitted sklearn.AgglomerativeClustering object and the clustering threshold, Draw the hierarchical
tree and return the Figure and Axes objects

> *staticmethod* make_hierarchical_tree(clustering, threshold: float)

Make a hierarchical tree plot return the `Figure` and `Axes` object in `matplotlib`.

- **Args**:
  - clustering: the return from `sklearn.AgglomerativeClustering`.
  - threshold: the distance threshold to perform features clustering.
- **Return**:
  - figure: `Figure` object in `matplotlib`.
  - axes: `Axes` object in `matplotlib`.

> *staticmethod* feat_hiera_clustering_(X, features, y=None, clustering_threshold=None):

- **Args**:
  - X(Numpy.Array): the input matrix, shape of [sample_number, feature_number].
  - features(Sequence[str]): sequence of feature names.
  - y: ignored.
  - clustering_threshold(float): the distance threshold values that features with a distance less than it will be put into a same cluster. The distance of each pair of features determined by their pairwise Pearson matrix. If not given, the threshold will be assign a default values according to the number of features, i.e., $T=\sqrt{(0.1)^2*N)}$, where the $T$ is the threshold values, $N$ is the number of features.
- **Return**:
  - clustering_threshold: the user-specified threshold or automatically determined threshold.
  - clustering_map: a dict mapping the clustered features (*key*) to their original features (*values*, a list of str).
  - clustering: the return from `sklearn.AgglomerativeClustering`.

> feat_hiera_clustering():

Constructing a Hierarchical tree for all non-trivial features, where leaf of the tree is all non-trivial
features and leaf will gradually merge with others from closer to further. The distances between features
are determined from Pearson matrix, where each of rows or columns is regarded as the coordinates of
corresponding features in Euclidean space.

The resulted *Hierarchical Tree* picture would be saved in `work_dir/picture_dir`. The `clustering_threshold`, `clustered_map`, and `clustering` object would assign to the `MachineLearning` instance as its attributes.

> clustered_map(which: str) -> dict:

Get a specific clustering map.

- **Args**:
  - which: which map to get.

> pac_dimension_reduction_(X:np.ndarray, features:list[str], clustering_map:dict=None, clustering_threshold:float=None):

Reduce groups of features to ones of reduced features, by Principal Component Analysis (PCA). The groups are given by the dict argument `clustering_map`.

- **Args**:
  - X: the orignal feature matrix before dimension reduction,
  - features: the original feature names, with a same order as the given `X`.
  - clustering_map(dict): the mapping between the reduced feature names (*key*) to the groups of feature names (*value*, list of original features).
- **Return**:
  - pcaX: the feature matrix after undergoing PCA dimensionality reduction does not include features that were not subjected to dimensionality reduction.
  - reduced_X: the feature matrix after undergoing PCA dimensionality reduction, including features that were not subjected to dimensionality reduction.
  - reduced_features: list of feature name, with a same order in `reduced_X`

> pac_dimension_reduction():

Merging similar feature to a new clustering feature by principal component analysis (PCA) method, the
similarity or distance determined by Pearson matrix and hierarchical tree.

> *staticmethod* recursive_feature_addition_(X_train:np.ndarray, y_train:np.ndarray, feature_measurer, prev_addi_threshold:float=0.2, recur_addi_threshold: float=0.):

Perform recursive feature addition.

Those important feature with feature importances bigger than `prev_addi_threshold` will be included into the final features set at first. For other less important features, the feature with best contribution to improve the model performance will be added into the final features set in a single cycle, until the contributions of all features for model imporvement are less than the specified `recur_addi_threshold`.

- **Args**:
  - X_train:
  - y_train:
  - feature_measurer:
  - prev_addi_threshold:
  - recur_addi_threshold:
- Return:
  - selected_feat_indices:
  - best_metric:

> recursive_feature_addition():

Carrying out Recursion Feature Addition (RFA) for the reduced_features, the non-trivial feature after
merging similar features to new ones by PCA. Recursively adding feature to the `essential feature`.

> *staticmethod* recursive_feature_elimination_(X_train: np.ndarray, y_train: np.ndarray, feat_measurer):

Select essential features for train final model by Recursion Feature Elimination (RFE).

- **Args**:
  - X_train:
  - y_train:
  - feat_measurer:
- **Return**:
  - feat_indices:
  - mean_cv_score:

> recursive_feature_elimination()

Carrying out Recursion Feature Elimination (RFE) for the reduced_features, the non-trivial feature
after merging similar features to new ones by PCA. Recursively adding most feature to the `essential feature`.

> recursive_feature_selection_(X, features, feat_indices_eli, score_eli, feat_indices_eli, score_add):

Compare the result between Recursion Feature Elimination (RFE) and Recursion Feature Addition (RFA) and select the better one from them.

- **Args**:
  - X:
  - features:
  - feat_indices_eli:
  - score_eli:
  - feat_indices_add:
  - score_add:
- **Return**:
  - X_after_feature_selection:
  - features_after_feature_selection:
  - cv_best_metric:
  - select_which: Which `Recursive addition` or `Recursive elimination` is selected.

> recursive_feature_selection():

Carrying out both Recursive Feature Addition (RFA) and Recursive Feature Elimination (RFE). the essential
features would be determined based on the model performance in cross validation (CV) and the feature numbers.

In general, the features collection gives a better performance in CV will be assigned as the essential ones.
However, if the performances are on par with each other, the collection with fewer features would be the
essential ones.

> *staticmethod* permutation_importance(estimator, *args, **kwargs):

Determining feature importance by value permutation

- **Args:**

  - estimator: an instance with method fit() and predict()
  - args and kwargs: arguments for `sklean.inspection.permutation_importance` function
- **Return:**

  - importance: a Numpy.Array, the importance with same order with the given X

> *staticmethod* gini_importance(estimator, *args, **kwargs):

Determine feature importance by gini method. Specially, train treelike model and the model-determined
feature importance.

- **Args:**

  - estimator: an instance with method fit() and predict()
  - X: feature matrix
  - y: the target
- **Return:**

  - importance: a Numpy.Array, the importance with same order with the given X

> determine_importance(feature_type: Literal['essential', 'reduced', 'scaled'] = 'essential',
> sample_type: Literal['train', 'test', 'all'] = 'train'):

Determine feature importance by all pre-defined methods. See the results by
attribute `MachineLearning.feature_importance`

- **Args:**
  - feature_type: select from 'essential', 'reduced', 'scaled', default 'essential'
  - sample_type: select from 'train', 'test', 'all', default 'train'

> cross_validation(feature_type: Literal['essential', 'reduced', 'scaled'] = 'essential',
> sample_type: Literal['train', 'test', 'all'] = 'train')

Performing cross validation under specified feature set and dataset. plot the resulted CV plot

- **Args:**

  - feature_type: select from 'essential', 'reduced', 'scaled', default 'essential'
  - sample_type: select from 'train', 'test', 'all', default 'train'
- **Return:**

  - importance: a Numpy.Array, the importance with same order with the given X

> train_model():

train the model by all train dataset, under essential feature and specified algorithm

> generate_test_X(
> self,
> template_X: np.ndarray = None,
> independent: bool = False,
> norm_uniform: Literal['norm', 'uniform'] = 'uniform',
> min_offset: Union[float, np.ndarray] = 0.,
> max_offset: Union[float, np.ndarray] = 0.,
> sample_num: int = 1000,
> seed_: int = None
> ):

Generates a hypothetical test X with similar covariance of features as template X.

- Args:

  - template_X: template X to define covariance (matrix). if not given, Essential_sX will be applied.
  - independent: whether to suppose each feature is independent with the other, defaults to False.
  - norm_uniform: generate from a uniform distribution or a normal distribution, defaults to 'uniform'
  - min_offset: the proportion between the offset lower than min value and the diff from min value to max value,
    the value should be a float or numpy array. if a float is provided, the same offset value will
    act on all feature; if a numpy array is provided, the length of array should be equal to the
    X.shape[1], in this case, different offset will assign to different features.
  - max_offset: the proportion between the offset higher than max value and the diff from min value to max value
    the value should be a float or numpy array. if a float is provided, the same offset value will
    act on all feature; if a numpy array is provided, the length of array should be equal to the
    X.shape[1], in this case, different offset will assign to different features.
  - sample_num: number of samples to be generated, i.e., X.shape[0] == sample_num.
  - seed_: random state for distribute generation
- Returns:
  generated test X with identical covariance of features as template X.

> calc_shap_values(
> self, estimator, X, y,
> feature_names: Union[Sequence, np.ndarray] = None,
> sample_size: int = 1000,
> X_test: np.ndarray = None,
> test_size: int = 1000,
> explainer_cls: shap.Explainer = shap.TreeExplainer,
> shap_values_save_path: Union[str, os.PathLike] = None,
> **kwargs
> ):

calculate shap values.

- Args:

  - estimator: estimator to be explained by SHAP analysis.
  - X: input data to train estimator
  - y: target data to train estimator
  - feature_names: feature names for each column of input data
  - sample_size: how many samples sample from input data to carry out SHAP analysis
  - X_test: data to be explained by SHAP explainer. if not given, test data is generated by
    workflow.generate_test_X method with a 'test_size' size
  - test_size: when the X_test is None, this arg is used to control the number of generated test samples
  - explainer_cls: class of SHAP explainer, defaults to shap.TreeExplainer.
  - shap_values_save_path: path to save the calculated SHAP results, should be an EXCEL file.
  - **kwargs: keyword arguments for workflow.generate_test_X method.
- Returns:

  - SHAP explainer, SHAP values.

> make_shap_bar_beeswarm(
> shap_values: shap.Explanation,
> max_display: int = 15,
> savefig_path: Union[str, os.PathLike] = None,
> **kwargs
> ):

Draw a bar and a beeswarm plots to overview the given shap values.

- Args:
  - shap_values: analyzed SHAP values
  - max_display: the max features to be shown
  - savefig_path: the path to save the resulted plot
  - **kwargs: keyword arguments for `hotpot.plot.SciPlotter` object

> shap_analysis():

Calculate SHAP value in given `estimator`, `essential features`, and `explainer`, where `estimator`
and `explainer` are specified in the initialization of `MachineLearning` instance.

## Examples

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from hotpot.plugins.ml.wf import MachineLearning

excel_path = 'path/to/excel/ChemData'
work_dir = '/the/dir/to/save/result'
data = pd.read_excel('')
feature_names = ['feature1', 'feature2', 'feature3', ...]
target_name = 'target_name'

hypers = {
    'n_estimators': 150,
    'max_depth': 15,
    '...': 'values'
}

model = RandomForestRegressor(**hypers)

ml_workflow = MachineLearning(
    work_dir=work_dir,
    data=data,
    features=feature_names,
    target=target_name,
    estimator=model
)
```

<h4><font color=#b8860s>**class**</font> MachineLearning(work_dir: Pathlike, 
data: pandas.DataFrame, features: Union[Sequence[str], numpy.ndarray[str]], 
target: Union[str, list[str]], estimator: Optional=GradientBoostingRegressor, 
**kwargs)</h4>
