# Machine Learning workflow

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
    self,
    template_X: np.ndarray = None,
    independent: bool = False,
    norm_uniform: Literal['norm', 'uniform'] = 'uniform',
    min_offset: Union[float, np.ndarray] = 0.,
    max_offset: Union[float, np.ndarray] = 0.,
    sample_num: int = 1000,
    seed_: int = None
):

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
    self, estimator, X, y,
    feature_names: Union[Sequence, np.ndarray] = None,
    sample_size: int = 1000,
    X_test: np.ndarray = None,
    test_size: int = 1000,
    explainer_cls: shap.Explainer = shap.TreeExplainer,
    shap_values_save_path: Union[str, os.PathLike] = None,
    **kwargs
):

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
    shap_values: shap.Explanation,
    max_display: int = 15,
    savefig_path: Union[str, os.PathLike] = None,
    **kwargs
):

  Draw a bar and a beeswarm plots to overview the given shap values.
  - Args:
    - shap_values: analyzed SHAP values
    - max_display: the max features to be shown
    - savefig_path: the path to save the resulted plot
    - **kwargs: keyword arguments for `hotpot.plot.SciPlotter` object
  
> shap_analysis():

  Calculate SHAP value in given `estimator`, `essential features`, and `explainer`, where `estimator`
  and `explainer` are specified in the initialization of `MachineLearning` instance.