"""
Functions for explaining classifiers that use tabular data (matrices).
"""
import collections
import copy
from functools import partial
import warnings

import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from pyDOE2 import lhs
from scipy.stats.distributions import norm

from . import explanation
from .lime_base import LimeBase


class TableDomainMapper(explanation.DomainMapper):
    """Maps feature ids to names, generates table views, etc"""

    def __init__(self, feature_names, feature_values, scaled_row,
                 categorical_features, feature_indexes=None):
        """Init.

        Args:
            feature_names: list of feature names, in order
            feature_values: list of strings with the values of the original row
            scaled_row: scaled row
            categorical_features: list of categorical features ids (ints)
            feature_indexes: optional feature indexes used in the sparse case
        """
        self.exp_feature_names = feature_names
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.feature_indexes = feature_indexes
        self.scaled_row = scaled_row
        if sp.sparse.issparse(scaled_row):
            self.all_categorical = False
        else:
            self.all_categorical = len(categorical_features) == len(scaled_row)
        self.categorical_features = categorical_features

    def map_exp_ids(self, exp):
        """Maps ids to feature names.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]

        Returns:
            list of tuples (feature_name, weight)
        """
        names = self.exp_feature_names
        return [(names[x[0]], x[1]) for x in exp]


class LimeTabularExplainer(object):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self,
                 training_data,
                 mode="classification",
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 sample_around_instance=False,
                 random_state=None):
        """Init function.

        Args:
            training_data: numpy 2d array
            mode: "classification" or "regression"
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
                If None, defaults to sqrt (number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            sample_around_instance: if True, will sample continuous features
                in perturbed samples from a normal centered at the instance
                being explained. Otherwise, the normal is centered on the mean
                of the feature data.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.random_state = check_random_state(random_state)
        self.mode = mode
        self.categorical_names = categorical_names or {}
        self.sample_around_instance = sample_around_instance

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * .75
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.feature_selection = feature_selection
        self.base = LimeBase(kernel_fn, verbose, random_state=self.random_state)
        self.class_names = class_names

        # Though set has no role to play if training data stats are provided
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            column = training_data[:, feature]

            feature_count = collections.Counter(column)
            values, frequencies = map(list, zip(*(sorted(feature_count.items()))))

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         sampling_method='gaussian'):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            sampling_method: Method to sample synthetic data. Defaults to Gaussian
                sampling. Can also use Latin Hypercube Sampling.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        # Step 1: Handle sparse data format
        # If data_row is sparse but not in csr format, convert it to csr format for easier processing.
        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            data_row = data_row.tocsr()

        # Step 2: Generate neighborhood data by perturbing features
        # Generates perturbed data points around the instance being explained and their corresponding original values.
        data, inverse = self._generate_neighborhood_data_inverse(data_row, num_samples, sampling_method)

        # Step 3: Scale the perturbed data
        # If the data is sparse, multiply by the scaling factor, otherwise subtract the mean and divide by the standard deviation.
        if sp.sparse.issparse(data):
            scaled_data = data.multiply(self.scaler.scale_)
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
            scaled_data = (data - self.scaler.mean_) / self.scaler.scale_

        # Step 4: Compute pairwise distances between the perturbed data and the original instance
        # The distance metric (e.g., Euclidean) is used to calculate the similarity between each perturbed sample and the original instance.
        distances = sklearn.metrics.pairwise_distances(
            scaled_data,
            scaled_data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        # Step 5: Predict with the original model
        # Use the original model to predict on the inverse (perturbed) data.
        yss = predict_fn(inverse)

        # Step 6: Handle classification case
        if self.mode == "classification":
            # If the model does not output probabilities, raise an error.
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores.")
            elif len(yss.shape) == 2:
                # If class names are not provided, assign default names '0', '1', '2', ...
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                # If the prediction probabilities do not sum to 1, raise a warning.
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilities do not sum to 1, and
                    thus do not constitute a valid probability space.
                    Check that your classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs arrays with {} dimensions".format(len(yss.shape)))

        # Step 7: Handle regression case
        # In the regression case, check that the predictions are in the expected format (1D array).
        else:
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            # Set predicted value, min, and max values
            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # Add a dimension to be compatible with downstream processes
            yss = yss[:, np.newaxis]

        # Step 8: Prepare feature names and values
        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        # If the data is sparse, retrieve the non-zero values for explanation
        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        # Update feature names for categorical features
        for i in self.categorical_features:
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        # Step 9: Create domain mapper
        # The domain mapper converts feature indices to feature names for explanation.
        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          feature_indexes=feature_indexes)

        # Step 10: Initialize explanation object
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)

        # Step 11: Set probabilities for classification tasks
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            # For regression, set predicted, min, and max values
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        # Step 12: Fit surrogate model and generate explanations for each label
        # Fit a local linear model (surrogate) to the neighborhood data and return the explanations.
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                scaled_data,
                yss,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

        # Step 13: Handle the regression case if needed
        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        # Return the explanation object
        return ret_exp

    def _generate_neighborhood_data_inverse(self,
                                             data_row,
                                             num_samples,
                                             sampling_method):
        """Generates a neighborhood around a prediction with numerical and categorical features perturbed."""

        # Check if the data is sparse and separate handling
        is_sparse = sp.sparse.issparse(data_row)
        num_cols = data_row.shape[1] if is_sparse else data_row.shape[0]
        instance_sample = data_row

        # Initialize data matrix
        data = self._initialize_data_matrix(is_sparse, num_samples, num_cols, data_row)

        # Set scaling factors
        scale, mean = self._get_scale_and_mean(data_row, num_cols, is_sparse)

        # Apply perturbation strategy (Gaussian, LHS, etc.)
        data = self._apply_perturbation(sampling_method, num_cols, num_samples, data, scale, mean,
                                                 instance_sample)

        # Handle sparse data after perturbation
        if is_sparse:
            data = self._handle_sparse_data(data_row, num_cols, data, num_samples)

        # Handle categorical features
        data, inverse = self._process_categorical_features(data, data_row, num_samples)

        return data, inverse

    def _initialize_data_matrix(self, is_sparse, num_samples, num_cols, data_row):
        """Initialize the data matrix, either sparse or dense."""
        if is_sparse:
            return sp.sparse.csr_matrix((num_samples, num_cols), dtype=data_row.dtype)
        else:
            return np.zeros((num_samples, num_cols))

    def _get_scale_and_mean(self, data_row, num_cols, is_sparse):
        """Retrieve scaling and mean values, adjusted for sparse data if needed."""
        scale = self.scaler.scale_
        mean = self.scaler.mean_
        if is_sparse:
            non_zero_indexes = data_row.nonzero()[1]
            scale = scale[non_zero_indexes]
            mean = mean[non_zero_indexes]
        return scale, mean

    def _apply_perturbation(self, sampling_method, num_cols, num_samples, data, scale, mean, instance_sample):
        """Applies the chosen perturbation strategy (Gaussian, LHS, etc.)."""
        if sampling_method == 'gaussian':
            data = self.random_state.normal(0, 1, num_samples * num_cols).reshape(num_samples, num_cols)
        elif sampling_method == 'lhs':
            data = lhs(num_cols, samples=num_samples).reshape(num_samples, num_cols)
            # Apply standard normal distribution to LHS
            for i in range(num_cols):
                data[:, i] = norm(loc=0, scale=1).ppf(data[:, i])
        else:
            warnings.warn('Invalid sampling method. Defaulting to Gaussian sampling.', UserWarning)
            data = self.random_state.normal(0, 1, num_samples * num_cols).reshape(num_samples, num_cols)

        # Apply scaling and centering around the instance or mean
        if self.sample_around_instance:
            return data * scale + instance_sample
        return data * scale + mean

    def _handle_sparse_data(self, data_row, num_cols, data, num_samples):
        """Handles sparse data post-perturbation."""
        non_zero_indexes = data_row.nonzero()[1]
        if num_cols == 0:
            return sp.sparse.csr_matrix((num_samples, data_row.shape[1]), dtype=data_row.dtype)
        else:
            indexes = np.tile(non_zero_indexes, num_samples)
            indptr = np.arange(0, len(non_zero_indexes) * (num_samples + 1), len(non_zero_indexes))
            data_1d = data.reshape(-1)
            return sp.sparse.csr_matrix((data_1d, indexes, indptr), shape=(num_samples, data_row.shape[1]))

    def _process_categorical_features(self, data, data_row, num_samples):
        """Processes categorical features by applying perturbations and returns the binary (data) and original (inverse) representations."""
        inverse = data.copy()
        for column in self.categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples, replace=True, p=freqs)
            binary_column = (inverse_column == data_row[column]).astype(int)
            binary_column[0] = 1  # Ensure first row (original instance) is the same
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column  # Binary encoded in data
            inverse[:, column] = inverse_column  # Original categorical values in inverse

        data[0] = data_row.copy()  # Ensure first row is the original data row
        inverse[0] = data_row  # Ensure first row is the original data row

        return data, inverse
