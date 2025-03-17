import numpy as np
import pandas as pd
import openml
from time import time

import logging

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from asforests.cb_computer import EnsemblePerformanceAssessor

from experiments.benchmark.result_storage import ResultStorage


class Benchmark:

    def __init__(self,
                 openmlid,
                 data_seed,
                 ensemble_seed,
                 ensemble_sequence_seed,
                 num_possible_ensemble_members,
                 training_size,
                 validation_size,
                 is_classification
                 ):
        
        # configuration variables
        self._openmlid = openmlid
        self._data_seed = data_seed
        self._ensemble_seed = ensemble_seed
        self._ensemble_sequence_seed = ensemble_sequence_seed
        self._num_possible_ensemble_members = num_possible_ensemble_members
        self._training_size = training_size
        self._validation_size = validation_size
        self._is_classification = is_classification
        self.logger = logging.getLogger("benchmark")

        # state variables
        self._X = self._y = None
        self._indices_train = self._indices_val = self._indices_oos = None
        self._means = self._vars = self._covs = self._covs_val = None
        self._deviations = None
        self._true_parameters = None
        self._prediction_matrix_generator = None
        self._approaches = None
        self._t_checkpoints = None
        self._t = None
        self._result_storage = None

    @property
    def openmlid(self):
        return self._openmlid

    @property
    def data_seed(self):
        return self._data_seed

    @property
    def training_size(self):
        return self._training_size

    @property
    def validation_size(self):
        return self._validation_size

    @property
    def is_classification(self):
        return self._is_classification

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def X_train(self):
        return self._X[self._indices_train]

    @property
    def X_val(self):
        return self._X[self._indices_val]

    @property
    def X_oos(self):
        return self._X[self._indices_oos]

    @property
    def y_train(self):
        return self._y[self._indices_train]

    @property
    def y_val(self):
        return self._y[self._indices_val]

    @property
    def y_oos(self):
        return self._y[self._indices_oos]

    @property
    def deviation_means(self):
        return self._means

    @property
    def deviation_means_val(self):
        return self._deviations[:, self._indices_val].mean(axis=0)

    @property
    def deviation_vars(self):
        return self._vars

    @property
    def deviation_vars_val(self):
        """
        :return: the array V[D^1_1 | validation data] with one entry for each label
        """
        return self._deviations[:, self._indices_val].var(axis=0)

    @property
    def deviation_covs_val(self):
        return self._covs_val

    @property
    def prediction_matrix_generator(self):
        return self._prediction_matrix_generator
    
    @property
    def t(self):
        return self._t

    @property
    def result_storage(self):
        return self._result_storage

    def get_true_performance_mean_on_iid_data(self):
        t = self._t_checkpoints
        return np.sum(self.deviation_means_val ** 2) + np.sum(self.deviation_vars_val) / t + (1 - 1/t) * np.sum(self.deviation_covs_val)
    
    def get_true_performance_mean_on_conditioned_data(self, instance_indices):
        t = self._t_checkpoints
        term1 = (self._deviations[:, instance_indices].mean(axis=0)**2).mean(axis=0).sum(axis=0)
        term2 = (self._deviations[:, instance_indices].var(axis=0).mean(axis=0).sum(axis=0)) / t  # in the conditional variance, the deviations are independent
        return term1 + term2

    def get_true_performance_var_on_iid_data(self):

        # TODO: Implement this
        return np.ones(len(self._t_checkpoints)) * (1 + self.data_seed) * (1 + self._ensemble_seed)
    
    def get_true_performance_var_on_conditioned_data(self, instance_indices):

        # TODO: Implement this
        return np.ones(len(self._t_checkpoints)) * (1 + self.data_seed) * (1 + self._ensemble_seed)
    
    def _get_mandatory_preprocessing(self, X, y):
        
        # determine fixed pre-processing steps for imputation and binarization
        types = [set([type(v) for v in r]) for r in X.T]
        numeric_features = [c for c, t in enumerate(types) if len(t) == 1 and list(t)[0] != str]
        numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical_features = [i for i in range(X.shape[1]) if i not in numeric_features]
        missing_values_per_feature = np.sum(pd.isnull(X), axis=0)
        self.logger.info(f"There are {len(categorical_features)} categorical features, which will be turned into integers.")
        self.logger.info(f"Missing values for the different attributes are {missing_values_per_feature}.")
        if len(categorical_features) > 0 or sum(missing_values_per_feature) > 0:
            categorical_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("binarizer", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
            ])
            return [("impute_and_binarize", ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
                ))]
        else:
            return []

    def _load_data(self):
        """
        Creates the three-fold split into training, validation, and out-of-sample data
        :return:
        """

        ds = openml.datasets.get_dataset(
            self.openmlid,
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False
        )
        df = ds.get_data()[0]

        # prepare data with label encoding for categorical attributes
        self._X = np.array(df.drop(columns=[ds.default_target_attribute]).values)
        self._y = np.array(df[ds.default_target_attribute].values)
        if self._y.dtype != int:
            y_int = np.zeros(len(self._y)).astype(int)
            vals = np.unique(self._y)
            for i, val in enumerate(vals):
                mask = self._y == val
                y_int[mask] = i
            self._y = y_int

        # partition the given data into train, validation, and out-of-sample data
        rs_data = np.random.RandomState(self._data_seed)
        splitter_val = StratifiedShuffleSplit(n_splits=1, random_state=rs_data, train_size=self.validation_size) if self.is_classification else ShuffleSplit(n_splits=1, random_state=rs_data, train_size=self.validation_size)
        validation_indices, rest_indices = next(splitter_val.split(self.X, self.y))
        splitter_rest = StratifiedShuffleSplit(n_splits=1, random_state=rs_data, train_size=self.training_size) if self.is_classification else ShuffleSplit(n_splits=1, random_state=rs_data, train_size=self.training_size)
        train_indices, oos_indices = next(splitter_rest.split(self.X[rest_indices], self.y[rest_indices]))
        train_indices = rest_indices[train_indices]
        oos_indices = rest_indices[oos_indices]
        train_indices.sort(), oos_indices.sort(), validation_indices.sort()
        assert len(set(train_indices) | set(validation_indices) | set(oos_indices)) == len(self.X)
        self._indices_train = train_indices
        self._indices_val = validation_indices
        self._indices_oos = oos_indices

        # check whether we need to overwrite the data
        preprocessing = self._get_mandatory_preprocessing(self._X, self._y)
        if preprocessing:
            pl = Pipeline(preprocessing)
            print(f"Modifying inputs with {pl}")
            pl.fit(self.X_train, self.y_train)
            self._X = pl.transform(self._X)

    def _compute_ground_truth_parameters(self):

        if self._means is not None:
            print(f"Warning: ground truth has already been computed, skipping.")
            return
        
        # send log message
        print(f"Computing ground truth parameter values.")
        t_start = time()

        # compute 3D tensor with all deviations of all ensemble members on all data points
        rf = RandomForestClassifier(
            n_estimators=self._num_possible_ensemble_members,
            random_state=self._ensemble_seed
            ).fit(self.X_train, self.y_train)
        ensemble_members = list(rf)
        classes_ = list(rf.classes_)
        indices = [classes_.index(i) for i in self.y]
        self.y_oh = np.eye(len(classes_))[indices]
        self._predictions = np.array([t.predict_proba(self.X) for t in ensemble_members])
        self._deviations = self._predictions - self.y_oh

        # compute ground truth from this tensor
        epa = EnsemblePerformanceAssessor(upper_bound_for_sample_size=10**10, population_mode="stream")
        self._means = self._deviations.mean(axis=(0, 1))
        self._vars = self._deviations.var(axis=(0, 1))
        for d in self._deviations:
            epa.add_deviation_matrix(d[self._indices_val])
        self._covs_val = epa.gap_cov_across_members_point

        #assert np.all(np.isclose(self._means, epa.gap_mean_point))
        #assert np.all(np.isclose(self._vars, epa.gap_var_point))
        print(f"Ground truth computation finished after {int(1000 * (time() - t_start))}ms.")
    
    
    def reset(self, approaches: dict, t_checkpoints: list, ensemble_sequence_seed: int = None):

        # initialize deviations if this has not happned yet
        if self._deviations is None:
            self._load_data()
            self._compute_ground_truth_parameters()
        
        # create/reset prediction matrix generator
        if ensemble_sequence_seed is not None:
            self._ensemble_sequence_seed = ensemble_sequence_seed
        deviation_generator_rs = np.random.RandomState(self._ensemble_sequence_seed)
        def f():
            while True:
                yield self._predictions[deviation_generator_rs.choice(range(len(self._deviations))), self._indices_val]

        self._prediction_matrix_generator = f()

        # register approaches
        self._approaches = approaches
        for approach in self._approaches.values():
            approach.reset()
            approach.tell_ground_truth_labels(self.y_oh[self._indices_val])
        
        # register check points and compute true values for those checkpoints
        if isinstance(t_checkpoints, int):
            t_checkpoints = np.array([t_checkpoints])
        if isinstance(t_checkpoints, list):
            t_checkpoints = np.array(t_checkpoints)
        if not isinstance(t_checkpoints, np.ndarray) or not t_checkpoints.dtype == int:
            raise ValueError(f"t_checkpoints must be an integer, a list of integers, or a np array of type int but is {type(t_checkpoints)}")
        self._t_checkpoints = t_checkpoints
        self._true_parameters = {
            "E[Z_nt|D_val]": self.get_true_performance_mean_on_conditioned_data(instance_indices=self._indices_val),
            "V[Z_nt|D_val]": self.get_true_performance_var_on_conditioned_data(instance_indices=self._indices_val)
        }

        # reset storage
        self._t = 0
        self._result_storage = ResultStorage(
            true_param_values=self._true_parameters,
            approach_names=list(approaches.keys()),
            t_checkpoints=t_checkpoints
            )

    
    def step(self):

        if self._approaches is None:
            raise ValueError("No approaches registered. Use `reset` to define the approaches.")

        # update knowledge of all approaches
        matrix = next(self.prediction_matrix_generator)
        self._t += 1
        for approach_name, approach_obj in self._approaches.items():
            t_0 = time()
            approach_obj.receive_predictions_of_new_ensemble_member(matrix)
            t_1 = time()

            keys_and_methods = {
                "E[Z_nt|D_val]": approach_obj.estimate_performance_mean,
                "V[Z_nt|D_val]": approach_obj.estimate_performance_var,
            }

            estimates = {
                int(t): {} for t in self._t_checkpoints
            }
            runtimes = {"add": int(1000 * (t_1 - t_0))}
            for p, m in keys_and_methods.items():
                t0 = time()
                e = m(self._t_checkpoints)
                t1 = time()
                for t, v in zip(self._t_checkpoints, e):
                    estimates[int(t)][p] = float(v)
                runtimes[p] = t1 - t0
            self._result_storage.add_estimates(approach_name, self.t, estimates, runtimes)
