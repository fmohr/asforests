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


import os
import psutil

import itertools as it
from tqdm import tqdm

from experiments.benchmark.result_storage import ResultStorage
from experiments.benchmark._util import get_unique_prediction_matrices
from experiments.benchmark.approaches.a_fromdatabase import DatabaseWiseApproach # used to compute ground truths as this is much more efficient than the naive way


class Benchmark:

    def __init__(self,
                 openmlid=None,
                 X=None,
                 y=None,
                 data_seed=0,
                 ensemble_seed=0,
                 ensemble_sequence_seed=0,
                 ensemble_prefix=None,
                 num_possible_ensemble_members=10,
                 training_instances_per_class=10,
                 validation_size=20,
                 is_classification=True,
                 captured_parameters=["E[Z_nt]", "E[Z_nt|D_val]", "V[Z_nt]", "V[Z_nt|D_val]"],
                 estimate_checkpoints=None,
                 precision=7,
                 upper_bound_for_sample_size_in_ground_truth_computation=10**8,
                 track_used_resources=False
                 ):
        
        # configuration variables
        self._openmlid = openmlid
        self._data_seed = data_seed
        self._ensemble_seed = ensemble_seed
        self._ensemble_sequence_seed = ensemble_sequence_seed
        self._ensemble_prefix = ensemble_prefix
        self._num_possible_ensemble_members = num_possible_ensemble_members
        self._training_instances_per_class = training_instances_per_class
        self._validation_size = validation_size
        self._is_classification = is_classification
        self.logger = logging.getLogger("benchmark")
        self.captured_parameters = captured_parameters
        self.upper_bound_for_sample_size_in_ground_truth_computation = upper_bound_for_sample_size_in_ground_truth_computation
        self._estimate_checkpoints = estimate_checkpoints
        self._precision = precision
        self.track_used_resources = track_used_resources

        # state variables
        if X is None:
            self._X = self._y = None
        else:
            self._X = X
            self._y = y
        self._indices_train = self._indices_val = self._indices_oos = None
        self._deviations = None
        self._true_parameters = None
        self._prediction_matrix_generator = None
        self._approaches = None
        self._t_checkpoints = None
        self._t = None
        self._result_storage = None
        self.process = psutil.Process(os.getpid()) # get process for memory surveillance

    @property
    def openmlid(self):
        return self._openmlid

    @property
    def data_seed(self):
        return self._data_seed

    @property
    def training_instances_per_class(self):
        return self._training_instances_per_class

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
    def deviation_covs(self):
        return self._covs

    @property
    def deviation_covs_val(self):
        return self._covs_val

    @property
    def ensemble_member_id_generator(self):
        return self._prediction_matrix_generator
    
    @property
    def t(self):
        return self._t

    @property
    def result_storage(self):
        return self._result_storage
    
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

        if self._X is None:
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
        label_count = {}
        if self._y.dtype != int:
            y_int = np.zeros(len(self._y)).astype(int)
            vals = np.unique(self._y)
            for i, val in enumerate(vals):
                mask = self._y == val
                label_count[val] = np.count_nonzero(mask)
                y_int[mask] = i
            self._y = y_int
        else:
            vals = np.unique(self._y)
            for i, val in enumerate(vals):
                label_count[val] = np.count_nonzero(self._y == val)

        # partition the given data into train, validation, and out-of-sample data
        self.logger.info(f"Label count: {label_count}")
        minority_class = min(list(label_count.keys()), key=label_count.get)

        training_size_relative = self.training_instances_per_class / label_count[minority_class]
        self.logger.info(f"Using {np.round(training_size_relative * 100, 2)}% of the  data for training")

        rs_data = np.random.RandomState(self._data_seed)
        splitter_val = StratifiedShuffleSplit(n_splits=1, random_state=rs_data, train_size=self.validation_size) if self.is_classification else ShuffleSplit(n_splits=1, random_state=rs_data, train_size=self.validation_size)
        validation_indices, rest_indices = next(splitter_val.split(self.X, self.y))
        splitter_rest = StratifiedShuffleSplit(n_splits=1, random_state=rs_data, train_size=training_size_relative) if self.is_classification else ShuffleSplit(n_splits=1, random_state=rs_data, train_size=training_size_relative)
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
            self.logger.info(f"Modifying inputs with {pl}")
            pl.fit(self.X_train, self.y_train)
            self._X = pl.transform(self._X)

    def _compute_predictions_and_deviations(self):
        if self._deviations is not None:
            self.logger.info(f"Warning: deviations have already been computed, skipping.")
            return
        
        # send log message
        self.logger.info(f"Computing predictions and deviations of all possible ensemble members.")
        t_start = time()

        # compute 3D tensor with all deviations of all ensemble members on all data points
        prediction_matrices, classes = get_unique_prediction_matrices(
            X=self.X,
            y=self.y,
            train_indices=self._indices_train,
            seed=self._ensemble_seed,
            num_matrices=self._num_possible_ensemble_members
        )

        # memorize prediction matrices
        indices = [classes.index(i) for i in self.y]
        self.y_oh = np.eye(len(classes))[indices]
        self._predictions = np.array(prediction_matrices[:self._num_possible_ensemble_members])
        assert self._predictions.shape == (self._num_possible_ensemble_members, self.X.shape[0], self.y_oh.shape[1])
        if np.any(np.isnan(self._predictions)):
            raise ValueError(f"predictions have nan entries: {self._predictions}")
        self._deviations = self._predictions - self.y_oh
        if np.any(np.isnan(self._deviations)):
            raise ValueError(f"deviations have nan entries: {self._deviations}")

        # check that predictions of ensembles are pairwise different
        for i, p1 in enumerate(self._predictions):
            for j, p2 in enumerate(self._predictions[:i]):
                assert not np.all(np.isclose(p1, p2)), f"Predictions of ensemble member {i} and {j} are identical."
        self.logger.info(f"Prediction and deviation computation finished after {int(1000 * (time() - t_start))}ms.")    
    
    def reset(self, approaches: dict, t_checkpoints: list, ensemble_sequence_seed: int = None):

        # initialize deviations if this has not happned yet
        if self._deviations is None:
            self._load_data()
            self._compute_predictions_and_deviations()
        
        # create/reset prediction matrix generator
        if ensemble_sequence_seed is not None:
            self._ensemble_sequence_seed = ensemble_sequence_seed
        deviation_generator_rs = np.random.RandomState(self._ensemble_sequence_seed)
        def f():
            while True:
                yield deviation_generator_rs.choice(range(len(self._deviations)))

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

        # we use the database-based approach to estimate the ground truth (only possible if we show exactly once the predictions of all ensemble members, cf unit tests)
        self.logger.info(f"Starting computation of ground truth. Maximum sample size at each stage is {self.upper_bound_for_sample_size_in_ground_truth_computation}")
        ground_truth_computer_iid = DatabaseWiseApproach(
            population_mode="stream",
            estimated_parameters=["E[Z_nt]", "V[Z_nt]"],
            upper_bound_for_sample_size=self.upper_bound_for_sample_size_in_ground_truth_computation,
            logger=self.logger
        )
        ground_truth_computer_cond = DatabaseWiseApproach(
            population_mode="stream",
            estimated_parameters=["E[Z_nt|D_val]", "V[Z_nt|D_val]"],
            upper_bound_for_sample_size=self.upper_bound_for_sample_size_in_ground_truth_computation,
            logger=self.logger
        )
        ground_truth_computer_iid.reset()
        ground_truth_computer_iid.tell_ground_truth_labels(y_oh=self.y_oh)
        ground_truth_computer_cond.reset()
        ground_truth_computer_cond.tell_ground_truth_labels(y_oh=self.y_oh[self._indices_val])
        for s, pm in enumerate(tqdm(self._predictions)):
            self.logger.debug(f"Feeding {s+1}-th prediction matrix to estimator to update the estimates.")
            ground_truth_computer_iid.receive_predictions_of_new_ensemble_member(pm)
            ground_truth_computer_cond.receive_predictions_of_new_ensemble_member(pm[self._indices_val])
        self._true_parameters = {
            "E[Z_nt]": ground_truth_computer_iid.estimate_performance_mean_in_iid_setup(t=self._t_checkpoints),
            "E[Z_nt|D_val]": ground_truth_computer_cond.estimate_performance_mean_in_conditional_setup(t=self._t_checkpoints),
            "V[Z_nt]": ground_truth_computer_iid.estimate_performance_var_for_two_instances_in_iid_setup(t=self._t_checkpoints),
            "V[Z_nt|D_val]": ground_truth_computer_cond.estimate_performance_var_in_conditional_setup(t=self._t_checkpoints)
        }
        self.logger.info(f"Ground truth parameter values are: {self._true_parameters}")

        # reset storage
        self._t = 0
        self._history_of_member_ids = []
        self._result_storage = ResultStorage(
            true_param_values=self._true_parameters,
            approach_names=list(approaches.keys()),
            t_checkpoints=t_checkpoints
            )
    
    def step(self):

        if self._approaches is None:
            raise ValueError("No approaches registered. Use `reset` to define the approaches.")

        self.logger.info(f"Starting round {self._t}.")

        # update knowledge of all approaches
        member_id = next(self.ensemble_member_id_generator) if (self._ensemble_prefix is None or self._t >= len(self._ensemble_prefix)) else self._ensemble_prefix[self._t]
        self._history_of_member_ids.append(member_id)
        matrix = self._predictions[member_id, self._indices_val]
        self._t += 1
        if self.track_used_resources:
            self.logger.debug(
                f"Current memory consumption is {self.process.memory_info().rss / (1024 ** 2):.2f}MB. "
                f"Current CPU usage is {self.process.cpu_percent(interval=1.0)}."
            )
        
        if np.any(np.isnan(matrix)):
            raise ValueError(f"Prediction matrix in round {self._t} has nan entries.")

        do_update_estimates = self._estimate_checkpoints is None or self._t in self._estimate_checkpoints

        for approach_name, approach_obj in self._approaches.items():
            self.logger.debug(f"Stepping {approach_name}.")
            keys_and_methods_available = {
                "add": (lambda: approach_obj.receive_predictions_of_new_ensemble_member(matrix), False),
                #"update_iid": (approach_obj._update_estimates_for_iid, False),
                #"update_cond": (approach_obj._update_estimates_for_conditional, False),
                "E[Z_nt|D_val]": (approach_obj.estimate_performance_mean_in_conditional_setup, True),
                "V[Z_nt|D_val]": (approach_obj.estimate_performance_var_in_conditional_setup, True),
                "E[Z_nt]": (approach_obj.estimate_performance_mean_in_iid_setup, True),
                "V[Z_nt]": (approach_obj.estimate_performance_var_for_two_instances_in_iid_setup, True)
            }

            enabled_keys = ["add"] + [p for p in ["E[Z_nt|D_val]", "E[Z_nt]", "V[Z_nt|D_val]", "V[Z_nt]"] if p in approach_obj.estimated_parameters and p in self.captured_parameters]
            keys_and_methods_applied = {k: keys_and_methods_available[k] for k in enabled_keys}

            estimates = {
                int(t): {} for t in self._t_checkpoints
            }
            
            runtimes = {}
            if do_update_estimates:
                self.logger.info(f"Requesting estimates for {list(keys_and_methods_applied.keys())} from {approach_name}")
            for p, (m, has_estimate) in keys_and_methods_applied.items():
                t0 = time()
                if has_estimate and do_update_estimates:
                    self.logger.debug(f"Requesting estimates for {p} from {approach_name}")
                    e = m(self._t_checkpoints)
                    self.logger.debug(f"{approach_name} estimates {e} for {p}")
                elif not has_estimate:
                    e = m()
                t1 = time()
                if has_estimate and do_update_estimates:
                    assert isinstance(e, np.ndarray), f"Returned estimates must be a numpy array, but {approach_name} returned {type(e)} for {p}"
                    for t, v in zip(self._t_checkpoints, e):
                        estimates[int(t)][p] = float(np.round(v, self._precision))
                runtimes[p] = t1 - t0
            
            if do_update_estimates:
                self.logger.info(f"Storing estimates {estimates} for approach {approach_name} with runtimes {runtimes}")
                self._result_storage.add_estimates(approach_name, self.t, estimates, runtimes)
            self.logger.debug(f"Stepped {approach_name}. Runtimes: {runtimes}. Estimates are {estimates}")
        self.logger.info(f"Step finished {self._t} finished.")
