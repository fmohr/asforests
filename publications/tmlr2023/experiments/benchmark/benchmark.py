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

import itertools as it
from tqdm import tqdm

from experiments.benchmark.result_storage import ResultStorage
from experiments.benchmark._ground_truth_computer import GroundTruthComputer


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
                 max_ground_truth_table_size=10**6
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
        self.max_ground_truth_table_size = max_ground_truth_table_size
        self._estimate_checkpoints = estimate_checkpoints
        self._precision = precision

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
    
    def get_coefficients_for_covariances_for_variance(self, t):
        return np.array([
            1,
            (t-1) * 2,
            (t-1) * 4,
            (t-1),
            (t-1)*(t-2)*2,
            (t-1)*(t-2)*4,
            (t-1)*(t-2)*(t-3)
        ])

    def get_true_performance_mean_on_iid_data(self, t=None):
        if t is None:
            t = self._t_checkpoints

        # compute all ingredients on RHS
        deviation_means = self._deviations.mean(axis=(0, 1))
        deviation_vars = self._deviations.var(axis=(0, 1), ddof=0)
        deviation_covs = []
        for behavior_on_target in self._deviations.transpose(2, 0, 1):
            col1 = []
            col2 = []
            for i, behavior_s1 in enumerate(behavior_on_target):
                for j, behavior_s2 in enumerate(behavior_on_target):
                    col1.extend(behavior_s1)
                    col2.extend(behavior_s2)
            m = np.array([col1, col2]).T
            deviation_covs.append(np.cov(m, rowvar=False, bias=True)[0, 1])
        deviation_covs = np.array(deviation_covs)

        # apply formula
        return np.sum(deviation_means ** 2) + np.sum(deviation_vars) / t + (1 - 1/t) * np.sum(deviation_covs)
    
    def _get_true_performance_mean_on_conditioned_data(self, instance_indices, t=None):
        if t is None:
            t = self._t_checkpoints
        deviation_means = self._deviations[:, instance_indices].mean(axis=0)
        deviation_vars = self._deviations[:, instance_indices].var(axis=0)
        term1 = (deviation_means**2).mean(axis=0).sum(axis=0)
        term2 = (deviation_vars.mean(axis=0).sum(axis=0)) / t  # in the conditional variance, the deviations are independent
        return term1 + term2

    def _get_true_performance_var_for_two_instances_on_iid_data(self, t=None):
        if t is None:
            t = self._t_checkpoints
        
        # check whether actual ground truth can be computed or needs to be approximated
        num_available_ensemble_members = self._deviations.shape[0]
        num_available_instances = self._deviations.shape[1]
        
        num_possible_datasets = num_available_instances ** 2
        num_possible_ensembles = num_available_ensemble_members**4
        required_table_entries_for_exact_computation = num_possible_datasets * num_possible_ensembles
        computation_feasible = required_table_entries_for_exact_computation <= self.max_ground_truth_table_size
        if not computation_feasible:
            self.logger.warning(
                "No exact computation of ground truth feasible for V[Z_2t], "
                f"because {required_table_entries_for_exact_computation} table entries would be required, "
                f"but only {self.max_ground_truth_table_size} are granted. Using an approximation."
                )

        # get all 14 cov terms for the independent instances and ensemble members
        gtc = GroundTruthComputer(deviations=self._deviations)
        ground_truth_table = gtc.get_ground_truth_table_under_sample_iid_assumption(
            max_entries=None if computation_feasible else self.max_ground_truth_table_size,
            seed=0,
            logger=self.logger
        )
        self._covariances_by_instance_pairs_iid = gtc.get_covariance_terms_for_each_instance_pair(ground_truth_table)
        mask = self._covariances_by_instance_pairs_iid["i_1"] == self._covariances_by_instance_pairs_iid["i_2"]
        self.cov_terms = (
            self._covariances_by_instance_pairs_iid[mask].drop(columns=["i_1", "i_2"]).mean(axis=0).to_list() +
            self._covariances_by_instance_pairs_iid[~mask].drop(columns=["i_1", "i_2"]).mean(axis=0).to_list()
        )

        # multiply cov terms with the proper coefficients
        out = []
        _n = 2  # by default we compute the variance for 2 instances, because then we needto take into account covariances across two instances
        for _t in t:
            coefs = self.get_coefficients_for_covariances_for_variance(_t)
            coefs = np.concat([coefs, coefs])
            terms = coefs * self.cov_terms
            out.append(float(sum(terms[:7] / (_n * _t**3) + terms[7:] * (_n - 1) / (_n * _t**3))))
        return np.array(out)
    
    def _get_true_performance_var_on_conditioned_data(self, instance_indices, t=None):

        if t is None:
            t = self._t_checkpoints

        # get the 7 covariance terms for *every ordered pair* of instances with index in `instance_indices`
        gtc = GroundTruthComputer(deviations=self._deviations[:, instance_indices])
        self._covariances_by_instance_pairs_conditioned = gtc.get_covariance_terms_for_each_instance_pair(gtc.get_conditional_ground_truth_table())
        cov_terms = self._covariances_by_instance_pairs_conditioned.drop(columns=["i_1", "i_2"]).mean(axis=0).values

        # multiply cov terms with the proper coefficients
        out = []
        for _t in t:
            coefs = self.get_coefficients_for_covariances_for_variance(_t)
            terms = coefs * cov_terms
            out.append(float(sum(terms / _t**3)))
        return np.array(out)
    
    
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
        rf = RandomForestClassifier(
            n_estimators=self._num_possible_ensemble_members,
            random_state=self._ensemble_seed
            ).fit(self.X_train, self.y_train)
        ensemble_members = list(rf)
        classes_ = list(rf.classes_)
        indices = [classes_.index(i) for i in self.y]
        self.y_oh = np.eye(len(classes_))[indices]
        self._predictions = np.array([t.predict_proba(self.X) for t in ensemble_members])
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

        self._true_parameters = {}
        call_definitions = [
            ("E[Z_nt|D_val]", self._get_true_performance_mean_on_conditioned_data, {"instance_indices": self._indices_val}),
            ("V[Z_nt|D_val]", self._get_true_performance_var_on_conditioned_data, {"instance_indices": self._indices_val}),
            ("E[Z_nt]", self.get_true_performance_mean_on_iid_data, {}),
            ("V[Z_nt]", self._get_true_performance_var_for_two_instances_on_iid_data, {})
        ]
        for p, fun, kwargs in call_definitions:
            if p in self.captured_parameters:
                self.logger.info(f"Computing ground truth for {p}")
                self._true_parameters[p] = fun(**kwargs)
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

        # update knowledge of all approaches
        member_id = next(self.ensemble_member_id_generator) if (self._ensemble_prefix is None or self._t >= len(self._ensemble_prefix)) else self._ensemble_prefix[self._t]
        self._history_of_member_ids.append(member_id)
        matrix = self._predictions[member_id, self._indices_val]
        self._t += 1
        self.logger.info(f"Starting round {self._t}")
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
