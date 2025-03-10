import numpy as np
import pandas as pd
import openml
from time import time
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from asforests.cb_computer import EnsemblePerformanceAssessor

from experiments.benchmark.result_storage import ResultStorage


class Benchmark:

    def __init__(self,
                 openmlid,
                 data_seed,
                 ensemble_seed,
                 application_size,
                 validation_size,
                 is_classification
                 ):
        
        # configuration variables
        self._openmlid = openmlid
        self._data_seed = data_seed
        self._ensemble_seed = ensemble_seed
        self._application_size = application_size
        self._validation_size = validation_size
        self._is_classification = is_classification
        self._rs_data = np.random.RandomState(data_seed)
        self._rs_ensemble = np.random.RandomState(ensemble_seed)

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
    def seed(self):
        return self._seed

    @property
    def application_size(self):
        return self._application_size

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
        return self._deviations[:, self._indices_val].mean(axis=(0, 1))

    @property
    def deviation_vars(self):
        return self._vars

    @property
    def deviation_vars_val(self):
        """
        :return: the array V[D^1_1 | validation data] with one entry for each label
        """
        return self._deviations[:, self._indices_val].var(axis=(0, 1))

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

    def get_true_performance_mean_on_val_data(self):
        t = self._t_checkpoints
        return np.sum(self.deviation_means_val ** 2) + np.sum(self.deviation_vars_val) / t + (1 - 1/t) * np.sum(self.deviation_covs_val)

    def get_true_performance_var_on_val_data(self):

        # TODO: Implement this
        return np.zeros(len(self._t_checkpoints))

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
        splitter_val = StratifiedShuffleSplit(n_splits=1, random_state=self._rs_data, train_size=self.validation_size) if self.is_classification else ShuffleSplit(n_splits=1, random_state=self._rs_data, train_size=self.validation_size)
        validation_indices, rest_indices = next(splitter_val.split(self.X, self.y))
        splitter_rest = StratifiedShuffleSplit(n_splits=1, random_state=self._rs_data, test_size=self.application_size) if self.is_classification else ShuffleSplit(n_splits=1, random_state=self._rs_data, test_size=self.application_size)
        train_indices, oos_indices = next(splitter_rest.split(self.X[rest_indices], self.y[rest_indices]))
        train_indices = rest_indices[train_indices]
        oos_indices = rest_indices[oos_indices]
        train_indices.sort(), oos_indices.sort(), validation_indices.sort()
        assert len(set(train_indices) | set(validation_indices) | set(oos_indices)) == len(self.X)
        self._indices_train = train_indices
        self._indices_val = validation_indices
        self._indices_oos = oos_indices

    def _compute_ground_truth_parameters(self):

        if self._means is not None:
            print(f"Warning: ground truth has already been computed, skipping.")
            return
        
        # send log message
        print(f"Computing ground truth parameter values.")
        t_start = time()

        # compute 3D tensor with all deviations of all ensemble members on all data points
        rf = RandomForestClassifier(n_estimators=10**2, random_state=self._rs_ensemble).fit(self.X_train, self.y_train)
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
    
    
    def reset(self, approaches: dict, t_checkpoints: list):

        # initialize deviations if this has not happned yet
        if self._deviations is None:
            self._load_data()
            self._compute_ground_truth_parameters()
        
        # create prediction matrix generator
        rs = np.random.RandomState(self._ensemble_seed)
        def f():
            while True:
                yield self._predictions[rs.choice(range(len(self._deviations))), self._indices_val]

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
            "E[Z_nt]": self.get_true_performance_mean_on_val_data(),
            "V[Z_nt]": self.get_true_performance_var_on_val_data()
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
            approach_obj.receive_predictions_of_new_ensemble_member(matrix)
            self._result_storage.add_estimates(approach_name, self.t, {
                "E[Z_nt]": approach_obj.estimate_performance_mean(self._t_checkpoints),
                "V[Z_nt]": approach_obj.estimate_performance_var(self._t_checkpoints)
            })
