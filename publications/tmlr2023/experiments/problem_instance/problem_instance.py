from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

import pandas as pd
import numpy as np
import openml

import logging
from time import time

from experiments.benchmark._util import get_unique_prediction_matrices
from experiments.benchmark._ground_truth_computer import GroundTruthComputer

class ProblemInstance:

    def __init__(
            self,
            data_description,
            is_classification,
            data_seed,
            ensemble_seed,
            num_possible_ensemble_members,
            training_instances_per_class,
            num_samples_allowed_for_ground_truth_approximation,
            validation_size,
            n_checkpoints,
            t_checkpoints,
            logger=None,
            predictions=None,
            deviations=None,
            true_means_for_iid_case=None,
            true_vars_for_iid_case=None,
            true_means_for_cond_case=None,
            true_vars_for_cond_case=None
        ):
        
        # config vars
        self.data_description = data_description
        self.is_classification = is_classification
        self.num_possible_ensemble_members = num_possible_ensemble_members
        self.training_instances_per_class = training_instances_per_class
        self.validation_size = validation_size
        self.logger = logger if logger is not None else logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
        self.data_seed = data_seed
        self.ensemble_seed = ensemble_seed
        self.num_samples_allowed_for_ground_truth_approximation = num_samples_allowed_for_ground_truth_approximation
        self.t_checkpoints = t_checkpoints
        self.n_checkpoints = n_checkpoints

        # state vars
        self._X = self._y = self._y_oh = None
        self._indices_train = self._indices_val = self._indices_oos = None
        self._predictions = predictions
        self._deviations = deviations
        self._true_means_for_iid_case = true_means_for_iid_case
        self._true_vars_for_iid_case = true_vars_for_iid_case
        self._true_means_for_cond_case = true_means_for_cond_case
        self._true_vars_for_cond_case = true_vars_for_cond_case
        self._prediction_matrix_generator = None

    @property
    def X(self):
        if self._X is None:
            self._load_data()
        return self._X
    
    @property
    def y(self):
        if self._y is None:
            self._load_data()
        return self._y
    
    @property
    def num_labels(self):
        return len(np.unique(self.y))
    
    @property
    def y_oh(self):
        if self._y_oh is None:
            self._compute_predictions_and_deviations()
        return self._y_oh
    
    @property
    def predictions(self):
        if self._predictions is None:
            self._compute_predictions_and_deviations()
        return self._predictions
    
    @property
    def deviations(self):
        if self._deviations is None:
            self._compute_predictions_and_deviations()
        return self._deviations

    @property
    def means_iid(self):
        if self._true_means_for_iid_case is None:
            self._approximate_ground_truth_parameters()
        return self._true_means_for_iid_case
    
    @property
    def vars_iid(self):
        if self._true_vars_for_iid_case is None:
            self._approximate_ground_truth_parameters()
        return self._true_vars_for_iid_case
    
    @property
    def means_cond(self):
        if self._true_means_for_cond_case is None:
            self._approximate_ground_truth_parameters()
        return self._true_means_for_cond_case
    
    @property
    def vars_cond(self):
        if self._true_vars_for_cond_case is None:
            self._approximate_ground_truth_parameters()
        return self._true_vars_for_cond_case

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
            if type(self.data_description) in [int, np.integer]:
                openmlid = self.data_description
                ds = openml.datasets.get_dataset(
                    openmlid,
                    download_data=False,
                    download_qualities=False,
                    download_features_meta_data=False
                )
                df = ds.get_data()[0]

                # prepare data with label encoding for categorical attributes
                self._X = np.array(df.drop(columns=[ds.default_target_attribute]).values)
                self._y = np.array(df[ds.default_target_attribute].values)
            elif type(self.data_description) == tuple:
                self._X, self._y = self.data_description
            else:
                raise ValueError(f"Unsupported data description of type {type(self.data_description)}")

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

        # extract validation data
        rs_data = np.random.RandomState(self.data_seed)
        splitter_val = StratifiedShuffleSplit(n_splits=1, random_state=rs_data, train_size=self.validation_size) if self.is_classification else ShuffleSplit(n_splits=1, random_state=rs_data, train_size=self.validation_size)
        validation_indices, rest_indices = next(splitter_val.split(self.X, self.y))

        # separate training and out-of-sample data from the rest that is not validation
        training_size_relative = self.training_instances_per_class / label_count[minority_class] if self.training_instances_per_class >= 1 else self.training_instances_per_class
        self.logger.info(f"Using {np.round(training_size_relative * 100, 2)}% of the  data for training")
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
            pl.fit(self._X[self._indices_train], self._y[self._indices_train])
            self._X = pl.transform(self._X) # apply transformation to all data, not just training
    
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
            seed=self.ensemble_seed,
            num_matrices=self.num_possible_ensemble_members,
            max_tries=100,
            rf_kwargs={ # make trees very weak and random to maximize diversity in the ensemble (besides, this makes training faster)
                "max_depth": 1,
                "max_features": 1
            },
            logger=self.logger
        )

        # memorize prediction matrices
        indices = [classes.index(i) for i in self.y]
        self._y_oh = np.eye(len(classes))[indices]
        self._predictions = np.array(prediction_matrices[:self.num_possible_ensemble_members])
        assert self._predictions.shape == (self.num_possible_ensemble_members, self.X.shape[0], self._y_oh.shape[1])
        if np.any(np.isnan(self._predictions)):
            raise ValueError(f"predictions have nan entries: {self._predictions}")
        self._deviations = self._predictions - self._y_oh
        if np.any(np.isnan(self._deviations)):
            raise ValueError(f"deviations have nan entries: {self._deviations}")

        # check that predictions of ensembles are pairwise different
        for i, p1 in enumerate(self._predictions):
            for j, p2 in enumerate(self._predictions[:i]):
                assert not np.all(np.isclose(p1, p2)), f"Predictions of ensemble member {i} and {j} are identical."
        self.logger.info(f"Prediction and deviation computation finished after {int(1000 * (time() - t_start))}ms.")    

    def _approximate_ground_truth_parameters(self, num_samples=None, num_samples_per_job=None, n_jobs=1):


        if num_samples is None:
            num_samples = self.num_samples_allowed_for_ground_truth_approximation

        # approximate ground truth for iid case
        gtc_iid = GroundTruthComputer(deviations=self.deviations)
        self._true_means_for_iid_case, self._true_vars_for_iid_case = gtc_iid.approximate_true_parameters_in_iid_setting_by_sampling(
            n_checkpoints=self.n_checkpoints,
            t_checkpoints=self.t_checkpoints,
            num_samples=num_samples,
            num_samples_per_job=num_samples_per_job,
            n_jobs=n_jobs
        )

        # compute exact ground truth for conditional case
        gtc_cond = GroundTruthComputer(deviations=self.deviations[:, self._indices_val])
        self._true_means_for_cond_case, self._true_vars_for_cond_case = gtc_cond.approximate_true_parameters_in_cond_setting_by_sampling(
            t_checkpoints=self.t_checkpoints,
            num_samples=num_samples,
            num_samples_per_job=num_samples_per_job,
            n_jobs=n_jobs
        )

    def to_dict(self):
        out = {
            "data_description": self.data_description,
            "is_classification": self.is_classification,
            "data_seed": self.data_seed,
            "ensemble_seed": self.ensemble_seed,
            "num_possible_ensemble_members": self.num_possible_ensemble_members,
            "training_instances_per_class": self.training_instances_per_class,
            "num_samples_allowed_for_ground_truth_approximation": self.num_samples_allowed_for_ground_truth_approximation,
            "validation_size": self.validation_size,
            "n_checkpoints": self.n_checkpoints.tolist() if self.n_checkpoints is not None else None,
            "t_checkpoints": self.t_checkpoints.tolist() if self.t_checkpoints is not None else None
        }
        
        if self._deviations is not None:
            out["deviations"] = self._deviations.tolist()
        if self._true_means_for_iid_case is not None:
            out["true_means_for_iid_case"] = self._true_means_for_iid_case.tolist()
        if self._true_vars_for_iid_case is not None:
            out["true_vars_for_iid_case"] = self._true_vars_for_iid_case.tolist()
        if self._true_means_for_cond_case is not None:
            out["true_means_for_cond_case"] = self._true_means_for_cond_case.tolist()
        if self._true_vars_for_cond_case is not None:
            out["true_vars_for_cond_case"] = self._true_vars_for_cond_case.tolist()
        return out

    @classmethod
    def from_dict(cls, dict):
        for field in ["n_checkpoints", "t_checkpoints", "true_means_for_iid_case", "true_vars_for_iid_case", "true_means_for_cond_case", "true_vars_for_cond_case"]:
            if field in dict:
                dict[field] = np.array(dict[field])
        return ProblemInstance(**dict)