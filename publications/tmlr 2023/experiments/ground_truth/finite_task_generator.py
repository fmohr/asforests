import numpy as np
import itertools as it
from tqdm import tqdm
from sklearn.datasets import make_classification, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class FiniteTaskGenerator:

    def __init__(
            self,
            num_possible_members,
            num_possible_instances,
            num_visible_instances,
            num_targets,
            method="finite_rf",
            method_kwargs={},
            rs=None,
            ignore_missing_uniqueness=False
    ):

        # configuration
        self.num_possible_members = num_possible_members
        self.num_possible_instances = num_possible_instances
        self.num_visible_instances = num_visible_instances
        self.num_targets = num_targets
        self.method = method
        self.method_kwargs = method_kwargs
        if rs is None:
            rs = np.random.RandomState()
        self.rs = rs
        self.ignore_missing_uniqueness = ignore_missing_uniqueness

        # state
        self.deviations_ = None
        self.means_ = None
        self.vars_ = None
        self.covars_ = None

    def generate_deviations(self, rs=None):
        if self.method == "uniform":
            self.deviations_ = self.rs.rand(self.num_possible_members, self.num_possible_instances, self.num_targets)
        elif self.method == "finite_rf":
            X, y = make_classification(
                n_samples=self.num_possible_instances,
                n_features=10,
                n_informative=5,
                n_repeated=0,
                n_redundant=5,
                n_classes=self.num_targets,
                random_state=self.rs
            )
            X, y = fetch_openml(data_id=61, return_X_y=True)
            X = X.values#[:self.num_possible_instances]
            #y = y[:self.num_possible_instances]
            X_visible, X_hidden, y_visible, y_hidden = train_test_split(X, y, random_state=self.rs)
            X_train, X_val, y_train, y_val = train_test_split(X_visible, y_visible, random_state=self.rs)
            rf = RandomForestClassifier(n_estimators=self.num_possible_members, random_state=self.rs).fit(X_train, y_train)
            ensemble_members = list(rf)
            classes_ = list(rf.classes_)
            indices = [classes_.index(i) for i in y_val]
            y_oh = np.eye(len(classes_))[indices]
            self.deviations_ = np.array([t.predict_proba(X_val) - y_oh for t in ensemble_members])

        else:
            raise ValueError(f"Unsupported method {self.method}")

        # check whether deviations are actually unique
        if not self.ignore_missing_uniqueness:
            devs_as_str = [str(d) for d in self.deviations_]
            assert len(set(devs_as_str)) == len(devs_as_str), f"Only {len(set(devs_as_str))} unique deviation matrices"

        # these are relevant when generating the observations
        self.deviations_visible_ = self.deviations_[:, :self.num_visible_instances]

        self.means_ = None
        self.vars_ = None
        self.covars_ = None

    @property
    def deviation_mean(self):
        if self.means_ is None:
            self.means_ = self.deviations_visible_.mean(axis=(0, 1))
        return self.means_

    @property
    def deviation_var(self):
        if self.vars_ is None:
            self.vars_ = self.deviations_visible_.var(axis=(0, 1))
        return self.vars_

    @property
    def deviation_cov_across_members(self):
        if self.covars_ is None:
            col1 = []
            col2 = []

            for p1, p2 in tqdm(list(it.combinations(range(self.num_possible_members), 2))):
                col1.extend(self.deviations_visible_[p1])
                col2.extend(self.deviations_visible_[p2])

            col1 = np.array(col1)
            col2 = np.array(col2)

            self.covars_ = np.array([np.cov(col1[:, j], col2[:, j])[0, 1] for j in range(self.num_targets)])
        return self.covars_

    def sample_deviation_matrix(self):
        return self.deviations_visible_[self.rs.choice(range(self.num_possible_members))]
