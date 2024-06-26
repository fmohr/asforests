import sklearn.model_selection
import sklearn.preprocessing
import numpy as np
import time


class ForestOutputSupplier:

    def __init__(self,
                 task_type,
                 forest,
                 X,
                 y,
                 normalize_outputs,
                 validation_size=0,
                 min_step_size=1,
                 random_state=None
                 ):

        if task_type not in ["classification", "regression"]:
            raise ValueError(f"task_type must be explicitly classification or regression.")

        self.task_type = task_type
        self.is_classification = task_type == "classification"
        self.forest = forest
        self.min_step_size = min_step_size

        # if outputs should be normalized, conduct range normalization
        if normalize_outputs:
            min_y = np.min(y)
            max_y = np.max(y)
            self.normalizer = lambda o: (o - min_y) / (max_y - min_y)
        else:
            self.normalizer = lambda o: o

            # check whether performance is checked based on OOB or a separate validation fold
        self.oob = validation_size == 0
        if not self.oob:
            self.X_train, self.X_val, self.y_train, self.y_val =\
                sklearn.model_selection.train_test_split(X, y, test_size=validation_size, random_state=random_state)
        else:
            self.X_train, self.X_val, self.y_train, self.y_val = X, X, y, y

        # stuff to efficiently compute OOB
        if self.oob:
            n_samples = y.shape[0]
            n_samples_bootstrap = sklearn.ensemble._forest._get_n_samples_bootstrap(
                n_samples,
                self.forest.max_samples,
            )

            self.get_unsampled_indices = lambda tree: sklearn.ensemble._forest._generate_unsampled_indices(
                    tree.random_state,
                    n_samples,
                    n_samples_bootstrap,
                )

        self.empty_distribution = np.empty((len(self.y_val), len(np.unique(y))))
        self.empty_distribution[:] = np.nan

        # state variables
        self.fit_times = []
        self.labels = None  # will be determined with first fit

    def __call__(self, num_members):

        # add new tree(s)
        num_members = max([num_members, self.min_step_size])
        start = time.time()
        self.forest.n_estimators += num_members
        self.forest.fit_(self.X_train, self.y_train)
        self.fit_times.append(time.time() - start)
        if self.labels is None and self.is_classification:
            self.labels = self.forest.classes_

        # update distribution based on last trees
        outputs = []
        for t in range(self.forest.n_estimators - num_members, self.forest.n_estimators):

            # get prediction of tree on the indices relevant for it (the others are left to nan)
            last_tree = self.forest.estimators_[t]
            indices_val = self.get_unsampled_indices(last_tree) if self.oob else self.indices_val
            tree_output = self.normalizer(
                last_tree.predict_proba(self.X_val[indices_val]) if self.is_classification else last_tree.predict(self.X_val[indices_val]).reshape(-1, 1)
            )

            # insert the behavior into the general format, which implies nan entries in case of OOB
            if self.oob:
                tree_output_extended = self.empty_distribution.copy()
                tree_output_extended[indices_val] = tree_output
            else:
                tree_output_extended = tree_output
            outputs.append(tree_output_extended)

        # shape an array and return it
        return np.array(outputs)

    @property
    def shape(self):
        return self.empty_distribution.shape
