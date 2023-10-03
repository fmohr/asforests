import numpy as np
import time
from scipy.stats import bootstrap
import sklearn.ensemble
import logging
from ._grower import ForestGrower


class RandomForestClassifier(sklearn.ensemble.RandomForestClassifier):
    
    def __init__(self, step_size = 5, epsilon = 0.01, min_trees = 10, max_trees = None, **kwargs):
        if "n_estimators" in kwargs:
            raise ValueError(
                "This is an automatically stopping RF, which does not accept the n_estimators parameter."
                "Please use max_trees instead if you want to limit the number."
            )
        super().__init__(**kwargs)
        
        if "n_jobs" in kwargs and kwargs["n_jobs"] > step_size:
            raise ValueError(f"The number of jobs cannot be bigger than step_size.")
        
        self.step_size = step_size
        self.epsilon = epsilon
        self.min_trees = min_trees
        self.max_trees = max_trees
        self.logger = logging.getLogger("ASRFClassifier")
        
    def __str__(self):
        return "ASRFClassifier"
    
    def predict_tree_proba(self, tree_id, X):
        tree = self.estimators_[tree_id]
        dist = tree.predict_proba(X)
        return dist, tree.classes_
    
    def get_score_generator(self, X, y, validation_size = 0.0, random_state = None):
        
        # check whether performance is checked based on OOB or a separate validation fold
        oob = validation_size == 0
        if not oob:
            X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, test_size = validation_size, random_state = random_state)
        else:
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # memorize labels
        labels = list(np.unique(y)) # it is ok to use all labels here, even if validation is used
        
        # one hot encoding of target
        n, k = len(y_val), len(labels)
        Y = np.zeros((n, k))
        for i, true_label in enumerate(y_val):
            Y[i,labels.index(true_label)] = 1
        
        # stuff to efficiently compute OOB
        if oob:
            n_samples = y.shape[0]
            n_samples_bootstrap = sklearn.ensemble._forest._get_n_samples_bootstrap(
                n_samples,
                self.max_samples,
            )
            def get_unsampled_indices(tree):
                return sklearn.ensemble._forest._generate_unsampled_indices(
                    tree.random_state,
                    n_samples,
                    n_samples_bootstrap,
                )

        # this is a variable that is being used by the supplier
        self.y_prob_pred = np.zeros(Y.shape)
        self.all_indices = list(range(Y.shape[0]))
        
        empty_distribution = np.empty((len(y), len(labels)))
        empty_distribution[:] = np.nan

        def f():
            
            while True: # the generator will add trees forever
                
                # add a new tree(s)
                start = time.time()
                self.n_estimators += self.step_size
                super(RandomForestClassifier, self).fit(X_train, y_train)
                traintime = time.time() - start
                traintime_per_added_tree = traintime / self.step_size * self.n_jobs

                # update distribution based on last trees
                for t in range(self.n_estimators - self.step_size, self.n_estimators):

                    # get prediction of tree on the indices relevant for it (the others are left to nan)
                    start = time.time()
                    last_tree = self.estimators_[t]
                    y_prob_tree = empty_distribution.copy()
                    indices_val = get_unsampled_indices(last_tree) if oob else self.indices_val
                    y_prob_tree_on_indices, classes_ = self.predict_tree_proba(t, X[indices_val])
                    y_prob_tree[indices_val] = y_prob_tree_on_indices
                    pred_time = time.time() - start

                    # update forest's prediction
                    start = time.time()
                    self.y_prob_pred[indices_val] = (y_prob_tree_on_indices + t * self.y_prob_pred[indices_val]) / (t + 1) # this will converge according to the law of large numbers
                    update_time = time.time() - start

                    yield y_prob_tree, traintime_per_added_tree, pred_time, update_time
        
        return f() # creates the generator and returns it
    
    def reset(self):
        # set numbers of trees to 0
        self.warm_start = False
        self.estimators_ = []
        self.n_estimators = 0
        self.warm_start = True
    
    def fit(self, X, y):
        
        self.reset()
        gen = self.get_score_generator(X, y)
        
        # always use Brier score supplier
        grower = ForestGrower(gen,  d = 1, logger = self.logger, random_state = self.random_state, **self.args)
        grower.grow()
        self.histories = grower.histories