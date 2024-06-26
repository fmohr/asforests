import numpy as np
from scipy.stats import bootstrap
import sklearn.ensemble
import logging
from ._grower import EnsembleGrower


class ExtraTreesClassifier(sklearn.ensemble.ExtraTreesClassifier):
    
    def __init__(self, step_size = 5, w_min = 50, delta = 10, epsilon = 0.01, extrapolation_multiplier = 1000, bootstrap_repeats = 5, max_trees = None, stop_when_horizontal = True, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, n_jobs=None, random_state=None, verbose=0, class_weight=None, ccp_alpha=0.0, max_samples=None):
        self.kwargs = {
            "n_estimators": 0, # will be increased steadily
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "bootstrap": bootstrap,
            "oob_score": False,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose,
            "class_weight": class_weight,
            "ccp_alpha": ccp_alpha,
            "max_samples": max_samples,
            "warm_start": True
        }
        super().__init__(**self.kwargs)
        
        if random_state is None:
            random_state = 0
        if type(random_state) == np.random.RandomState:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)
   
        self.step_size = step_size
        self.w_min = w_min
        self.epsilon = epsilon
        self.extrapolation_multiplier = extrapolation_multiplier
        self.max_trees = max_trees
        self.bootstrap_repeats = bootstrap_repeats
        self.stop_when_horizontal = stop_when_horizontal
        self.args = {
            "step_size": step_size,
            "w_min": w_min,
            "delta": delta,
            "epsilon": epsilon,
            "extrapolation_multiplier": extrapolation_multiplier,
            "bootstrap_repeats": bootstrap_repeats,
            "max_trees": max_trees,
            "stop_when_horizontal": stop_when_horizontal
        }
        self.logger = logging.getLogger("ASRFClassifier")
        
    def __str__(self):
        return "ASRFClassifier"
    
    def predict_tree_proba(self, tree_id, X):
        return self.estimators_[tree_id].predict_proba(X)
    
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
        
        # create a function that can efficiently compute the Brier score for a probability distribution
        def get_brier_score(Y_prob):
            return np.mean(np.sum((Y_prob - Y)**2, axis=1))
        
        # this is a variable that is being used by the supplier
        self.y_prob_pred = np.zeros(Y.shape)
        self.all_indices = list(range(Y.shape[0]))
        
        def f():
            
            while True: # the generator will add trees forever
                
                # add a new tree(s)
                start = time.time()
                self.n_estimators += self.step_size
                super(RandomForestClassifier, self).fit(X_train, y_train)
                traintime = time.time() - start

                # update distribution based on last trees
                for t in range(self.n_estimators - self.step_size, self.n_estimators):

                    start = time.time()
                    
                    # get t-th last tree
                    last_tree = self.estimators_[t]

                    # get prediction of tree on the indices relevant for it
                    relevant_indices = get_unsampled_indices(last_tree) if oob else self.all_indices # this is what J is in the paper
                    y_prob_tree, classes_ = self.predict_tree_proba(t, X_val[relevant_indices])
                    
                    # update forest's prediction
                    self.y_prob_pred[relevant_indices] = (y_prob_tree + t * self.y_prob_pred[relevant_indices]) / (t + 1) # this will converge according to the law of large numbers

                    pred_time = time.time() - start
                    start = time.time()
                    score = get_brier_score(self.y_prob_pred)
                    score_time = time.time() - start
                    
                    yield score, traintime / self.step_size, pred_time, score_time
        
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