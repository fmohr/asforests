import logging
import numpy as np
import sklearn.datasets
from sklearn import *
import unittest
from parameterized import parameterized
import itertools as it
import time
import openml
import pandas as pd
from asforests import RandomForestClassifier
from asforests import ExtraTreesClassifier
from asforests import RandomForestRegressor
from asforests import ExtraTreesRegressor

import warnings



def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0]
    num_rows = len(df)
        
    # prepare label column as numpy array
    print(f"Read in data frame. Size is {len(df)} x {len(df.columns)}.")
    X = np.array(df.drop(columns=[ds.default_target_attribute]).values)
    y = np.array(df[ds.default_target_attribute].values)
    if y.dtype != int:
        y_int = np.zeros(len(y)).astype(int)
        vals = np.unique(y)
        for i, val in enumerate(vals):
            mask = y == val
            y_int[mask] = i
        y = y_int
        
    print(f"Data is of shape {X.shape}.")
    return X, y

class TestASForests(unittest.TestCase):
    
    

    def setUpClass():
        # setup logger for this test suite
        logger = logging.getLogger('asforests_test')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # configure lccv logger (by default set to WARN, change it to DEBUG if tests fail)
        asforests_logger = logging.getLogger("ASRFClassifier")
        asforests_logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        asforests_logger.addHandler(ch)
        
    def setUp(self):
        self.logger = logging.getLogger("asforests_test")
        self.asforests_logger = logging.getLogger("ASRFClassifier")
        warnings.filterwarnings("error")
        
    def testStringCastability(self):
        rf = RandomForestClassifier()
        str(rf)
        rf = RandomForestRegressor()
        str(rf)
        rf = ExtraTreesClassifier()
        str(rf)
        rf = ExtraTreesRegressor()
        str(rf)
        
    def testClonability(self):
        rf = RandomForestClassifier()
        sklearn.base.clone(rf)
        rf = RandomForestRegressor()
        sklearn.base.clone(rf)
        rf = ExtraTreesClassifier()
        sklearn.base.clone(rf)
        rf = ExtraTreesRegressor()
        sklearn.base.clone(rf)
        
    def testParallelization(self):
        return
        X, y = get_dataset(531)
        
        start = time.time()
        rf = ExtraTreesRegressor()
        rf.fit(X, y)
        runtime_simple = time.time() - start
        print(runtime_simple, rf.n_estimators)

    def test_reproducibility(self):
        self.logger.info("Start Test on Reproducibility")
        seeds = 3
        
        return
    
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        for seed in range(seeds):
            for step_size in [1, 2, 5, 10]:
                self.logger.info(f"Run test for seed {seed} and step_size {step_size} on classification.")

                # create train-test-split
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = seed)

                # train the two forests
                rf1 = RandomForestClassifier(step_size = step_size, epsilon = 0.01, random_state = seed)
                rf1.fit(X_train, y_train)
                rf2 = RandomForestClassifier(step_size = step_size, epsilon = 0.01, random_state = seed)
                rf2.fit(X_train, y_train)
                y1_hat = rf1.predict_proba(X_test)
                y2_hat = rf2.predict_proba(X_test)

                # assume exact same test set, also when train set is double
                np.testing.assert_array_equal(y1_hat, y2_hat)
                self.logger.info(f"Finished test for seed {seed}")
                
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        for seed in range(seeds):
            for step_size in [1, 5, 10, 50]:
                self.logger.info(f"Run test for seed {seed} and step_size {step_size} on regression.")

                # create train-test-split
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = seed)

                # train the two forests
                eps = 10
                w_min = step_size + 1
                rf1 = RandomForestRegressor(step_size = step_size, w_min = w_min, epsilon = eps, random_state = seed)
                rf1.fit(X_train, y_train)
                self.logger.info("First forest ready.")
                rf2 = RandomForestRegressor(step_size = step_size, w_min = w_min, epsilon = eps, random_state = seed)
                rf2.fit(X_train, y_train)
                y1_hat = rf1.predict(X_test)
                y2_hat = rf2.predict(X_test)

                # assume exact same test set, also when train set is double
                np.testing.assert_array_equal(y1_hat, y2_hat)
                self.logger.info(f"Finished test for seed {seed}")
            
    def test_that_final_distribution_is_similar_to_huge_forest(self):
        for load_fun in [sklearn.datasets.load_iris]:
            X, y = load_fun(return_X_y=True)
            for seed in range(1):

                self.logger.info(f"Testing quality of final distribution for seed {seed} on {load_fun}")

                # create train-test-split
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = seed)

                # train the two forests
                rf1 = RandomForestClassifier(random_state = seed)
                rf1.fit(X_train, y_train)
                rf2 = sklearn.ensemble.RandomForestClassifier(random_state = seed, n_estimators = 2000)
                rf2.fit(X_train, y_train)
                p_as = np.max(rf1.predict_proba(X_test), axis=1)
                p_orig = np.max(rf2.predict_proba(X_test), axis=1)

                # compute deviations
                deviations = [abs(q_orig - q_as) for q_orig, q_as in zip(p_orig, p_as)]
                self.logger.info(f"Trained forest with {rf1.n_estimators} trees. 90percentile deviation: {np.percentile(deviations, 90)}. Max deviation: {np.max(deviations)}")
                
                # assume exact same test set, also when train set is double
                print(deviations)
                print(np.percentile(deviations, 90))
                print(np.max(deviations))
                self.assertTrue(np.percentile(deviations, 90) < 0.01)
                self.assertTrue(np.max(deviations) < 0.05)

        for load_fun in [sklearn.datasets.load_diabetes]:
            X, y = load_fun(return_X_y=True)
            for seed in range(1):
                
                self.logger.info(f"Testing quality of final predictions for seed {seed} on {load_fun}")

                # create train-test-split
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = seed)

                # train the two forests
                rf1 = RandomForestRegressor(random_state = seed)
                rf1.fit(X_train, y_train)
                rf2 = sklearn.ensemble.RandomForestRegressor(random_state = seed, n_estimators = 2000)
                rf2.fit(X_train, y_train)
                p_as = rf1.predict(X_test)
                p_orig = rf2.predict(X_test)

                # compute deviations
                deviations = [abs(q_orig - q_as) for q_orig, q_as in zip(p_orig, p_as)]
                self.logger.info(f"Trained forest with {rf1.n_estimators} trees. 90percentile deviation: {np.percentile(deviations, 90)}. Max deviation: {np.max(deviations)}")

                # assume exact same test set, also when train set is double
                self.assertTrue(np.percentile(deviations, 90) < 5)
                self.assertTrue(np.max(deviations) < 10)
            
    def test_that_final_distribution_independent_of_step_size(self):
        return
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        for seed in range(1):
            
            # create train-test-split
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = seed)
            
            # train the two forests
            step_size1 = 1
            step_size2 = 2
            eps = 0.3
            rf1 = RandomForestClassifier(step_size = step_size1, epsilon = eps, random_state = seed, max_trees = 2)
            rf1.fit(X_train, y_train)
            rf2 = RandomForestClassifier(step_size = step_size2, epsilon = eps, random_state = seed, max_trees = 2)
            rf2.fit(X_train, y_train)
            y1_hat = rf1.predict_proba(X_test)
            y2_hat = rf2.predict_proba(X_test)
            
            # assume exact same test set, also when train set is double
            np.testing.assert_array_equal(y1_hat, y2_hat)
            
            
    def test_that_convergence_criteria_independent_of_step_size(self):
        return
        self.logger.info("Start Test that checks that the convergence does not rely on the step-size module the step size itself")
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        for seed in range(1):
            self.logger.info(f"Run test for seed {seed}")
            
            # create train-test-split
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = seed)
            
            # train the two forests
            step_size1 = 1
            step_size2 = 5
            eps = 0.3
            rf1 = RandomForestClassifier(step_size = step_size1, epsilon = eps, random_state = seed)
            rf1.fit(X_train, y_train)
            rf2 = RandomForestClassifier(step_size = step_size2, epsilon = eps, random_state = seed)
            rf2.fit(X_train, y_train)
            
            # assume exact same test set, also when train set is double
            gap = abs(rf1.n_estimators - rf2.n_estimators)
            self.assertTrue(gap <= abs(step_size2 - step_size1), f"The gap in trained trees is {gap} but should be at most 5. Trained {rf1.n_estimators} trees with step_size {step_size1} and {rf2.n_estimators} trees with step_size {step_size2}")
            self.logger.info(f"Finished test for seed {seed}")
            
    def test_that_convergence_is_independent_of_step_size(self):
        return
        self.logger.info("Start Test that checks that the convergence does not rely on the step-size module the step size itself")
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        for seed in range(10):
            self.logger.info(f"Run test for seed {seed}")
            
            # create train-test-split
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = seed)
            
            # train the two forests
            step_size1 = 1
            step_size2 = 5
            eps = 0.05
            extrapolation_multiplier = 100
            rf1 = RandomForestClassifier(step_size = step_size1, epsilon = eps, extrapolation_multiplier = extrapolation_multiplier, random_state = seed)
            rf1.fit(X_train, y_train)
            rf2 = RandomForestClassifier(step_size = step_size2, epsilon = eps, extrapolation_multiplier = extrapolation_multiplier, random_state = seed)
            rf2.fit(X_train, y_train)
            
            # assume exact same test set, also when train set is double
            gap = abs(rf1.n_estimators - rf2.n_estimators)
            self.assertTrue(gap <= 10 * abs(step_size2 - step_size1), f"The gap in trained trees is {gap} but should be at most 5. Trained {rf1.n_estimators} trees with step_size {step_size1} and {rf2.n_estimators} trees with step_size {step_size2}")
            self.logger.info(f"Finished test for seed {seed}")
            
    """
        This checks whether LCCV respects the timeout
    
    #@parameterized.expand(list(it.product(preprocessors, learners, [(61, 0.0), (1485, 0.2)])))
    def test_lccv_respects_timeouts(self, preprocessor, learner, dataset):
        X, y = get_dataset(dataset[0])
        r = dataset[1]
        self.logger.info(f"Start Test LCCV when running with r={r} on dataset {dataset[0]} wither preprocessor {preprocessor} and learner {learner}")
        
        # configure pipeline
        steps = []
        if preprocessor is not None:
            pp = preprocessor()
            if "copy" in pp.get_params().keys():
                pp = preprocessor(copy=False)
            steps.append(("pp", pp))
        learner_inst = learner()
        if "warm_start" in learner_inst.get_params().keys(): # active warm starting if available, because this can cause problems.
            learner_inst = learner(warm_start=True)
        steps.append(("predictor", learner_inst))
        pl = sklearn.pipeline.Pipeline(steps)
        
        timeout = 1.5
        
        # do tests
        try:
            
            # run 80lccv
            self.logger.info("Running 80LCCV")
            start = time.time()
            score_80lccv = lccv.lccv(sklearn.base.clone(pl), X, y, r=r, target_anchor=.8, MAX_EVALUATIONS=5, timeout=timeout)[0]
            end = time.time()
            runtime_80lccv = end - start
            self.assertTrue(runtime_80lccv <= timeout, msg=f"Permitted runtime exceeded. Permitted was {timeout}s but true runtime was {runtime_80lccv}")
        except ValueError:
            print("Skipping case in which training is not possible!")
"""