import numpy as np
import pandas as pd
import openml
import os, psutil
import gc
import zlib
import logging

import time
import random

import itertools as it
import scipy.stats
from scipy.sparse import lil_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sklearn
from sklearn import metrics
import sklearn.impute
import sklearn.preprocessing

from tqdm import tqdm

import asforests

eval_logger = logging.getLogger("evalutils")


def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    print("dataset info loaded")
    df = ds.get_data()[0]
    num_rows = len(df)
    
    print("Data in memory, now creating X and y")
        
    # prepare label column as numpy array
    X = np.array(df.drop(columns=[ds.default_target_attribute]).values)
    y = np.array(df[ds.default_target_attribute].values)
    if y.dtype != int:
        y_int = np.zeros(len(y)).astype(int)
        vals = np.unique(y)
        for i, val in enumerate(vals):
            mask = y == val
            y_int[mask] = i
        y = y_int
    
    print(f"Data read. Shape is {X.shape}.")
    return X, y

def get_mandatory_preprocessing(X, y):
    
    # determine fixed pre-processing steps for imputation and binarization
    types = [set([type(v) for v in r]) for r in X.T]
    numeric_features = [c for c, t in enumerate(types) if len(t) == 1 and list(t)[0] != str]
    numeric_transformer = Pipeline([("imputer", sklearn.impute.SimpleImputer(strategy="median"))])
    categorical_features = [i for i in range(X.shape[1]) if i not in numeric_features]
    missing_values_per_feature = np.sum(pd.isnull(X), axis=0)
    eval_logger.info(f"There are {len(categorical_features)} categorical features, which will be turned into integers.")
    eval_logger.info(f"Missing values for the different attributes are {missing_values_per_feature}.")
    if len(categorical_features) > 0 or sum(missing_values_per_feature) > 0:
        categorical_transformer = Pipeline([
            ("imputer", sklearn.impute.SimpleImputer(strategy="most_frequent")),
            ("binarizer", sklearn.preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ])
        return [("impute_and_binarize", ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        ))]
    else:
        return []
    
def get_required_num_trees(prob_history, tree_score_history, eps, certainty_factor = 3):
    t = 1
    prob_history = np.array(prob_history)
    avg_summed_var = np.nanmean(np.nanvar(prob_history, axis=0).sum(1))
    std_of_scores = np.std(tree_score_history)
    instances_missed_on_avg = np.sum(np.isnan(prob_history)) / np.prod(prob_history.shape)
    #actual_instance_factor = 1 - instances_missed_on_avg
    
    num_classes = prob_history.shape[2]
    
    gap = lambda t: certainty_factor * std_of_scores / np.sqrt(t) + num_classes / (4 * t)
    
    while True:
        if gap(t) <= eps:
            return t
        t += 1

def build_full_classification_forest(openmlid, seed, zfactor, eps):
    
    print("Loading dataset")
    X, y = get_dataset(openmlid)
    print("Splitting the data")
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = seed)
    train_labels = set(np.unique(y_train))
    test_labels = set(np.unique(y_test))
    train_labels_not_in_test = [l for l in train_labels if not l in test_labels]
    test_labels_not_in_train = [l for l in test_labels if not l in train_labels]
                               
    print(f"split created. {len(train_labels_not_in_test)} labels occur in train data but not in the test data, and {len(test_labels_not_in_train)} labels occur in the test data but not in the training data.\nNow building forest.")
    
    # check whether we must pre-process
    preprocessing = get_mandatory_preprocessing(X_train, y_train)
    if preprocessing:
        pl = sklearn.pipeline.Pipeline(preprocessing)
        print(f"Modifying inputs with {pl}")
        X_train = pl.fit_transform(X_train, y_train)
        X_test = pl.transform(X_test)
        eval_logger.info(f"Dataset sizes: {X_train.shape} for training and {X_test.shape} for test data.")
    
    # prepare everything to efficiently compute the brier score
    labels = list(np.unique(y))
    n, k = len(y_test), len(labels)
    Y_test = np.zeros((n, k))
    for i, true_label in enumerate(y_test):
        Y_test[i,labels.index(true_label)] = 1
    Y_train = np.zeros((len(y_train), k))
    for i, true_label in enumerate(y_train):
        Y_train[i,labels.index(true_label)] = 1
        
    def get_brier_score(Y, Y_prob):
        return np.nanmean(np.sum((Y_prob - Y)**2, axis=1))
    
    
    start = time.time()
    history = []
    
    memory_init = psutil.Process().memory_info().rss / (1024 * 1024)
    
    Y_train_hat = np.zeros(Y_train.shape)
    Y_test_hat = np.zeros(Y_test.shape)
    
    required_trees = np.inf
    t = 1
    prob_history_oob = []
    prob_history_val = []
    score_history = []
    tree_score_history_oob = []
    tree_score_history_val = []
    step_size = 100 if len(X) < 10**4 else 10
    
    times_train = []
    times_predict_train = []
    times_update = []
    times_predict_val = []
    
    while True:
        
        # create new forest (for speed ups)
        eval_logger.info(f"Building {step_size} new trees.")
        n_jobs = 1 if len(X) < 1000 or step_size < 8 else 8
        rf = asforests.RandomForestClassifier(step_size = step_size, random_state=t, n_jobs=n_jobs)
        gen = rf.get_score_generator(X_train, y_train)
        for inner_tree_id in range(step_size):
            y_prob_tree, train_time, pred_time, update_time = next(gen)
            times_train.append(train_time)
            times_predict_train.append(pred_time)
            times_update.append(update_time)
            prob_history_oob.append(y_prob_tree)
            #print(int(1000 * train_time), int(1000 * pred_time), int(1000 * update_time))


            # update posterior distribution on test set
            start = time.time()
            y_prob_test, classes_ = rf.predict_tree_proba(inner_tree_id, X_test)
            prob_history_val.append(y_prob_test)

            # if there are test labels not known to the forest, rearrange predictions
            if len(rf.classes_) != len(labels) or any(rf.classes_ != labels):
                missing_labels = [l for l in labels if not l in rf.classes_]
                missing_predictions = np.zeros((y_prob_test.shape[0], len(missing_labels)))
                indices_augmented = (list(rf.classes_) + missing_labels)
                indices = [indices_augmented.index(l) for l in labels]
                y_prob_test = np.column_stack([y_prob_test, missing_predictions])[:,indices]

            # update forest's prediction
            mask = ~np.isnan(y_prob_tree[:,0])
            Y_train_hat[mask] = (y_prob_tree[mask] + t * Y_train_hat[mask]) / (t + 1)
            Y_test_hat = (y_prob_test + t * Y_test_hat) / (t + 1) 
            pred_time_val = time.time() - start
            times_predict_val.append(pred_time_val)

            memory_now = psutil.Process().memory_info().rss / (1024 * 1024) - memory_init
            history.append((
                int(np.round(10**6 * train_time)),
                int(np.round(10**6 * pred_time)),
                int(np.round(10**6 * pred_time_val)),
                np.round(y_prob_tree, 4),
                np.round(y_prob_test, 4),
                np.round(memory_now, 1)
            ))

            brier_score_oob_tree = get_brier_score(Y_train, y_prob_tree)
            brier_score_val_tree = get_brier_score(Y_test, y_prob_test)
            brier_score_oob = get_brier_score(Y_train, Y_train_hat)
            brier_score_val = get_brier_score(Y_test, Y_test_hat)
            #score_history.append(brier_score_oob)
            tree_score_history_oob.append(brier_score_oob_tree)
            tree_score_history_val.append(brier_score_val_tree)
            t += 1
            
            
            
            eval_logger.info(
                f"{t}. "
                f"Current score: {np.round(brier_score_oob, 5)} (OOB) {np.round(brier_score_val, 5)} (VAL). "
                f"Estimated required trees: {required_trees}. "
                f"Memory; {np.round(memory_now, 1)}MB."
            )

            if memory_now > 900 * 1024:
                eval_logger.info("Approaching memory limit. Stopping!")
                break
        
        required_trees = get_required_num_trees(prob_history_val, tree_score_history_val, eps=eps, certainty_factor=zfactor)
        eval_logger.info(f"New estimate for required number of trees: {required_trees}. Currently have info for {t}.")
        if t >= required_trees:
            break
        step_size = np.min([10**4, required_trees - t])

    #oob_history = [e.tolist() for e in prob_history_oob]
    #val_history = [e.tolist() for e in prob_history_oob]

    oob_history_as_string = str([e.tolist() for e in prob_history_oob])
    val_history_as_string = str([e.tolist() for e in prob_history_val])

    oob_history_compressed = str(zlib.compress(oob_history_as_string.encode()))
    val_history_compressed = str(zlib.compress(val_history_as_string.encode()))

    eval_logger.info(f"History compressed, now returning.")

    #eval_logger.info(f"History ready. Now returning.")
    return [
        X.tolist(),
        Y_train.tolist(),
        Y_test.tolist(),
        oob_history_compressed,
        val_history_compressed,
        times_train,
        times_predict_train,
        times_predict_val,
        times_update
    ]

def build_full_regression_forest(openmlid, seed, binarize_sparse, drop):
    
    print("Loading dataset")
    X, y = get_dataset(openmlid)
    print("Splitting the data")
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = seed)
    
    print("split created, now building forest")
    
    # determine scaling
    scale = lambda x: (x - np.median(y_train)) / (np.max(y_train) - np.min(y_train))
    
    # check whether we must pre-process
    preprocessing = get_mandatory_preprocessing(X_train, y_train, binarize_sparse, drop)
    print(f"Preprocessing: {'yes' if preprocessing else 'no'}")
    if preprocessing:
        pl = sklearn.pipeline.Pipeline(preprocessing)
        print(f"Modifying inputs with {pl}")
        X_train = pl.fit_transform(X_train, y_train)
        X_test = pl.transform(X_test)
    
    y_test_scaled = scale(y_test)
    def get_mse_score(y_hat):
        return np.mean((scale(y_hat) - y_test_scaled)**2)
    
    rf = asforests.RandomForestRegressor(step_size = 1, prediction_map_for_scoring = scale)
    gen = rf.get_score_generator(X_train, y_train)
    
    start = time.time()
    history = []
    
    memory_init = psutil.Process().memory_info().rss / (1024 * 1024)
    
    y_hat = np.zeros(len(y_test))
    
    t = 1
    while True:
        next_score_train, train_time, pred_time, score_time = next(gen)
        
        # update posterior distribution on validation set and compute score
        start = time.time()
        y_prob_oob_tree = rf.predict_tree(-1, X_test)
        y_hat = (y_prob_oob_tree + t * y_hat) / (t + 1)
        pred_time_val = time.time() - start
        start = time.time()
        next_score_test = get_mse_score(y_hat)
        score_time_val = time.time() - start
        
        memory_now = psutil.Process().memory_info().rss / (1024 * 1024) - memory_init
        history.append((int(np.round(10**6 * train_time)), int(np.round(10**6 * pred_time)), int(np.round(10**6 * score_time)), int(np.round(10**6 * pred_time_val)), int(np.round(10**6 * score_time_val)), np.round(next_score_train, 6), np.round(next_score_test, 6), np.round(memory_now, 1)))
        
        diff = np.nan
        if len(history) > iterations_with_max_diff:
            window = np.array(history[-iterations_with_max_diff:])
            diff = np.round(max(window[:,5]) - min(window[:,5]), 6)
        eval_logger.info(f"{len(history)}. Current score: {np.round(history[-1][5], 6)} (OOB) {np.round(history[-1][6], 6)} (VAL). Max diff in window: {diff}. Memory; {np.round(memory_now, 1)}MB")
        
        if not np.isnan(diff) and diff <= max_diff:
            break
        
        t += 1
    return history