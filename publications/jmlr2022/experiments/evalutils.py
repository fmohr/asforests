import numpy as np
import pandas as pd
import openml
import os, psutil
import gc
import logging

from func_timeout import func_timeout, FunctionTimedOut

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

from func_timeout import func_timeout, FunctionTimedOut
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

def get_mandatory_preprocessing(X, y, binarize_sparse = False, drop = 'first'):
    
    # determine fixed pre-processing steps for imputation and binarization
    types = [set([type(v) for v in r]) for r in X.T]
    numeric_features = [c for c, t in enumerate(types) if len(t) == 1 and list(t)[0] != str]
    numeric_transformer = Pipeline([("imputer", sklearn.impute.SimpleImputer(strategy="median"))])
    categorical_features = [i for i in range(X.shape[1]) if i not in numeric_features]
    missing_values_per_feature = np.sum(pd.isnull(X), axis=0)
    eval_logger.info(f"There are {len(categorical_features)} categorical features, which will be binarized.")
    eval_logger.info(f"Missing values for the different attributes are {missing_values_per_feature}.")
    if len(categorical_features) > 0 or sum(missing_values_per_feature) > 0:
        categorical_transformer = Pipeline([
            ("imputer", sklearn.impute.SimpleImputer(strategy="most_frequent")),
            ("binarizer", sklearn.preprocessing.OneHotEncoder(drop=drop, sparse = binarize_sparse, handle_unknown = 'error' if drop == 'first' else 'ignore')),
        ])
        return [("impute_and_binarize", ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        ))]
    else:
        return []

def build_full_classification_forest(openmlid, seed, max_diff, iterations_with_max_diff, binarize_sparse, drop):
    
    print("Loading dataset")
    X, y = get_dataset(openmlid)
    print("Splitting the data")
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = seed)
    
    print("split created, now building forest")
    
    # check whether we must pre-process
    preprocessing = get_mandatory_preprocessing(X_train, y_train, binarize_sparse, drop)
    if preprocessing:
        pl = sklearn.pipeline.Pipeline(preprocessing)
        print(f"Modifying inputs with {pl}")
        X_train = pl.fit_transform(X_train, y_train)
        X_test = pl.transform(X_test)
    
    # prepare everything to efficiently compute the brier score
    labels = list(np.unique(y_test))
    n, k = len(y_test), len(labels)
    Y = np.zeros((n, k))
    for i, true_label in enumerate(y_test):
        Y[i,labels.index(true_label)] = 1
    def get_brier_score(Y_prob):
        return np.mean(np.sum((Y_prob - Y)**2, axis=1))
    
    rf = asforests.RandomForestClassifier(step_size = 1)
    gen = rf.get_score_generator(X_train, y_train)
    
    start = time.time()
    history = []
    
    memory_init = psutil.Process().memory_info().rss / (1024 * 1024)
    
    Y_test_hat = np.zeros(Y.shape)
    
    t = 1
    while True:
        t_0 = time.time()
        next_score_train = np.round(next(gen), 4)
        t_1 = time.time()
        
        # update posterior distribution on test set
        y_prob_oob_tree = rf.predict_tree_proba(-1, X_test)

        # update forest's prediction
        Y_test_hat = (y_prob_oob_tree + t * Y_test_hat) / (t + 1) 
        
        next_score_test = np.round(get_brier_score(Y_test_hat), 4)
        t_2 = time.time()
        memory_now = psutil.Process().memory_info().rss / (1024 * 1024) - memory_init
        t_3 = time.time()
        history.append((int(np.round(1000 * (t_0 - start))), int(np.round(1000 * (t_1 - t_0))), int(np.round(1000 * (t_2 - t_1))), next_score_train, next_score_test, np.round(memory_now, 1)))
        
        diff = np.nan
        if len(history) > iterations_with_max_diff:
            window = np.array(history[-iterations_with_max_diff:])
            diff = np.round(max(window[:,3]) - min(window[:,3]), 5)
            if diff <= max_diff:
                break
        eval_logger.info(f"{len(history)}. Current score: {np.round(history[-1][3], 4)}. Max diff in window: {diff}. Memory; {np.round(memory_now, 1)}MB")
        
        t += 1
    return history

def build_full_regression_forest(openmlid, seed, max_diff, iterations_with_max_diff, binarize_sparse, drop):
    
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
        #return np.mean((y_hat - y_test)**2)
    
    rf = asforests.RandomForestRegressor(step_size = 1, prediction_map_for_scoring = scale)
    gen = rf.get_score_generator(X_train, y_train)
    
    start = time.time()
    history = []
    
    memory_init = psutil.Process().memory_info().rss / (1024 * 1024)
    
    y_hat = np.zeros(len(y_test))
    
    t = 1
    while True:
        t_0 = time.time()
        next_score_train = np.round(next(gen), 6)
        t_1 = time.time()
        
        # update posterior distribution on test set
        y_prob_oob_tree = rf.predict_tree(-1, X_test)

        # update forest's prediction
        y_hat = (y_prob_oob_tree + t * y_hat) / (t + 1) 
        
        next_score_test = np.round(get_mse_score(y_hat), 6)
        t_2 = time.time()
        memory_now = psutil.Process().memory_info().rss / (1024 * 1024) - memory_init
        t_3 = time.time()
        history.append((int(np.round(10**6 * (t_0 - start))), int(np.round(10**6 * (t_1 - t_0))), int(np.round(10**6 * (t_2 - t_1))), next_score_train, next_score_test, np.round(memory_now, 1)))
        
        diff = np.nan
        if len(history) > iterations_with_max_diff:
            window = np.array(history[-iterations_with_max_diff:])
            diff = np.round(max(window[:,3]) - min(window[:,3]), 6)
        eval_logger.info(f"{len(history)}. Current score: {np.round(history[-1][3], 6)} (train) {np.round(history[-1][4], 6)} (test). Max diff in window: {diff}. Memory; {np.round(memory_now, 1)}MB")
        
        if not np.isnan(diff) and diff <= max_diff:
            break
        
        t += 1
    return history