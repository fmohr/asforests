from sklearn.datasets import fetch_openml, make_classification

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone

import itertools as it
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import f_oneway

from tqdm.notebook import tqdm

def get_random_dataset(n, d, k):
    
    X = np.random.rand(n, d)
    y = np.random.randint(0, k, n)
    
    return X, y

def get_one_hot_encoding(y, k=None):
    if k is None:
        k = len(np.unique(y))
    y_onehot = np.zeros((len(y), k))
    for label in range(k):
        mask = y == label
        y_onehot[mask, label] = 1
    return y_onehot

def get_data_setup(n_train = 100, n_test = 10**7, data_seed = 0, n_features=4):
    
    X, y = make_classification(n_samples=n_train + n_test, n_features=n_features, n_informative=n_features, n_redundant=0, random_state=data_seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=data_seed,
        stratify=y,
        train_size=n_train,
        test_size=n_test
    )
    
    rs = np.random.RandomState(0)
    resampled_indices = rs.choice(range(n_test), n_test, replace=False)
    X_test = X_test[resampled_indices]
    y_test = y_test[resampled_indices]
    return X_train, X_test, y_train, y_test


def get_deltas(t=10, n_train=100, n_test=100, random_state_gen=0, random_state_split=0, random_state_clf=0, n_samples=10**4):
    
    if n_train + n_test > n_samples:
        raise ValueError("Not enough samples for problem!")
    
    X, y = make_classification(n_samples=n_samples, n_features=5, n_informative=5, n_redundant=0, random_state=random_state_gen)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=random_state_split,
        stratify=y,
        train_size=n_train,
        test_size=n_test
    )

def get_deltas_from_data(X_train, X_test, y_train, y_test, clf = ExtraTreesClassifier()):
    
    # create learner
    clf.fit(X_train, y_train)
    
    # compute score
    y_pred = np.array([l.predict_proba(X_test) for l in clf])
    y_true = get_one_hot_encoding(y_test, k=2)
    return y_pred - y_true

def get_lambdas(deltas=None, **kwargs):
    if deltas is None:
        deltas = get_deltas(**kwargs)
    return pd.DataFrame(
        [[i, s, sp, (deltas[s,i] * deltas[sp,i]).sum()] for s in range(deltas.shape[0]) for sp in range(s, deltas.shape[0]) for i in range(deltas.shape[1])],
        columns=["i", "s1", "s2", "lambda"]
    )

class LambdaBuilder:
    
    def __init__(self, lambdas=None, **kwargs):
        if lambdas is None:
            lambdas = get_lambdas(**kwargs)
        self.lambdas = lambdas
    
    def with_instances(self, instance_indices):
        return LambdaBuilder(self.lambdas[(self.lambdas["i"].isin(instance_indices))])
        
    def with_members(self, indices):
        return LambdaBuilder(self.lambdas[(self.lambdas["s1"].isin(indices)) & (self.lambdas["s2"].isin(indices))])
    
    def equals(self):
        return LambdaBuilder(lambdas=self.lambdas[self.lambdas["s1"] == self.lambdas["s2"]])
    
    def unequals(self):
        return LambdaBuilder(self.lambdas[self.lambdas["s1"] != self.lambdas["s2"]])
    
    def unique(self, across_instances=False, one_per_instance=False):
        
        if across_instances:
            seen_indices = set()
            included_instances = set()
            mask = []
            
            for row in self.lambdas.values:
                i, s1, s2, s3, s4 = (row[0], row[1], row[2], row[4], row[5]) if "s3" in self.lambdas else (row[0], row[1], row[2], None, None)
                forbidden = s1 in seen_indices or s2 in seen_indices or s3 in seen_indices or s4 in seen_indices or (one_per_instance and i in included_instances)
                mask.append(forbidden)
                if not forbidden:
                    seen_indices.add(s1)
                    seen_indices.add(s2)
                    if s3 is not None:
                        seen_indices.add(s3)
                        seen_indices.add(s4)
                    included_instances.add(i)
            mask = np.array(mask)
            return LambdaBuilder(self.lambdas[~mask])
        
        else:
            dfs = []
            for i, df_i in self.lambdas.groupby("i"):
                seen_indices = set()
                mask = []
                for s1, s2 in df_i[["s1", "s2"]].values:
                    forbidden = s1 in seen_indices or s2 in seen_indices
                    mask.append(forbidden)
                    if not forbidden:
                        seen_indices.add(s1)
                        seen_indices.add(s2)
                mask = np.array(mask)
                dfs.append(df_i[~mask])
            return LambdaBuilder(pd.concat(dfs))
    
    def copy(self):
        return LambdaBuilder(self.lambdas)
    
    def reduce(self, reduce_fun):
        return LambdaBuilder(self.lambdas[reduce_fun(self.lambdas)])
    
    def combine(self, builder, on=None, lambda_filter=None, require_right_is_not_smaller=True, only_one_per_instance=False):
        df_y = builder.lambdas.rename(columns={"s1": "s3", "s2": "s4", "lambda": "lambda2"})
        if on is None:
            df_y = df_y.rename(columns={"i": "ip"})
            df = self.lambdas.rename(columns={"lambda": "lambda1"}).merge(df_y, how="cross")
        else:
            df = self.lambdas.rename(columns={"lambda": "lambda1"}).merge(df_y, on=on)
        if require_right_is_not_smaller:
            df = df[df["s3"] >= df["s2"]]
        lb = LambdaBuilder(df)
        if lambda_filter is not None:
            return lb.reduce(lambda_filter)
        return lb
    
    def cov(self, **kwargs):
        if "s3" not in self.lambdas.columns:
            raise Exception()
        return self.lambdas[["lambda1", "lambda2"]].cov(**kwargs).values[0, 1]