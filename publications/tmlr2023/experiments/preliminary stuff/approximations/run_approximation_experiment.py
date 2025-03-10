import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from tqdm import tqdm
from asforests import EnsemblePerformanceAssessor

import itertools as it

import matplotlib.pyplot as plt

import sys, os

# load utils from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evalutils import get_splitted_data

if __name__ == "__main__":

    # prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = get_splitted_data(
        openmlid=61,
        seed_application=0,
        seed_training=0,
        application_size=0.2,
        validation_size=0.2,
        is_classification=True
    )
    y_val_oh = None

    rs = np.random.RandomState(0)
    forest = ExtraTreesClassifier(n_estimators=0, warm_start=True, random_state=rs)

    ass = EnsemblePerformanceAssessor(
        upper_bound_for_sample_size=10**2,
        population_mode="resample_with_replacement"
    )
    performance_curve_mean = ass.expected_performance

    hist_mean = []
    hist_var = []
    hist_curve_beliefs = []

    domain_t = list(range(1, 11))
    domain_forecast = np.arange(1, 101)
    for t in tqdm(domain_t):

        # train new trees
        forest.set_params(n_estimators=t)
        forest.fit(X_train, y_train)
        if y_val_oh is None:
            labels = list(forest.classes_)
            y_val_oh = np.zeros((len(y_val), len(labels)))
            for i, label in enumerate(y_val):
                y_val_oh[i, labels.index(label)] = 1

        # update ensemble monitor
        ass.add_deviation_matrix(forest[-1].predict_proba(X_val) - y_val_oh)

        # retrieve updated beliefs
        hist_mean.append(ass.gap_mean_point.copy())
        hist_var.append(ass.gap_var_point.copy())
        hist_curve_beliefs.append(performance_curve_mean(domain_forecast).copy())

    # manually compute the true covariance
    plt.plot(np.array(hist_curve_beliefs[-5:]).T)
    plt.show()
