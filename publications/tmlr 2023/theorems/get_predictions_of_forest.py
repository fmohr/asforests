from utils import *
from scipy.stats import norm
import pathlib

from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from tqdm import tqdm

import sys

if __name__ == "__main__":

    seed = int(sys.argv[1])

    filename = f"theorem2_data/predictions_{seed}.csv"

    if not pathlib.Path(filename).exists():
        
    
        kwargs = {
            "max_depth": 1
        }

        print("Generating data")
        X_train, X_test, y_train, y_test = get_data_setup()
        Y_test = get_one_hot_encoding(y_test)
    
        clf = ExtraTreesClassifier(n_estimators=0, random_state=seed, warm_start=True, **kwargs)

        print("Ready. Now starting training")
        for i in tqdm(range(100)):
            clf.n_estimators += 10**3
            clf.fit(X_train, y_train)
        print(f"Finished training a forest with {len(clf)} trees")
        print("Now getting predictions")
        predictions = clf.predict_proba(X_test[:10**5])
        print("Writing result file.")
        pd.DataFrame(predictions, columns=["p1", "p2"]).to_csv(filename)
    
    else:
        print(f"File {filename} exists already. Skipping.")
