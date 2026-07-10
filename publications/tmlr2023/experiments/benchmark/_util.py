from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import json
import numpy as np


def get_unique_prediction_matrices(X, y, train_indices, num_matrices, seed, max_tries=10, logger=None, rf_kwargs={}, cache_files=None):
    """
        Determines `num_matrices` pairwise different prediction matrices on the given data, where only the indices in `train_indices` are used for training.
        It returns a list of such matrices and a list of the classes assigned to each column in the matrices

    Args:
        X (_type_): _description_
        y (_type_): _description_
        train_indices (_type_): _description_
        num_matrices (_type_): _description_
        seed (_type_): _description_
        max_tries (int, optional): _description_. Defaults to 10.
        logger (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    failed_tries = 0
    classes = None
    prediction_matrices = []
    #prediction_matrices_as_str = set()
    n_estimators = max(10, num_matrices)

    # make folder for prediction matrices if it does not exist yet
    if cache_files is not None:
        pm_path = Path(cache_files[0])
        class_path = Path(cache_files[1])
        pm_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_files is None or not pm_path.exists() or not class_path.exists():
        if logger is not None and cache_files is not None:
            logger.info(f"One of the cache files {pm_path} and {class_path} do not exist yet, creating them by training RFs until we have {num_matrices} different prediction matrices.")
        while len(prediction_matrices) < num_matrices and failed_tries < max_tries:

            if logger is not None:
                logger.info(f"Training RF with {n_estimators} trees.")
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=seed + len(prediction_matrices),
                **rf_kwargs
                ).fit(X[train_indices], y[train_indices])
            ensemble_members = list(rf)
            _classes = [int(c) if type(c) == np.int64 else c for c in rf.classes_]
            if logger is not None:
                logger.info(f"RF trained, classes are {_classes}.")
            if classes is not None:
                assert classes == _classes
            else:
                classes = _classes
            
            # update deviation metrices
            if logger is not None:
                logger.info(f"Generating prediction matrices for {X.shape[0]} instances.")
            size_before = len(prediction_matrices)
            for prediction_matrix in [t.predict_proba(X) for t in ensemble_members]:
                #pm_as_str = str(prediction_matrix)
                if not np.any([np.allclose(m, prediction_matrix) for m in prediction_matrices]):
                    prediction_matrices.append(prediction_matrix)
                    if len(prediction_matrices) == num_matrices:
                        break

            size_after = len(prediction_matrices)
            if size_before == size_after:
                failed_tries += 1
                if logger is not None:
                    logger.info(f"Prediction matrix set stalled at size {len(prediction_matrices)}, trying new batch (# failed tries is now {failed_tries})")
            else:
                if logger is not None:
                    logger.info(f"Prediction matrix set has now size {len(prediction_matrices)}.")
                failed_tries = 0

        if len(prediction_matrices) < num_matrices:
            raise RuntimeError(f"Could not create {num_matrices} different ensemble members but just {len(prediction_matrices)}")
        
        prediction_matrices = np.array(prediction_matrices[:num_matrices])

        # write files
        if cache_files is not None:
            if logger is not None:
                logger.info(f"Writing prediction matrices of shape {prediction_matrices.shape} and classes to {pm_path} and {class_path}.")
            np.save(pm_path, prediction_matrices)
            with open(class_path, "w") as f:
                f.write(json.dumps(classes))
    
    else:
        if logger is not None:
            logger.info(f"Cache files {pm_path} and {class_path} already exist, loading prediction matrices and classes from them.")
        prediction_matrices = np.load(pm_path)
        with open(class_path, "r") as f:
            classes = json.load(f)

    return prediction_matrices, classes

