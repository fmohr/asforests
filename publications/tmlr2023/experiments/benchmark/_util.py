from sklearn.ensemble import RandomForestClassifier
import numpy as np

def get_unique_prediction_matrices(X, y, train_indices, num_matrices, seed, max_tries=10, logger=None, rf_kwargs={}):
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
    while len(prediction_matrices) < num_matrices and failed_tries < max_tries:

        if logger is not None:
            logger.info(f"Training RF with {n_estimators} trees.")
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed + len(prediction_matrices),
            **rf_kwargs
            ).fit(X[train_indices], y[train_indices])
        ensemble_members = list(rf)
        _classes = list(rf.classes_)
        if classes is not None:
            assert classes == _classes
        else:
            classes = _classes
        
        # update deviation metrices
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

    return np.array(prediction_matrices[:num_matrices]), classes

