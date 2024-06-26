import time
import sklearn.ensemble
import logging
from ._grower import EnsembleGrower
from .supplier_ import ForestOutputSupplier

class RandomForestRegressor(sklearn.ensemble.RandomForestRegressor):

    def __init__(self,
                 step_size=5,
                 tolerances=(0.01, 0.95),
                 min_trees=4,
                 max_trees=None,
                 normalize_outputs=True,
                 **kwargs):
        if "n_estimators" in kwargs:
            raise ValueError(
                "This is an automatically stopping RF, which does not accept the n_estimators parameter."
                "Please use max_trees instead if you want to limit the number."
            )
        if "warm_start" in kwargs:
            raise ValueError(
                "This is an automatically stopping RF, which does not accept the warm_start parameter."
                "Warm-starting is automatically enabled.")
        super().__init__(n_estimators=0, warm_start=True, **kwargs)

        if "n_jobs" in kwargs and kwargs["n_jobs"] > step_size:
            raise ValueError(f"The number of jobs cannot be bigger than step_size.")

        self.normalize_outputs = normalize_outputs
        self.step_size = step_size
        self.tolerances = tolerances if isinstance(tolerances, list) else [tolerances]
        self.min_trees = min_trees
        self.max_trees = max_trees
        self.logger = logging.getLogger("ASRFClassifier")

        # state variables
        self.fittime_net = 0
        self.fittime_overhead = 0

    def __str__(self):
        return "ASRFClassifier"

    def fit_(self, X, y):
        super(RandomForestRegressor, self).fit(X, y)

    def fit(self, X, y):

        start = time.time()
        supplier = ForestOutputSupplier(
            task_type="regression",
            forest=self,
            X=X,
            y=y,
            normalize_outputs=self.normalize_outputs,
            min_step_size=8,
            validation_size=0
        )
        n, k = supplier.shape
        grower = EnsembleGrower(
            output_supplier=supplier,
            n=n,
            k=k,
            tolerances=self.tolerances
        )
        grower.grow()
        runtime = time.time() - start
        self.fittime_net = sum(supplier.fit_times)
        self.fittime_overhead = runtime - self.fittime_net
