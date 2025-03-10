from benchmark import Benchmark
from approaches.a_fromdatabase import DatabaseWoseApproach
from approaches.a_bootstrapping import BootstrappingApproach
from approaches.a_parametric import ParametricModelApproach

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    b = Benchmark(
        openmlid=61,
        data_seed=0,
        ensemble_seed=0,
        validation_size=20,
        application_size=50,
        is_classification=True
    )

    # get generator for the estimates of the approach on the given problem
    t_checkpoints = [10, 100, 1000]#, 10000]
    approaches = {
        "bootstrapping": BootstrappingApproach(num_resamples=10**2),
        "theorem with datasets": DatabaseWoseApproach(upper_bound_for_sample_size=10**10),
        "parametric model": ParametricModelApproach(rs=np.random.RandomState(0), num_simulated_ensembles=100)
    }

    # run benchmark for 10 iterations (10 ensemble members)
    b.reset(approaches, t_checkpoints=t_checkpoints)
    for _ in tqdm(range(10**2)):
        b.step()
    
    with open("results/data.json", "w") as f:
        b.result_storage.serialize(f)