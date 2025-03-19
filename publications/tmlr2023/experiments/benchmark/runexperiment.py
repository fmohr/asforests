from benchmark import Benchmark
from approaches import DatabaseWiseApproach, BootstrappingApproach, ParametricModelApproach

import logging

from tqdm import tqdm
from py_experimenter.experimenter import PyExperimenter

import pathlib
import sys
import time
import numpy as np


ACCEPTED_APPROACHES = ["bootstrapping", "databaseperparameter", "parametricmodel"]

def run_experiment(keyfields: dict, result_processor, custom_config):
    openmlid = int(keyfields["openmlid"])
    data_seed = 0
    ensemble_sequence_seed = int(keyfields["ensemble_sequence_seed"])
    num_possible_ensemble_members = int(keyfields["num_possible_ensemble_members"])
    
    training_instances_per_class = 10
    validation_size = 50

    b = Benchmark(
        openmlid=openmlid,
        data_seed=data_seed,
        ensemble_seed=data_seed,
        ensemble_sequence_seed=ensemble_sequence_seed,
        num_possible_ensemble_members=num_possible_ensemble_members,
        training_instances_per_class=training_instances_per_class,
        validation_size=validation_size,
        is_classification=True
    )

    # get generator for the estimates of the approach on the given problem
    t_checkpoints = [10, 100, 1000]#, 10000]

    approaches = {
        "bootstrapping 1": BootstrappingApproach(num_resamples=1),
        "bootstrapping 10": BootstrappingApproach(num_resamples=10),
        "bootstrapping 100": BootstrappingApproach(num_resamples=100),
        "parametric 1": ParametricModelApproach(num_simulated_ensembles=1),
        "parametric 2": ParametricModelApproach(num_simulated_ensembles=2),
        "parametric 4": ParametricModelApproach(num_simulated_ensembles=4),
        "parametric 16": ParametricModelApproach(num_simulated_ensembles=16),
        "parametric 64": ParametricModelApproach(num_simulated_ensembles=64),
        "model free - stream": DatabaseWiseApproach(population_mode="stream"),
        "model free - resample_no_replacement": DatabaseWiseApproach(population_mode="resample_no_replacement"),
        "model free - resample_with_replacement": DatabaseWiseApproach(population_mode="resample_with_replacement")
    }
    
    # define stream handler
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # configure logger
    log_level = logging.WARN
    logger = logging.getLogger("benchmark")
    logger.handlers.clear()
    logger.addHandler(ch)
    ch.setLevel(log_level)
    logger.setLevel(log_level)
    
    # run benchmark for 10 iterations (10 ensemble members)
    print(f"Running experiment on dataset {openmlid} with seeds {data_seed}/{ensemble_sequence_seed}")
    b.reset(approaches, t_checkpoints=t_checkpoints)
    for _ in tqdm(range(10**3)):
        b.step()
    
    folder = f"results/"
    pathlib.Path(folder).mkdir(exist_ok=True, parents=True)
    with open(f"{openmlid}_{data_seed}_{ensemble_sequence_seed}.json", "w") as f:
        b.result_storage.serialize(f)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError(f"Please specify exactly one argument (the job name).")
    name = sys.argv[1]
    sleep_time = np.random.rand() * 30
    print(f"Sleeping {sleep_time}s")
    time.sleep(sleep_time)

    pe = PyExperimenter(
        name=name,
        use_codecarbon=False,
        experiment_configuration_file_path=f"config/experiments.yaml"
        )
    pe.execute(max_experiments=-1, experiment_function=run_experiment)