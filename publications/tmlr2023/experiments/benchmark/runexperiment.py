from experiments.problem_instance.problem_instance import ProblemInstance
from benchmark import Benchmark
from approaches import DatabaseWiseApproach, BootstrappingApproach, ParametricModelApproach, ParametricDifferenceModelApproach
from approaches.approach import Approach
from typing import Dict


import matplotlib.pyplot as plt
import logging

from tqdm import tqdm
from py_experimenter.experimenter import PyExperimenter
import gzip

import matplotlib.gridspec as gridspec
from numpy.lib.stride_tricks import sliding_window_view

import json
import pathlib
import sys
import time
import numpy as np
import itertools as it


ACCEPTED_APPROACHES = ["bootstrapping", "databaseperparameter", "parametricmodel"]

def run_experiment(keyfields: dict, result_processor, custom_config):

    # define stream handler
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    # configure logger for experiment runner
    logger = logging.getLogger("experimenter")
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

    folder = f"results/"
    pathlib.Path(folder).mkdir(exist_ok=True, parents=True)

    openmlid = int(keyfields["openmlid"])
    data_seed = int(keyfields["data_seed"])
    t = int(keyfields["t"])
    n = int(keyfields["n"])

    num_possible_ensemble_members = int(keyfields["num_possible_ensemble_members"])
    validation_size = int(keyfields["validation_instances"])
    
    training_instances_per_class = 10

    # read problem instance with known ground truth parameter values
    with open(f"{PATH_TO_PROBLEM_INSTANCES}/{openmlid}_{data_seed}_{num_possible_ensemble_members}_{validation_size}.json", 'r') as f:
        pi = ProblemInstance.from_dict(json.load(f))
        for _t in pi.t_checkpoints:
            if _t != t:
                pi.drop_t_checkpoint(t=_t)
        for _n in pi.n_checkpoints:
            if _n != n:
                pi.drop_n_checkpoint(n=_n)
        assert pi._true_means_for_iid_case is not None
        if pi._true_means_for_iid_case.shape == (len(pi.n_checkpoints), len(pi.t_checkpoints)):
            pi._true_means_for_iid_case = pi._true_means_for_iid_case.mean(axis=0)
        assert pi._true_means_for_iid_case.shape == (len(pi.t_checkpoints), )
        assert pi._true_vars_for_iid_case is not None
        assert pi._true_vars_for_iid_case.shape == (len(pi.n_checkpoints), len(pi.t_checkpoints))
        assert pi._true_means_for_cond_case is not None
        assert pi._true_means_for_cond_case.shape == (len(pi.t_checkpoints), )
        assert pi._true_vars_for_cond_case is not None
        assert pi._true_vars_for_cond_case.shape == (len(pi.t_checkpoints), )
        #assert len(pi._indices_val) == validation_size, f"There should be {validation_size} validation instances, but there are {len(pi._indices_val)}"

    for ensemble_sequence_seed in range(5):

        # define name for result file and skip if we already have results for this
        filename = f"{folder}/{openmlid}_{data_seed}_{ensemble_sequence_seed}_{num_possible_ensemble_members}_{training_instances_per_class}_{validation_size}_{n}_{t}.json"
        if pathlib.Path(filename).exists():
            print(f"Skipping seed {ensemble_sequence_seed} since result file already exists.")
            continue
        

        # create benchmark
        captured_parameters = ["E[Z_nt]", "E[Z_nt|D_val]", "V[Z_nt]", "V[Z_nt|D_val]"]
        b = Benchmark(
            problem_instance=pi,
            captured_parameters=captured_parameters,
            ensemble_sequence_seed=ensemble_sequence_seed
        )

        # configure logger of approach
        a_logger = logging.getLogger("approach")
        a_logger.handlers.clear()
        a_logger.addHandler(ch)
        a_logger.setLevel(logging.DEBUG)
        approaches = {}
        for captured_parameter in captured_parameters:
            for num_simulated_ensembles in [100, 1000]:
                approaches[f"{captured_parameter}::biparametric - {num_simulated_ensembles}"] = ParametricDifferenceModelApproach(
                    random_state=0,
                    estimated_parameters=[captured_parameter],
                    num_simulated_ensembles=num_simulated_ensembles,
                    logger=a_logger
                )
            
            for num_resamples, bootstrap_size in it.product([1, 10], [10]):#, 100]):
                approaches[f"{captured_parameter}::bootstrapping - {num_resamples}x{bootstrap_size}"] = BootstrappingApproach(
                    random_state=0,
                    estimated_parameters=[captured_parameter],
                    bootstrap_size=bootstrap_size,
                    num_resamples=num_resamples,
                    logger=a_logger
                )
        
            approaches[f"{captured_parameter}::model free"] = DatabaseWiseApproach(
                random_state=0,
                estimated_parameters=[captured_parameter],
                population_mode="stream",
                threshold_for_number_of_samples_to_exclude_param=10**6,
                logger=a_logger
            )

        # configure logger of benchmark
        bm_logger = logging.getLogger("benchmark")
        bm_logger.handlers.clear()
        bm_logger.addHandler(ch)
        bm_logger.setLevel(logging.DEBUG)
        
        # run benchmark for 10 iterations (10 ensemble members)
        logger.info(f"Running experiment on dataset {openmlid} with data seed {data_seed} for {n=} and {t=}, ensemble sequence seed {ensemble_sequence_seed}, {validation_size} validation instances, and {num_possible_ensemble_members} possible ensemble members.")
        logger.info(f"Computing ground truth")
        b.reset(approaches)
        
        max_budget = 10**3
        logger.info(f"Done. Now obtaining estimates for ensemble sizes of size up to {max_budget}")
        for _ in tqdm(range(max_budget)):
            b.step()
        
        logger.info(f"Done, writing results to {filename}.")
        with open(filename, "w") as f:
            b.result_storage.serialize(f)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        raise ValueError(f"Please specify exactly two argument (the path to the problem instances with pre-computed ground truth values and the job name).")
    PATH_TO_PROBLEM_INSTANCES = pathlib.Path(sys.argv[1])
    name = sys.argv[2]

    if not PATH_TO_PROBLEM_INSTANCES.exists():
        raise ValueError(f"The path to the problem instances {PATH_TO_PROBLEM_INSTANCES} does not exist.")


    #run_experiment({
    #    "openmlid": 1049,
    #    "num_possible_ensemble_members": 8,
    #    "validation_instances": 8,
    #    "data_seed": 1,
    #    "t": 1000,
    #    "n": 1000
    #}, None, None)
    #exit(0)

    sleep_time = np.random.rand() * 3
    print(f"Sleeping {sleep_time}s")
    time.sleep(sleep_time)

    pe = PyExperimenter(
        name=name,
        use_codecarbon=False,
        experiment_configuration_file_path=f"config/experiments.yaml"
        )

    while True:
        try:
            pe.execute(max_experiments=-1, experiment_function=run_experiment)
            break
        except Exception as e:
            print("Observed a problem. Waiting 5 seconds and re-running the script.")
            print(e)
            time.sleep(5)