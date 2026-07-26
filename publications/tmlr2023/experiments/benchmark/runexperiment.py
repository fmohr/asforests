from experiments.problem_instance.problem_instance import ProblemInstance
from benchmark import Benchmark
from approaches import DatabaseWiseApproach, BootstrappingApproach, ParametricModelApproach, ParametricDifferenceModelApproach
from approaches.approach import Approach
from typing import Dict


from pprint import pprint
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
import os


def run_experiment(keyfields: dict, result_processor, custom_config):

    # define stream handler
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)

    # configure logger for experiment runner
    logger = logging.getLogger("experimenter")
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

    # configure logger of approach
    a_logger = logging.getLogger("approach")
    a_logger.handlers.clear()
    a_logger.addHandler(ch)
    a_logger.setLevel(logging.WARNING)

    # configure logger of benchmark
    bm_logger = logging.getLogger("benchmark")
    bm_logger.handlers.clear()
    bm_logger.addHandler(ch)
    bm_logger.setLevel(logging.WARNING)

    info_str = ""
    for k, v in keyfields.items():
        info_str += f"\n\t{k}: {v}"
    logger.info(f"Starting experiment with specification{info_str}")


    # make sure that we have the results folder
    folder = f"results"
    pathlib.Path(folder).mkdir(exist_ok=True, parents=True)

    # read problem instance with known ground truth parameter values
    openmlid = int(keyfields["openmlid"])
    data_seed = int(keyfields["data_seed"])
    num_possible_ensemble_members = int(keyfields["num_possible_ensemble_members"])
    validation_size = int(keyfields["validation_instances"])
    pi = ProblemInstance.load_from_instance_file(
        openmlid=openmlid,
        data_seed=data_seed,
        num_possible_ensemble_members=num_possible_ensemble_members,
        validation_instances=validation_size
    )
    assert pi is not None, f"no problem instance file found for {keyfields}"
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
    logger.info(f"Successfully loaded problem instance {pi}")
    pi.logger = logger

    # configure ensemble sequence seed
    ensemble_sequence_seed = int(keyfields["ensemble_sequence_seed"])

    # get approach object
    if "method" not in keyfields:
        raise ValueError("No 'method' field set, not clear which method to evaluate.")
    method_name = keyfields["method"]

    # create benchmark
    captured_parameters = ["E[Z_nt]", "E[Z_nt|D_val]", "V[Z_nt]", "V[Z_nt|D_val]"]
    #captured_parameters = ["V[Z_nt]", "V[Z_nt|D_val]"]
    for captured_parameter in captured_parameters:

        # get approach object
        if method_name.startswith("parametric"):
            _, exp_for_threshold_number_of_datapoints = method_name.split("-")
            num_simulated_ensembles = int(10**int(exp_for_threshold_number_of_datapoints))
            approach = ParametricDifferenceModelApproach(
                random_state=0,
                estimated_parameters=[captured_parameter],
                num_simulated_ensembles=num_simulated_ensembles,
                logger=a_logger
            )
        elif method_name.startswith("bootstrapping"):
            _, num_resamples, bootstrap_size = method_name.split("-")
            num_resamples = int(num_resamples)
            bootstrap_size = int(bootstrap_size)
            approach = BootstrappingApproach(
                random_state=0,
                estimated_parameters=[captured_parameter],
                bootstrap_size=bootstrap_size,
                num_resamples=num_resamples,
                logger=a_logger
            )
            assert approach.bootstrap_size == bootstrap_size, f"Bootstrap size not correctly copied. Should be {bootstrap_size} but is {approach.bootstrap_size}"
            assert approach.num_resamples == num_resamples, f"Number of resamples not correctly copied. Should be {num_resamples} but is {approach.num_resamples}"
        elif method_name.startswith("direct"):
            _, exp_for_threshold_for_number_of_samples_to_exclude_param = method_name.split("-")
            exp_for_threshold_for_number_of_samples_to_exclude_param = int(exp_for_threshold_for_number_of_samples_to_exclude_param)
            
            approach = DatabaseWiseApproach(
                random_state=0,
                estimated_parameters=[captured_parameter],
                population_mode="stream",
                threshold_for_number_of_samples_to_exclude_param=10**exp_for_threshold_for_number_of_samples_to_exclude_param,
                logger=a_logger
            )
        else:
            raise ValueError(f"Unknown method {method_name}")
        approaches = {
            f"{captured_parameter}::{method_name}": approach
        }

        t_checkpoints = pi.t_checkpoints
        n_checkpoints = pi.n_checkpoints if captured_parameter == "V[Z_nt]" else [None]

        folder_for_task = f"{folder}/{openmlid}/{captured_parameter}"
        pathlib.Path(folder_for_task).mkdir(exist_ok=True, parents=True)

        for n, t in it.product(n_checkpoints, t_checkpoints):
            n = int(n) if n is not None else None
            t = int(t)

            # define name for result file and skip if we already have results for this
            filename = f"{folder_for_task}/{method_name}_{data_seed}_{ensemble_sequence_seed}_{num_possible_ensemble_members}_{validation_size}_{n}_{t}.json"
            gz_filename = f"{filename}.gz"
            if pathlib.Path(gz_filename).exists():
                logger.info(f"Skipping {method_name} on {openmlid} with seed {ensemble_sequence_seed} and n/t combo {n}/{t} since result file {gz_filename} already exists.")
                continue
            logger.info(f"Starting evaluation of {captured_parameter}-estimates of {method_name} on {openmlid} with data seed {data_seed} and ensemble sequence seed {ensemble_sequence_seed} for {n=}, {t=}, {num_possible_ensemble_members} possible ensemble members, and {validation_size} validation instances.")

            # create copy of the problem instance only for this case
            pi_nt = pi.copy()
            for _t in pi.t_checkpoints:
                if _t != t:
                    pi_nt.drop_t_checkpoint(_t)
            if n is not None:
                for _n in pi.n_checkpoints:
                    if _n != n:
                        pi_nt.drop_n_checkpoint(_n)
            else:
                for _n in pi.n_checkpoints:
                    pi_nt.drop_n_checkpoint(_n)
            
            assert len(pi_nt.t_checkpoints) == 1
            if captured_parameter == "V[Z_nt]":
                assert len(pi_nt.n_checkpoints) == 1
            else:
                assert len(pi_nt.n_checkpoints) == 0

            # now create a benchmark for this case
            b = Benchmark(
                problem_instance=pi_nt,
                captured_parameter=captured_parameter,
                ensemble_sequence_seed=ensemble_sequence_seed
            )

            # run benchmark 
            logger.info(f"Resetting benchmark, including extraction of ground truth.")
            b.reset(approaches)
            max_budget = 10**2
            logger.info(f"Done. Now obtaining estimates for budgets b from 1 to {max_budget}")

            for _ in tqdm(range(max_budget)):
                b.step()
            
            logger.info(f"Done, writing results to {gz_filename}.")
            with gzip.open(gz_filename, "wt", encoding="utf-8") as f:
                f.write(b.result_storage.serialize())
    
    logger.info(f"Finished experiment with specification{info_str}")

if __name__ == "__main__":

    if False:
        run_experiment({
            "openmlid": 1049,
            "num_possible_ensemble_members": 64,
            "validation_instances": 64,
            "data_seed": 0,
            "ensemble_sequence_seed": 1,
            "method": "bootstrapping-100-100"
        }, None, None)
        exit(0)

    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        sleep_time = 3 * rank
        print(f"Sleeping for {sleep_time}s to avoid potential issues with multiple jobs starting at the same time.")
        time.sleep(sleep_time)
    else:
        print("Not running in a SLURM environment, so not sleeping.")


    pe = PyExperimenter(
        name="ensembles",
        use_codecarbon=False,
        experiment_configuration_file_path=f"config/experiments_full.yaml"
        )

    while True:
        try:
            pe.execute(max_experiments=-1, experiment_function=run_experiment)
            break
        except Exception as e:
            print("Observed a problem. Waiting 5 seconds and re-running the script.")
            print(e)
            time.sleep(5)