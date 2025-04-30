from benchmark import Benchmark
from approaches import DatabaseWiseApproach, BootstrappingApproach, ParametricModelApproach, ParametricDifferenceModelApproach

import logging

from tqdm import tqdm
from py_experimenter.experimenter import PyExperimenter

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
    ch.setLevel(logging.DEBUG)

    # configure logger for experiment runner
    logger = logging.getLogger("experimenter")
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

    folder = f"results/"
    pathlib.Path(folder).mkdir(exist_ok=True, parents=True)

    openmlid = int(keyfields["openmlid"])
    data_seed = int(keyfields["data_seed"])

    num_possible_ensemble_members = int(keyfields["num_possible_ensemble_members"])
    validation_size = int(keyfields["validation_size"])
    
    training_instances_per_class = 10

    for ensemble_sequence_seed in range(5):
        filename = f"{folder}/{openmlid}_{data_seed}_{ensemble_sequence_seed}_{num_possible_ensemble_members}_{training_instances_per_class}_{validation_size}.json"
        if pathlib.Path(filename).exists():
            print(f"Skipping seed {ensemble_sequence_seed} since result file already exists.")
            continue
            
        captured_parameters = ["E[Z_nt]", "E[Z_nt|D_val]", "V[Z_nt]", "V[Z_nt|D_val]"]

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=data_seed,
            ensemble_sequence_seed=ensemble_sequence_seed,
            num_possible_ensemble_members=num_possible_ensemble_members,
            training_instances_per_class=training_instances_per_class,
            validation_size=validation_size,
            is_classification=True,
            captured_parameters=captured_parameters,
            max_ground_truth_table_size=10**6
        )

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000, 10000]

        # configure logger of approach
        a_logger = logging.getLogger("approach")
        a_logger.handlers.clear()
        a_logger.addHandler(ch)
        a_logger.setLevel(logging.DEBUG)
        approaches = {}
        for captured_parameter in captured_parameters:
            iid_estimates_required = "|D_val" not in captured_parameter
            for num_simulated_ensembles in [1, 10, 100, 1000]:
                approaches[f"{captured_parameter}::biparametric - {num_simulated_ensembles}"] = ParametricDifferenceModelApproach(
                    estimated_parameters=[captured_parameter],
                    num_simulated_ensembles=num_simulated_ensembles,
                    logger=a_logger
                )
                #approaches[f"triparametric - {num_simulated_ensembles}"] = ParametricModelApproach(num_simulated_ensembles=num_simulated_ensembles)
            
            for num_resamples, bootstrap_size in it.product([1, 10], [10, 100]):
                approaches[f"{captured_parameter}::bootstrapping - {num_resamples}x{bootstrap_size}"] = BootstrappingApproach(
                    estimated_parameters=[captured_parameter],
                    bootstrap_size=bootstrap_size,
                    num_resamples=num_resamples,
                    logger=a_logger
                )

            for single_instance_per_ensemble_member in [False]:
                if not iid_estimates_required and single_instance_per_ensemble_member:
                    continue
                approaches[f"{captured_parameter}::model free - stream - {'1 instance per member' if single_instance_per_ensemble_member else 'full'}"] = DatabaseWiseApproach(
                    estimated_parameters=[captured_parameter],
                    population_mode="stream",
                    single_data_point_per_ensemble_member=single_instance_per_ensemble_member,
                    create_estimates_for_iid_scenario=iid_estimates_required,
                    logger=a_logger
                )
            #"model free - resample_no_replacement": DatabaseWiseApproach(population_mode="resample_no_replacement"),
            #"model free - resample_with_replacement": DatabaseWiseApproach(population_mode="resample_with_replacement")

        # configure logger of benchmark
        bm_logger = logging.getLogger("benchmark")
        bm_logger.handlers.clear()
        bm_logger.addHandler(ch)
        bm_logger.setLevel(logging.INFO)
        
        # run benchmark for 10 iterations (10 ensemble members)
        logger.info(f"Running experiment on dataset {openmlid} with data seed {data_seed}, ensemble sequence seed {ensemble_sequence_seed}, {validation_size} validation instances, and {num_possible_ensemble_members} possible ensemble members.")
        logger.info(f"Computing ground truth")
        b.reset(approaches, t_checkpoints=t_checkpoints)
        
        max_budget = 10**2
        logger.info(f"Done. Now obtaining estimates for ensemble sizes of size up to {max_budget}")
        for _ in tqdm(range(max_budget)):
            b.step()
        
        logger.info(f"Done, writing results to {filename}.")
        with open(filename, "w") as f:
            b.result_storage.serialize(f)


if __name__ == "__main__":

    
    run_experiment({
        "openmlid": 40668,
        "num_possible_ensemble_members": 1,
        "validation_size": 32,
        "data_seed": 0
    }, None, None)
    exit(0)
    

    if len(sys.argv) != 2:
        raise ValueError(f"Please specify exactly one argument (the job name).")
    name = sys.argv[1]
    sleep_time = np.random.rand() * 3
    print(f"Sleeping {sleep_time}s")
    time.sleep(sleep_time)

    pe = PyExperimenter(
        name=name,
        use_codecarbon=False,
        experiment_configuration_file_path=f"config/experiments.yaml"
        )
    pe.execute(max_experiments=-1, experiment_function=run_experiment)