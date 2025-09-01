from experiments.problem_instance.problem_instance import ProblemInstance
from experiments.benchmark._ground_truth_computer import GroundTruthComputer
from py_experimenter.experimenter import PyExperimenter
from pathlib import Path

from sklearn.datasets import fetch_openml
import logging

import sys
import numpy as np
import json

import gzip
import shutil

import time

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

    create_problem_instance_file_with_ground_truth_values(
        openmlid=int(keyfields["openmlid"]),
        data_seed=int(keyfields["data_seed"]),
        num_possible_ensemble_members=int(keyfields["num_possible_ensemble_members"]),
        validation_instances=int(keyfields["validation_instances"])
    )

def create_problem_instance_file_with_ground_truth_values(openmlid, data_seed, num_possible_ensemble_members, validation_instances):

    logger = logging.getLogger("experimenter")
    filename = f"problem_instances/{openmlid}_{data_seed}_{num_possible_ensemble_members}_{validation_instances}.json"
    filename_gz = filename + ".gz"
    path = Path(filename_gz)
    if path.exists():
        return

    # core configuration
    n_checkpoints=np.array([2, 10, 100, 1000])
    t_checkpoints=np.array([1, 2, 10, 100, 1000])
    num_samples_allowed_for_ground_truth_approximation = NUM_SAMPLES
    n_jobs=N_JOBS

    # get problem instance with 80% data out of sample (5% for training and a constant number for validation per class)
    pi = ProblemInstance(
        data_description=openmlid,
        is_classification=True,
        data_seed=data_seed,
        ensemble_seed=0,
        num_possible_ensemble_members=num_possible_ensemble_members,
        training_instances_per_class=0.05,
        validation_size=validation_instances,
        num_samples_allowed_for_ground_truth_approximation=num_samples_allowed_for_ground_truth_approximation,
        n_checkpoints=n_checkpoints,
        t_checkpoints=t_checkpoints,
        logger=logger
    )
    assert pi.predictions_val.shape[1] == validation_instances, f"Expected {validation_instances} validation instances, but ProblemInstance has {len(pi.predictions_val)}"

    # approximate ground truth for IID case
    num_samples = num_samples_allowed_for_ground_truth_approximation
    num_samples_per_job = int(np.ceil(num_samples / n_jobs))
    logger.info(
        f"Starting ground truth approximation for dataset {openmlid} (seed {data_seed}) under {num_possible_ensemble_members} possible ensemble members and {validation_instances} validation instances per class using {num_samples} samples generated through {n_jobs} jobs."
    )
    logger.info(f"Number of samples per job is {num_samples_per_job}")
    t_start = time.time()
    gtc_iid = GroundTruthComputer(deviations=pi.deviations, logger=logger)
    scores_iid = gtc_iid.sample_iid_scores(
        t_checkpoints=t_checkpoints,
        n_checkpoints=n_checkpoints,
        num_samples=num_samples_allowed_for_ground_truth_approximation,
        num_samples_per_job=num_samples_per_job,
        n_jobs=N_JOBS,
        max_entries_in_batch_matrix=10**8
    )
    print(scores_iid.shape)
    gtc_cond = GroundTruthComputer(deviations=pi.deviations_val, logger=logger)
    scores_cond = gtc_cond.sample_conditional_scores(
        t_checkpoints=t_checkpoints,
        num_samples=num_samples_allowed_for_ground_truth_approximation,
        num_samples_per_job=num_samples_per_job,
        n_jobs=N_JOBS,
        max_entries_in_batch_matrix=10**8
    )
    print(scores_cond.shape)
    t_end = time.time()
    logger.info(f"Overall time to approximate ground truth was {t_end - t_start}s")

    # dump instance
    d = pi.to_dict()
    d["scores_iid"] = np.round(scores_iid, 4).tolist()
    d["scores_cond"] = np.round(scores_cond, 4).tolist()
    d.pop("y_oh") # we don't want/need to serialize the ground truth labels
    d.pop("deviations") # we don't want/need to serialize the deviations
    d["validation_instances_per_class"] = validation_instances # memorize this configuration for easier later comparison
    path.parent.mkdir(exist_ok=True, parents=True)
    #print(d)
    with open(path, "w") as f:
        json.dump(d, f)

    # check deserializability and coincidence
    with open(path, "r") as f:
        d_rec = json.load(f)
        assert d == d_rec

    # gzip file and remove original
    with open(path, "rb") as f_in:
        with gzip.open(filename_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    path.unlink() # remove uncompressed file


if __name__ == "__main__":

    if len(sys.argv) != 3:
        raise ValueError(f"Please specify exactly two arguments (the job name and the number of cores to be used).")
    name = sys.argv[1]
    N_JOBS = int(sys.argv[2])
    NUM_SAMPLES = 10**5

    time.sleep(np.random.randint(0, 120))

    pe = PyExperimenter(
        name=name,
        use_codecarbon=False,
        experiment_configuration_file_path=f"ground_truth_experiments.yaml"
        )
    

    while True:
        try:
            pe.execute(max_experiments=-1, experiment_function=run_experiment)
            break
        except Exception as e:
            print("Observed a problem. Waiting 5 seconds and re-running the script.")
            print(e)
            time.sleep(5)
