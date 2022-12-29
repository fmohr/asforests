from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
import logging

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

import sys
from datetime import datetime

from asforests._grower import *


ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger = logging.getLogger("exp")
logger.setLevel(logging.WARN)
logger.addHandler(ch)

eval_logger = logging.getLogger("evalutils")
eval_logger.setLevel(logging.DEBUG)
eval_logger.addHandler(ch)


def get_gap_to_final_score(scores, openmlid, seed, w_min, epsilon, extrapolation_multiplier, delta, bootstrap_repeats):

    # create info supplier
    info_supplier = get_dummy_info_supplier(scores)
    final_score = scores[-1]

    # now run algorithm
    start = time.time()
    step_size = min(w_min - 1, min(100, max(10, int(extrapolation_multiplier / 10**2))))
    grower = ForestGrower(info_supplier, d = 1, step_size = step_size, w_min = w_min, epsilon = epsilon, extrapolation_multiplier = extrapolation_multiplier, delta = delta, max_trees = np.inf, random_state = seed, stop_when_horizontal = True, bootstrap_repeats = bootstrap_repeats, logger = logger)
    try:
        grower.grow()
    except StopIteration:
        pass
    end = time.time()
    output_history = grower.histories[0]
    runtime = np.round((end - start) * 10**6 / len(output_history))
    return len(output_history), output_history[-1] - final_score, runtime


def run_experiment(keyfields: dict, result_processor: ResultProcessor, custom_config):
    
    # Extracting given parameters
    openmlid = keyfields['openmlid']
    seed = keyfields['seed']
    target_type = keyfields['target_type']
    problem_type = sys.argv[2]

    # load results
    logger.debug("Reading in data.")
    df_results = pd.read_csv(f"results_{problem_type}_base.csv", sep=";")
    scores = json.loads(df_results[(df_results["openmlid"] == openmlid) & (df_results["seed"] == seed)][f"scores_{target_type}"].values[0])

    # prepare experiment
    epsilons = [10**exp for exp in range(-3, 0)]
    w_mins = [i for i in list(range(2, 10)) + list(range(10, 100, 10)) + list(range(100, 1001, 100)) + list(range(1000, 10001, 1000))]
    w_mins.reverse()
    deltas = w_mins
    extrapolation_multipliers = [10**exp for exp in range(7)]
    extrapolation_multipliers.reverse()
    bootstrap_repeats_options = [0, 2, 5, 10, 20]
    bootstrap_repeats_options.reverse()

    domains = [epsilons, w_mins, deltas, extrapolation_multipliers, bootstrap_repeats_options]
    num_combinations = np.prod([len(D) for D in domains])

    print(f"Starting experiment with {num_combinations} entries for openmlid {openmlid}, seed {seed}. Target type is {target_type}.")

    # treat data sparse?
    pbar = tqdm(total = num_combinations)
    rows = []
    for epsilon in epsilons:
        for i, w_min in enumerate(w_mins):
            min_delta = deltas[-1]#deltas[max(0, i - 9)]
            for delta in deltas:
                for c in extrapolation_multipliers:
                    for bootstrap_repeats in bootstrap_repeats_options:
                        if delta >= min_delta and delta <= w_min:
                            print(f"{datetime.now()}: eps = {epsilon}, w_min = {w_min}, delta = {delta}, c = {c}, bt_repeats = {bootstrap_repeats}.")
                            m,g,t = get_gap_to_final_score(scores, openmlid, seed, w_min, epsilon, c, delta, bootstrap_repeats)
                            rows.append([epsilon, w_min, delta, c, bootstrap_repeats, m, np.round(g, 4), t])
                        pbar.update(1)
    pbar.close()
    out = json.dumps(rows)

    # Write intermediate results to database
    if result_processor is not None:
        resultfields = {
            "analysis": out
        }
        result_processor.process_results(resultfields)
    else:
        return out


if __name__ == '__main__':
    job_name = sys.argv[1]
    problem_type = sys.argv[2]
    if True:
        experimenter = PyExperimenter(experiment_configuration_file_path=f"config/experiments-analysis-{problem_type}.cfg", name = job_name)
        experimenter.execute(run_experiment, max_experiments=-1, random_order=True)
    else:
        run_experiment({
            "openmlid": 8,
            "seed": 4,
            "target_type": "val"
        }, None, None)
