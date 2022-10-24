from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
import logging

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

import sys

from asforests._grower import *



logger = logging.getLogger("exp")
logger.setLevel(logging.WARN)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_gap_to_final_score(input_history, openmlid, seed, w_min, epsilon, extrapolation_multiplier, delta, bootstrap_repeats):
    
    # create info supplier
    input_scores = [e[3] for e in input_history]
    info_supplier = get_dummy_info_supplier(input_scores)
    final_score = input_scores[-1]
    
    # now run algorithm
    start = time.time()
    grower = ForestGrower(info_supplier, d = 1, step_size = 1, w_min = w_min, epsilon = epsilon, extrapolation_multiplier = extrapolation_multiplier, delta = delta, max_trees = np.inf, random_state = seed, stop_when_horizontal = True, bootstrap_repeats = bootstrap_repeats, logger = logger)
    grower.grow()
    end = time.time()
    output_history = grower.histories[0]
    runtime = np.round((end - start) * 1000)
    return len(output_history), output_history[-1] - final_score, runtime

def run_experiment(keyfields: dict, result_processor: ResultProcessor, custom_config):
    
    # Extracting given parameters
    openmlid = keyfields['openmlid']
    seed = keyfields['seed']
    
    # load results
    df_results = pd.read_csv("results.csv", sep=";")
    row = df_results[(df_results["openmlid"] == openmlid) & (df_results["seed"] == seed)]["scores"].values[0].replace("b'", "").replace("'", "")
    input_history = json.loads(row)
    
    # prepare experiment
    epsilons = [10**i for i in range(-3, 0)]
    w_mins = [i for i in list(range(2, 10))]# + list(range(10, 100, 10)) + list(range(100, 1001, 100))]
    deltas = w_mins
    extrapolation_multipliers = [10**exp for exp in range(3)]
    bootstrap_repeats_options = [0, 2]#, 5, 10, 20, 100]
    
    domains = [epsilons, w_mins, deltas, extrapolation_multipliers, bootstrap_repeats_options]
    num_combinations = np.prod([len(D) for D in domains])
    
    logger.info(f"Starting experiment with {num_combinations} entries for openmlid {openmlid} and seed {seed}")
    
    # treat data sparse?
    pbar = tqdm(total = num_combinations)
    rows = []
    for epsilon in epsilons:
        for w_min in w_mins:
            for delta in deltas:
                for c in extrapolation_multipliers:
                    for bootstrap_repeats in bootstrap_repeats_options:
                        if delta <= w_min:
                            m,g,t = get_gap_to_final_score(input_history, openmlid, seed, w_min, epsilon, c, delta, bootstrap_repeats)
                            rows.append([epsilon, w_min, delta, c, bootstrap_repeats, m, np.round(g, 4), t])
                        pbar.update(1)
    pbar.close()
    out = json.dumps(rows)
    
    # Write intermediate results to database    
    resultfields = {
        "analysis": out
    }
    result_processor.process_results(resultfields)
    
    #logger.info("Finished")
    

if __name__ == '__main__':
    job_name = sys.argv[1]
    experimenter = PyExperimenter(config_file="config/experiments-analysis.cfg", name = job_name)
    experimenter.execute(run_experiment, max_experiments=-1, random_order=True)