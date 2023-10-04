from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
import logging

from evalutils import *

import json

import sys


    
logger = logging.getLogger("exp")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

eval_logger = logging.getLogger("evalutils")
eval_logger.setLevel(logging.DEBUG)
eval_logger.addHandler(ch)

def run_experiment(keyfields: dict, result_processor: ResultProcessor, custom_config):
    
    # Extracting given parameters
    openmlid = int(keyfields['openmlid'])
    seed = int(keyfields['seed'])
    zfactor = float(keyfields['zfactor'])
    eps = float(keyfields['eps'])
    is_classification = sys.argv[2] == "classification"
    
    logger.info(f"Starting experiment for openmlid {openmlid} and seed {seed}. Treated as {'classification' if is_classification else 'regression'}.")
    
    # Write intermediate results to database
    if is_classification:
        resultfields = {
            'scores': json.dumps(build_full_classification_forest(openmlid, seed, zfactor=zfactor, eps=eps))
        }
    else:
        resultfields = {
            'scores': json.dumps(build_full_regression_forest(openmlid, seed))
        }

    with open(f"results/{openmlid}_{seed}.json", "w") as f:
        json.dump(resultfields["scores"], f)

    #result_processor.process_results(resultfields)
    
    logger.info("Finished")
    

if __name__ == '__main__':
    job_name = sys.argv[1]
    job_type = sys.argv[2]
    if True:
        experimenter = PyExperimenter(
            experiment_configuration_file_path=f"config/experiments-fullforests-{job_type}.cfg",
            name=job_name,
            use_codecarbon=False
        )
        experimenter.execute(run_experiment, max_experiments=-1)
    else:
        run_experiment({
            'openmlid':  22,
            'seed': 1,
            'zfactor': 2,
            'eps': 0.1
        }, None, None)
