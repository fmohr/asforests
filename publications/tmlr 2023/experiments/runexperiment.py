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
    seed = 0
    is_classification = sys.argv[2] == "classification"
    
    logger.info(f"Starting experiment for openmlid {openmlid} and seed {seed}. Treated as {'classification' if is_classification else 'regression'}.")
    
    # treat data sparse?
    binarize_sparse = openmlid in [1111, 4135, 4541, 41150, 42728, 42732, 42733, 42734]
    drop = None if openmlid in [3, 33, 38, 46, 57, 179, 180, 188, 231, 315, 897, 930, 953, 993, 1000, 1002, 1111, 1018, 1037, 1116, 1119, 1590, 4135, 4541, 4552, 40701, 40981, 41021, 41142, 41162, 41143, 42563, 42570, 42571, 42688, 42727, 42728, 42729, 42732, 42733, 42734] else 'first'
    
    # Write intermediate results to database
    if is_classification:
        resultfields = {
            'scores': json.dumps(build_full_classification_forest(openmlid, seed, zfactor=zfactor, eps=eps, binarize_sparse = binarize_sparse, drop = drop))
        }
    else:
        resultfields = {
            'scores': json.dumps(build_full_regression_forest(openmlid, seed, binarize_sparse = binarize_sparse, drop = drop))
        }
    result_processor.process_results(resultfields)
    
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
            'openmlid':  3,
            'seed': 4,
            'zfactor': 3,
            'eps': 0.001
        }, None, None)
