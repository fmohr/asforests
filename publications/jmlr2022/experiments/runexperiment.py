from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
import logging

from evalutils import *

import json

experimenter = PyExperimenter(config_file="config/experiments.cfg")


    
logger = logging.getLogger("exp")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

eval_logger = logging.getLogger("evalutils")
eval_logger.setLevel(logging.DEBUG)

def run_experiment(keyfields: dict, result_processor: ResultProcessor, custom_config):
    
    # Extracting given parameters
    openmlid = keyfields['openmlid']
    seed = keyfields['seed']
    max_diff = keyfields['max_diff']
    iterations_with_max_difff = keyfields['iterations_with_max_difff']
    
    logger.info(f"Starting experiment for openmlid {openmlid} and seed {seed}")
    
    # Write intermediate results to database    
    resultfields = {
        'scores': json.dumps(build_full_forest(openmlid, seed, max_diff, iterations_with_max_difff))
    }
    result_processor.process_results(resultfields)
    
    logger.info("Finished")
    

if __name__ == '__main__':
    experimenter.execute(run_experiment, max_experiments=-1, random_order=True)