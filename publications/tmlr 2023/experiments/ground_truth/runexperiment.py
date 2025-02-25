import time

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
import logging

import matplotlib.pyplot as plt

from evalutils import *

import json

import sys

# configure logging
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
    data_seed = 0  #int(keyfields['data_seed'])
    ensemble_seed = int(keyfields['ensemble_seed'])
    patience = 500  # int(keyfields['patience'])
    eps = 10**-5  # float(keyfields['eps'])
    is_classification = sys.argv[2] == "classification"

    logger.info(
        f"Starting experiment for openmlid {openmlid} and ensemble seed {ensemble_seed}. "
        f"Treated as {'classification' if is_classification else 'regression'}."
    )

    # Write intermediate results to database
    if is_classification:
        curve = get_performance_curve(
            openmlid,
            problem_type="classification",
            data_seed_application=data_seed,
            data_seed_training=data_seed,
            ensemble_seed=ensemble_seed,
            eps=eps,
            patience=patience
        )
        #plt.plot(curve)
        #plt.show()
    else:
        curve = None

    with open(f"results/{openmlid}_{data_seed}_{ensemble_seed}.json", "w") as f:
        json.dump(curve, f)

    # result_processor.process_results(resultfields)

    logger.info("Finished")


if __name__ == '__main__':

    sleeptime = np.random.rand() * 10
    print(f"Synthetically sleeping {sleeptime}s to make sure that experiments don't overlap.")
    time.sleep(sleeptime)

    job_name = sys.argv[1]
    job_type = sys.argv[2]
    experimenter = PyExperimenter(
        experiment_configuration_file_path=f"config-{job_type}.yaml",
        name=job_name,
        use_codecarbon=False
    )
    experimenter.execute(run_experiment, max_experiments=-1)
