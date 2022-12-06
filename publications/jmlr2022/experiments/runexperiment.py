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
    openmlid = keyfields['openmlid']
    seed = keyfields['seed']
    max_diff = keyfields['max_diff']
    iterations_with_max_difff = keyfields['iterations_with_max_difff']
    is_classification = openmlid in [3, 6, 12, 14, 16, 18, 21, 22, 23, 24, 26, 28, 30, 31, 32, 36, 38, 44, 46, 54, 57, 60, 179, 180, 181, 182, 184, 185, 188, 273, 293, 300, 351, 354, 357, 389, 390, 391, 392, 393, 395, 396, 398, 399, 401, 554, 679, 715, 718, 720, 722, 723, 727, 728, 734, 735, 737, 740, 741, 743, 751, 752, 761, 772, 797, 799, 803, 806, 807, 813, 816, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847, 849, 866, 871, 881, 897, 901, 903, 904, 910, 912, 913, 914, 917, 923, 930, 934, 953, 958, 959, 962, 966, 971, 976, 977, 978, 979, 980, 991, 995, 1000, 1002, 1019, 1020, 1021, 1036, 1037, 1039, 1040, 1041, 1042, 1049, 1050, 1053, 1059, 1067, 1068, 1069, 1116, 1119, 1120, 1128, 1130, 1134, 1138, 1139, 1142, 1146, 1161, 1166, 1216, 1242, 1457, 1461, 1464, 1468, 1475, 1485, 1486, 1487, 1489, 1494, 1501, 1515, 1569, 1590, 4134, 4136, 4137, 4534, 4538, 23380, 23512, 23517, 40497, 40498, 40668, 40670, 40685, 40691, 40701, 40900, 40926, 40971, 40975, 40978, 40981, 40982, 40983, 40984, 40996, 41026, 41027, 41064, 41065, 41066, 41138, 41142, 41143, 41144, 41145, 41146, 41150, 41156, 41157, 41158, 41159, 41161, 41163, 41164, 41165, 41166, 41167, 41168, 41169, 41946]
    
    logger.info(f"Starting experiment for openmlid {openmlid} and seed {seed}. Treated as {'classification' if is_classification else 'regression'}.")
    
    # treat data sparse?
    binarize_sparse = openmlid in [1111, 41147, 41150, 42732, 42733]
    drop = None if openmlid in [315] else 'first'
    
    # Write intermediate results to database
    if is_classification:
        resultfields = {
            'scores': json.dumps(build_full_classification_forest(openmlid, seed, max_diff, iterations_with_max_difff, binarize_sparse = binarize_sparse, drop = drop))
        }
    else:
        resultfields = {
            'scores': json.dumps(build_full_regression_forest(openmlid, seed, max_diff, iterations_with_max_difff, binarize_sparse = binarize_sparse, drop = drop))
        }
    result_processor.process_results(resultfields)
    
    logger.info("Finished")
    

if __name__ == '__main__':
    job_name = sys.argv[1]
    job_type = sys.argv[2]
    if False:
        experimenter = PyExperimenter(experiment_configuration_file_path=f"config/experiments-fullforests-{job_type}.cfg", name = job_name)
        experimenter.execute(run_experiment, max_experiments=-1, random_order=True)
    else:
        run_experiment({
            'openmlid': 3,
            'seed': 4,
            'max_diff': 0.00001,
            'iterations_with_max_difff': 1000
        }, None, None)