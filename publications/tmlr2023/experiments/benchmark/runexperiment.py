from benchmark import Benchmark
from approaches import DatabaseWiseApproach, BootstrappingApproach, ParametricModelApproach

from tqdm import tqdm
from py_experimenter.experimenter import PyExperimenter

import pathlib
import sys


ACCEPTED_APPROACHES = ["bootstrapping", "databaseperparameter", "parametricmodel"]

def run_experiment(keyfields: dict, result_processor, custom_config):
    openmlid = int(keyfields["openmlid"])
    data_seed = 0
    ensemble_seed = int(keyfields["ensemble_seed"])
    training_size = 50
    validation_size = 20

    b = Benchmark(
        openmlid=openmlid,
        data_seed=data_seed,
        ensemble_seed=ensemble_seed,
        training_size=training_size,
        validation_size=validation_size,
        is_classification=True
    )

    # get generator for the estimates of the approach on the given problem
    t_checkpoints = [10, 100, 1000]#, 10000]
    if approach == "bootstrapping":
        approach_kwargs = {
            "num_resamples": int(keyfields["num_resamples"])
        }
        approach_obj = BootstrappingApproach(**approach_kwargs)
    elif approach == "databaseperparameter":
        approach_kwargs = {
            "population_mode": keyfields["population_mode"]
        }
        approach_obj = DatabaseWiseApproach(**approach_kwargs)
    elif approach == "parametricmodel":

        approach_kwargs = {
            "num_simulated_ensembles": int(keyfields["num_simulated_ensembles"]),
            "with_replacement": bool(keyfields["with_replacement"])
        }
        approach_obj = ParametricModelApproach(**approach_kwargs)
    else:
        raise ValueError(f"Unsupported approach {approach}")

    # run benchmark for 10 iterations (10 ensemble members)
    print(f"Running experiment on dataset {openmlid} with seeds {data_seed}/{ensemble_seed} estimating with {approach}({approach_kwargs})")
    b.reset({approach: approach_obj}, t_checkpoints=t_checkpoints)
    for _ in tqdm(range(10**3)):
        b.step()
    
    folder = f"results/{openmlid}_{data_seed}_{ensemble_seed}"
    pathlib.Path(folder).mkdir(exist_ok=True, parents=True)
    with open(f"{folder}/{approach}_{approach_kwargs}.json", "w") as f:
        b.result_storage.serialize(f)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError(f"Please specify exactly one argument (for the approach).")
    approach = sys.argv[1]
    if approach not in ACCEPTED_APPROACHES:
        raise ValueError(f"Please specify a valid approach (one of {ACCEPTED_APPROACHES}).")
    print(f"Evaluated approach is: {approach}")

    pe = PyExperimenter(
        use_codecarbon=False,
        experiment_configuration_file_path=f"config/{approach}.yaml"
        )
    pe.execute(max_experiments=-1, experiment_function=run_experiment)