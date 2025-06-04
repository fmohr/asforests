from experiments.problem_instance.problem_instance import ProblemInstance
from benchmark import Benchmark
from approaches import DatabaseWiseApproach, BootstrappingApproach, ParametricModelApproach, ParametricDifferenceModelApproach
from approaches.approach import Approach
from typing import Dict

import logging

from tqdm import tqdm
from py_experimenter.experimenter import PyExperimenter

import matplotlib.gridspec as gridspec
from numpy.lib.stride_tricks import sliding_window_view

import json
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

    # read problem instance with known ground truth parameter values
    with open(f"{PATH_TO_PROBLEM_INSTANCES}/{openmlid}_{data_seed}_{num_possible_ensemble_members}_{validation_size}.json") as f:
        d = json.load(f)
        if "validation_instances_per_class" in d: # this is a relict not supported anymore
            d.pop("validation_instances_per_class")
        pi = ProblemInstance.from_dict(d)
        assert pi._true_means_for_iid_case is not None
        if pi._true_means_for_iid_case.shape == (len(pi.n_checkpoints), len(pi.t_checkpoints)):
            pi._true_means_for_iid_case = pi._true_means_for_iid_case.mean(axis=0)
        assert pi._true_means_for_iid_case.shape == (len(pi.t_checkpoints), )
        assert pi._true_vars_for_iid_case is not None
        assert pi._true_vars_for_iid_case.shape == (len(pi.n_checkpoints), len(pi.t_checkpoints))
        assert pi._true_means_for_cond_case is not None
        assert pi._true_means_for_cond_case.shape == (len(pi.t_checkpoints), )
        assert pi._true_vars_for_cond_case is not None
        assert pi._true_vars_for_cond_case.shape == (len(pi.t_checkpoints), )
        pi._load_data()
        #assert len(pi._indices_val) == validation_size, f"There should be {validation_size} validation instances, but there are {len(pi._indices_val)}"

    for ensemble_sequence_seed in range(5):

        # define name for result file and skip if we already have results for this
        filename = f"{folder}/{openmlid}_{data_seed}_{ensemble_sequence_seed}_{num_possible_ensemble_members}_{training_instances_per_class}_{validation_size}.json"
        if pathlib.Path(filename).exists():
            print(f"Skipping seed {ensemble_sequence_seed} since result file already exists.")
            continue
        
        import matplotlib.pyplot as plt
        history = {}
        def hook(approaches: Dict[str, Approach], result_store):
            df = result_store.get_errors_on_highest_budget()

            for param, df_param in df.groupby("param"):

                if param == "E[Z_nt]":
                    ground_truth = pi.means_iid
                elif param == "E[Z_nt|D_val]":
                    ground_truth = pi.means_cond
                elif param == "V[Z_nt]":
                    ground_truth = pi.vars_iid[0]
                elif param == "V[Z_nt|D_val]":
                    ground_truth = pi.vars_cond
                    print(df_param)
                else:
                    raise ValueError()
                
                if param not in history:
                    history[param] = {}
                
                param_history = history[param]

                margin = 10**-4 if param == "V[Z_nt|D_val]" else 0.1

                lower = ground_truth.min() - margin
                upper = ground_truth.max() + margin

                fig = plt.figure(figsize=(24, 12))
                gs = gridspec.GridSpec(5, 3, width_ratios=[1, 1, 1])

                if param == "V[Z_nt]":
                    df_param = df_param[df_param["n"] == 2]

                # actual approximations
                ax = fig.add_subplot(gs[0:2, 0])
                #domain = np.linspace(1, 1000, 20)
                domain = pi.t_checkpoints
                for a_name, df_approach in df_param.groupby("approach"):
                    estimates_of_approach_in_round = df_approach.sort_values("t")["estimate"]
                    ax.plot(domain, estimates_of_approach_in_round, marker="o")
                ax.scatter(pi.t_checkpoints, ground_truth, color="black", marker="*", s=100, label="Ground Truth", zorder=10**2)
                ax.grid()
                b = df_param['budget'].iloc[0]
                ax.axvline(b, color="black", linewidth=1, label="Budget $b$")
                ax.set_xscale("log")
                ax.set_xlabel("t")
                #ax.set_ylim([lower, upper])
                if "V[Z_nt" in param:
                    ax.set_yscale("log")
                ax.set_title(f"Predictions after {b} built ensemble members")
                ax.legend()

                # errors per anchor on log-scale
                ax = fig.add_subplot(gs[2:4, 0])
                for a_name, df_approach in df_param.groupby("approach"):
                    if a_name not in param_history:
                        param_history[a_name] = []
                    error_of_approach_in_round = np.abs(df_approach.sort_values("t")["error"])
                    param_history[a_name].append(error_of_approach_in_round)
                    ax.scatter(pi.t_checkpoints, error_of_approach_in_round)
                ax.axvline(b, color="black", linewidth=1, label="Budget $b$")
                ax.legend()
                ax.set_xscale("log")
                ax.set_xlabel("t")
                ax.set_yscale("log")
                ax.set_title(f"MAE after {b} built ensemble members")
                ax.grid()
                ax.set_ylim([10**-5 if param != "V[Z_nt|D_val]" else 10**-7, 10**-1 if param != "V[Z_nt|D_val]" else 10**-2])

                # learning curves
                for i, t in enumerate(pi.t_checkpoints, start=1):
                    ax = fig.add_subplot(gs[i - 1, 1])
                    for a_name, history_of_approach in param_history.items():
                        history_of_approach_as_array = np.array(history_of_approach)[:, i-1]
                        ax.plot(range(1, history_of_approach_as_array.shape[0] + 1), history_of_approach_as_array, label=a_name)
                    ax.set_xlim([1, 10**3])
                    ax.set_xscale("log")
                    ax.set_xlabel("Budget $b$ (Number of built ensemble members)")
                    ax.set_yscale("log")
                    ax.set_title(f"MAE over time for t={int(t)}")
                    ax.grid()
                    ax.set_ylim([10**-5 if param != "V[Z_nt|D_val]" else 10**-7, 10**-1 if param != "V[Z_nt|D_val]" else 10**-2])
                ax_with_labels_for_legend = ax

                # smoothened learning curves
                window_size = 10
                for i, t in enumerate(pi.t_checkpoints, start=1):
                    ax = fig.add_subplot(gs[i - 1, 2])
                    for a_name, history_of_approach in param_history.items():
                        history_of_approach_as_array = np.array(history_of_approach)[:, i - 1]
                        windows = sliding_window_view(history_of_approach_as_array, window_shape=min(len(history_of_approach_as_array), window_size))
                        if windows.shape[0] > 1:
                            ax.plot(range(window_size, window_size + windows.shape[0]), windows.mean(axis=1), label=a_name)
                    ax.set_xlim([1, 10**3])
                    ax.set_xscale("log")
                    ax.set_xlabel("Budget $b$ (Number of built ensemble members)")
                    ax.set_yscale("log")
                    ax.set_title(f"MAE over time for t={int(t)}")
                    ax.grid()
                    ax.set_ylim([10**-5 if param != "V[Z_nt|D_val]" else 10**-7, 10**-1 if param != "V[Z_nt|D_val]" else 10**-2])
                
                # create global legend of shared items
                handles, labels = ax_with_labels_for_legend.get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))

                # save figure
                fig.tight_layout()
                path = pathlib.Path(f"plots/animation_{openmlid}/{param}/{ensemble_sequence_seed}_{b}.jpeg")
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, bbox_inches="tight")
                print(f"Stored to {path}")
                plt.close()

        # create benchmark
        captured_parameters = ["V[Z_nt|D_val]"] #["E[Z_nt]", "E[Z_nt|D_val]", "V[Z_nt]", "V[Z_nt|D_val]"]
        b = Benchmark(
            problem_instance=pi,
            captured_parameters=captured_parameters,
            hooks=[hook]
        )

        # configure logger of approach
        a_logger = logging.getLogger("approach")
        a_logger.handlers.clear()
        a_logger.addHandler(ch)
        a_logger.setLevel(logging.INFO)
        approaches = {}
        for captured_parameter in captured_parameters:
            for num_simulated_ensembles in [10]:#, 1000]:
                approaches[f"{captured_parameter}::biparametric - {num_simulated_ensembles}"] = ParametricDifferenceModelApproach(
                    random_state=0,
                    estimated_parameters=[captured_parameter],
                    num_simulated_ensembles=num_simulated_ensembles,
                    logger=a_logger
                )
            
            
            for num_resamples, bootstrap_size in it.product([1, 10], [10]):#, 100]):
                approaches[f"{captured_parameter}::bootstrapping - {num_resamples}x{bootstrap_size}"] = BootstrappingApproach(
                    random_state=0,
                    estimated_parameters=[captured_parameter],
                    bootstrap_size=bootstrap_size,
                    num_resamples=num_resamples,
                    logger=a_logger
                )
            
            if False:

                for single_instance_per_ensemble_member in [False]:
                    approaches[f"{captured_parameter}::model free - stream - {'1 instance per member' if single_instance_per_ensemble_member else 'full'}"] = DatabaseWiseApproach(
                        random_state=0,
                        estimated_parameters=[captured_parameter],
                        population_mode="stream",
                        single_data_point_per_ensemble_member=single_instance_per_ensemble_member,
                        max_number_of_xi_terms_to_include_in_update=10**7,
                        upper_bound_for_sample_size=10**7,
                        logger=a_logger
                    )

        # configure logger of benchmark
        bm_logger = logging.getLogger("benchmark")
        bm_logger.handlers.clear()
        bm_logger.addHandler(ch)
        bm_logger.setLevel(logging.DEBUG)
        
        # run benchmark for 10 iterations (10 ensemble members)
        logger.info(f"Running experiment on dataset {openmlid} with data seed {data_seed}, ensemble sequence seed {ensemble_sequence_seed}, {validation_size} validation instances, and {num_possible_ensemble_members} possible ensemble members.")
        logger.info(f"Computing ground truth")
        b.reset(approaches)
        
        max_budget = 10**3
        logger.info(f"Done. Now obtaining estimates for ensemble sizes of size up to {max_budget}")
        for _ in tqdm(range(max_budget)):
            b.step()
        
        logger.info(f"Done, writing results to {filename}.")
        with open(filename, "w") as f:
            b.result_storage.serialize(f)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        raise ValueError(f"Please specify exactly two argument (the path to the problem instances with pre-computed ground truth values and the job name).")
    PATH_TO_PROBLEM_INSTANCES = pathlib.Path(sys.argv[1])
    name = sys.argv[2]

    if not PATH_TO_PROBLEM_INSTANCES.exists():
        raise ValueError(f"The path to the problem instances {PATH_TO_PROBLEM_INSTANCES} does not exist.")


    run_experiment({
        "openmlid": 3,
        "num_possible_ensemble_members": 8,
        "validation_size": 64,
        "data_seed": 0
    }, None, None)
    exit(0)

    sleep_time = np.random.rand() * 3
    print(f"Sleeping {sleep_time}s")
    time.sleep(sleep_time)

    pe = PyExperimenter(
        name=name,
        use_codecarbon=False,
        experiment_configuration_file_path=f"config/experiments.yaml"
        )

    while True:
        try:
            pe.execute(max_experiments=-1, experiment_function=run_experiment)
            break
        except Exception as e:
            print("Observed a problem. Waiting 5 seconds and re-running the script.")
            print(e)
            time.sleep(5)