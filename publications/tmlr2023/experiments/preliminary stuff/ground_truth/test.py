import matplotlib.pyplot as plt

from finite_task_generator import FiniteTaskGenerator
from asforests import EnsemblePerformanceAssessor
import numpy as np

from tqdm import tqdm

if __name__ == "__main__":

    gen = FiniteTaskGenerator(
        num_possible_instances=10,
        num_visible_instances=10,
        num_possible_members=4,
        num_targets=3,
        method="finite_rf",
        ignore_missing_uniqueness=True,
        rs=np.random.RandomState(0)
    )
    gen.generate_deviations()
    
    target_mean = gen.deviation_mean
    target_var = gen.deviation_var

    epas = {
        m: EnsemblePerformanceAssessor(
            upper_bound_for_sample_size=10**10,
            population_mode=m,
            estimate_deviation_covs=False
        )
        for m in ["stream"]#, "resample_no_replacement"]
    }
    errors_mean = {
        m: [] for m in epas.keys()
    }
    errors_var = {
        m: [] for m in epas.keys()
    }

    matrices = []
    for i in tqdm(range(10**7)):
        m = gen.sample_deviation_matrix()
        matrices.append(m)
        for epa_name, epa in epas.items():
            epa.add_deviation_matrix(m)
            errors_mean[epa_name].append(np.abs(epa.gap_mean_point - target_mean))
            errors_var[epa_name].append(np.abs(epa.gap_var_point - target_var))

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    for ax in axs[:, 0]:
        for key, errors_for_key in errors_mean.items():
            ax.plot(np.array(errors_for_key)[::10], label=key)
    for ax in axs[:, 1]:
        for key, errors_for_key in errors_var.items():
            ax.plot(np.array(errors_for_key)[::10], label=key)

    for ax in axs.flatten():
        ax.set_xscale("log")
        ax.grid()
        ax.set_ylim([0, 0.1])
        ax.legend()

    for ax in axs[1]:
        ax.set_yscale("log")
        ax.set_ylim([10**-6, 1])

    plt.show()
