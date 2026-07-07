import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import pandas as pd
import json
import os
import tarfile
import gzip
from tqdm import tqdm
from experiments.benchmark.result_storage import ResultStorage
import matplotlib.pyplot as plt


def aggregate_results(raw_result_folders, aggregate_result_folder):

    # first collect all storages we can find
    for folder in raw_result_folders:
        print(f"Checking {folder}")
        for filename in sorted(os.listdir(folder)):
            if not filename.endswith(".tar"):
                continue
            try:
                openmlid = int(filename[:-len(".tar")])
            except:
                continue

            
            out_file = f"{aggregate_result_folder}/{openmlid}.json"
            if Path(out_file).exists():
                continue
            else:
                Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            
            results_for_dataset = []
            print(f"Treating {folder}/{filename}")
            with tarfile.open(f"{folder}/{filename}", "r") as tar:
                # Iterate over all members (files) in the archive
                for member in tqdm(tar.getmembers()):
                    if member.isfile():  # Skip directories

                        # Extract the file as a file-like object
                        f = tar.extractfile(member)
                        if f is not None:
                            if member.name.endswith(".gz"):

                                d = member.name.split("/")[-1][:-8]
                                parts = d.split("_")
                                n = int(parts[-2]) if parts [-2] != "None" else None
                                t = int(parts[-1])

                                # Handle gzip file inside the tar
                                with gzip.open(f, "rt", encoding="utf-8", errors="ignore") as gz_file:
                                    try:
                                        rs = ResultStorage.unserialize(src=gz_file)
                                        for approach in rs.approach_names:
                                            run_data = rs.get_errors_from_approach_for_checkpoint(approach_name=approach, t=t, n_for_var_in_iid_case=n)
                                            param = run_data["param"].iloc[0]

                                            error_curve = list(run_data["error"])
                                            runtime_curve = list(run_data["runtime"])
                                            results_for_dataset.append([openmlid] + [int(p) if p != "None" else None for p in parts[1:]] + [approach, param, error_curve, runtime_curve])
                                    except:
                                        print(f"Error in reading of results {member.name}")
                            else:
                                print(f"Ignoring {member.name}")

                # create dataframe
                with open(out_file, "w") as f:
                    json.dump(pd.DataFrame(results_for_dataset, columns=["openmlid", "data_seed", "ensemble_seed", "num_possible_ensemble_members", "val_size", "n", "t", "approach", "param", "error_curve", "runtime_curve"]).to_json(), f)


def load_aggregated_results(folder, limit=None):
    subfolders = os.listdir(folder)
    if limit is not None:
        subfolders = subfolders[:limit]
    for file in tqdm(subfolders):
        with open(f"{folder}/{file}") as f:
            content = json.loads(json.load(f))
            yield pd.DataFrame(content).reset_index(drop=True)


def get_efficiency(error_curve, required_performance, tol):
    hits = np.where(np.array(error_curve) <= required_performance + tol)
    if len(hits) > 0 and len(hits[0]):
        return hits[0][0] + 1
    else:
        return np.inf
    
def plot_error_curves_for_approaches_on_single_dataset(df, window_size=1, b_max=10**10, x_scale="log", y_scale="log"):
    assert len(df) >= 1, f"No data available!"
    assert len(pd.unique(df["num_possible_ensemble_members"])) == 1
    assert len(pd.unique(df["val_size"])) == 1
    assert len(pd.unique(df["t"])) == 1
    assert len(pd.unique(df["n"])) == 1
    assert len(pd.unique(df["param"])) == 1
    param = df["param"].values[0]
    t = int(df["t"].values[0])
    n = int(df["n"].values[0]) if param == "V[Z_nt]" else None
    val_size = df["val_size"].values[0]

    for openmlid, df_dataset in df.groupby("openmlid"):

        fig, ax = plt.subplots(figsize=(20, 4))
        num_included_runs = []
        for algorithm, df_algo in df_dataset.groupby("approach"):
            num_included_runs.append(len(df_algo))

            errors_of_approach = np.abs(np.array([r for r in df_algo["RMSE_curve"]]))
            if window_size > 1:
                smoothened_errors_of_approach = []
                for row in errors_of_approach:
                    smoothened_errors_of_approach.append([np.mean(row[max(0, i - window_size):i]) for i in range(len(row))])
            else:
                smoothened_errors_of_approach = errors_of_approach
            smoothened_errors_of_approach = np.array(smoothened_errors_of_approach)
            mu = smoothened_errors_of_approach.mean(axis=0)[:b_max]
            sig = smoothened_errors_of_approach.std(axis=0)[:b_max]
            budgets = np.arange(1, len(mu) + 1)
            ax.plot(budgets, mu, label=algorithm)
            ax.fill_between(budgets, mu - sig, mu + sig, alpha=0.2)
        
        #ax.set_ylim([10**-4, 1])
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.set_xlabel("$b$")
        if b_max <= 30:
            ax.set_xticks(range(1, b_max + 1))
            ax.set_xticklabels(range(1, b_max + 1))
        ax.legend()
        ax.grid()
        ax.set_title(f"Estimation Error on {param} for dataset {openmlid}, {n=}, {t=}, and {val_size} instances for validation available ({np.mean(num_included_runs)} runs included on avg per approach)")
        plt.show()

def plot_runtime_curves_for_approaches_on_single_dataset(df):
    assert len(df) >= 1, f"No data available!"
    assert len(pd.unique(df["num_possible_ensemble_members"])) == 1
    assert len(pd.unique(df["val_size"])) == 1
    assert len(pd.unique(df["t"])) == 1
    assert len(pd.unique(df["n"])) == 1
    assert len(pd.unique(df["param"])) == 1
    
    param = df["param"].values[0]
    t = int(df["t"].values[0])
    n = int(df["n"].values[0]) if param == "V[Z_nt]" else None
    val_size = df["val_size"].values[0]

    for openmlid, df_dataset in df.groupby("openmlid"):

        fig, ax = plt.subplots(figsize=(20, 4))
        num_included_runs = []
        for algorithm, df_algo in df_dataset.groupby("approach"):
            num_included_runs.append(len(df_algo))

            runtimes_of_approach = np.abs(np.array([r for r in df_algo["runtime_curve"]]))
            runtimes_of_approach_acc = np.cumsum(runtimes_of_approach, axis=1)
            mu = runtimes_of_approach_acc.mean(axis=0)
            sig = runtimes_of_approach_acc.std(axis=0)
            budgets = np.arange(1, len(mu) + 1)
            ax.plot(budgets, mu, label=algorithm)
            ax.fill_between(budgets, mu - sig, mu + sig, alpha=0.2)
        
        #ax.set_ylim([0, 0.1])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$b$")
        ax.legend()
        ax.grid()
        ax.set_title(f"Runtimes to estimate {param} for dataset {openmlid}, {n=}, {t=}, and {val_size} instances for validation available ({np.mean(num_included_runs)} runs included on avg per approach)")
        plt.show()


def plot_stability_over_time_on_single_dataset(df, window=10):
    assert len(df) >= 1, f"No data available!"
    assert len(pd.unique(df["num_possible_ensemble_members"])) == 1
    assert len(pd.unique(df["val_size"])) == 1
    assert len(pd.unique(df["t"])) == 1
    assert len(pd.unique(df["n"])) == 1
    assert len(pd.unique(df["param"])) == 1
    
    param = df["param"].values[0]
    t = int(df["t"].values[0])
    n = int(df["n"].values[0]) if param == "V[Z_nt]" else None
    val_size = df["val_size"].values[0]

    for openmlid, df_dataset in df.groupby("openmlid"):

        fig, ax = plt.subplots(figsize=(20, 4))
        num_included_runs = []
        for algorithm, df_algo in df_dataset.groupby("approach"):
            num_included_runs.append(len(df_algo))

            errors_of_approach = [r for r in df_algo["RMSE_curve"]]
            stabilities_of_approach = []
            for row in errors_of_approach:
                stabilities_of_approach.append([np.std(row[max(0, i - window):i]) for i in range(len(row))])
            stabilities_of_approach = np.array(stabilities_of_approach)
            print(stabilities_of_approach.shape)
            mu = stabilities_of_approach.mean(axis=0)
            sig = stabilities_of_approach.std(axis=0)
            budgets = np.arange(1, len(mu) + 1)
            ax.plot(budgets, mu, label=algorithm)
            ax.fill_between(budgets, mu - sig, mu + sig, alpha=0.2)
        
        #ax.set_ylim([0, 0.1])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$b$")
        ax.legend()
        ax.grid()
        ax.set_title(f"{window}-std on Estimation Error on {param} for dataset {openmlid}, {n=}, {t=}, and {val_size} instances for validation available ({np.mean(num_included_runs)} runs included on avg per approach)")
        plt.show()