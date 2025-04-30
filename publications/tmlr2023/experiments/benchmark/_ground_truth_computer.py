import itertools as it
import pandas as pd
from tqdm import tqdm
import numpy as np


class GroundTruthComputer:

    def __init__(self, deviations):
        self.deviations = deviations

    def get_all_ensemble_combinations_on_deviations(self, t, compute_deviations=False):
        
        num_possible_ensembles = len(self.deviations)**t
        n = self.deviations[0].shape[0]
        total_size = num_possible_ensembles
        print(f"Determined {total_size} worlds, based on {num_possible_ensembles} possible ensembles of size {t}. Considering {n} validation instances.")

        # get definitions of possible ensembles and datasets for the given data
        ensembles = it.product(*(t * [range(len(self.deviations))]))
        
        # compose all possible worlds
        rows = []
        pbar = tqdm(total=total_size)
        for ensemble in ensembles:
            row = list(ensemble)
            ensemble_member_deviations = np.array([self.deviations[e_idx] for e_idx in ensemble])
            ensemble_deviation = ensemble_member_deviations.mean(axis=0)
            if compute_deviations:
                row.extend([ensemble_member_deviations[s, i, j] for i in range(n) for j in range(self.deviations.shape[2]) for s in range(t)])

            z = float((ensemble_deviation**2).mean(axis=0).sum())
            row.append(z)
            rows.append(row)
            pbar.update(1)
        pbar.close()
        
        columns = [f"s_{s + 1}" for s in range(t)]
        if compute_deviations:
            columns.extend([f"D_{i + 1}{j + 1}^{s + 1}" for i in range(n) for j in range(self.deviations.shape[2]) for s in range(t)])
        columns.append("z")
        
        return pd.DataFrame(rows, columns=columns)
    
    def get_all_ensemble_data_combinations(self, t, n, compute_deviations=False):
        
        num_possible_ensembles = len(self.deviations)**t
        num_existing_instances = self.deviations[0].shape[0]
        num_possible_application_sets = num_existing_instances**n
        total_size = num_possible_ensembles * num_possible_application_sets
        print(f"Determined {total_size} worlds, based on {num_possible_ensembles} possible ensembles of size {t} and {num_possible_application_sets} possible application sets of size {n} that can be formed from {num_existing_instances} instances.")

        # get definitions of possible ensembles and datasets for the given data
        ensembles = it.product(*(t * [range(len(self.deviations))]))
        datasets = it.product(*(n * [range(num_existing_instances)]))
        
        # compose all possible worlds
        rows = []
        pbar = tqdm(total=total_size)
        for ensemble, dataset in it.product(ensembles, datasets):
            row = list(dataset) + list(ensemble)
            ensemble_member_deviations = np.array([self.deviations[e_idx, dataset] for e_idx in ensemble])
            ensemble_deviation = ensemble_member_deviations.mean(axis=0)
            if compute_deviations:
                row.extend([ensemble_member_deviations[s, i, j] for i in range(n) for j in range(self.deviations.shape[2]) for s in range(t)])

            z = float((ensemble_deviation**2).mean(axis=0).sum())
            row.append(z)
            rows.append(row)
            pbar.update(1)
        pbar.close()
        
        columns = [f"x_{i + 1}" for i in range(n)] + [f"s_{s + 1}" for s in range(t)]
        if compute_deviations:
            columns.extend([f"D_{i + 1}{j + 1}^{s + 1}" for i in range(n) for j in range(self.deviations.shape[2]) for s in range(t)])
        columns.append("z")
        
        return pd.DataFrame(rows, columns=columns)

    def get_ground_truth_table(self, n, max_entries=None, seed=0, logger=None):
        deviations = self.deviations
        d_ensemble_members = range(deviations.shape[0])
        d_instances = range(deviations.shape[1])
        d_targets = range(deviations.shape[2])

        if logger is not None:
            logger.info(
                f"Trying to compute ground truth for {len(d_instances)} instances with {n=}, and {len(d_ensemble_members)} possible ensemble members using at most {max_entries=}."
            )

        # create dataframe with possible instance pairs and ensembles indices of size 4, and, for each of them, the deviations
        if max_entries is None or (n < 10 and len(d_instances)**n * len(d_ensemble_members)**4 <= max_entries):
            possible_datasets = list(it.product(*(n * [d_instances]))) if n > 1 else [(i, ) for i in d_instances]
        else:
            
            rs = np.random.RandomState(seed)
            possible_datasets = []
            generated_datasets = set()
            num_datasets = max_entries // len(d_ensemble_members)**4
            pbar = tqdm(total=num_datasets, disable=True)
            for _ in range(num_datasets):
                while True:
                    dataset = rs.choice(d_instances, n, replace=True)  # draw a dataset of n random instances
                    if not str(dataset) in generated_datasets:
                        possible_datasets.append(dataset)
                        generated_datasets.add(str(dataset))
                        pbar.update(1)
                        break
            pbar.close()
            possible_datasets = list(possible_datasets)
            if logger is not None:
                logger.warning(f"Cannot compute full ground truth table, approximating with {len(possible_datasets) * len(d_ensemble_members)**4} entries.")

        n_total = len(possible_datasets) * len(d_ensemble_members)**4
        pbar = tqdm(total=n_total)
        rows = []
        cols = [f"i_{i}" for i in range(1, n + 1)] + ["s_1", "s_2", "s_3", "s_4"]
        for i in range(1, n + 1):
            for j in d_targets:
                cols.extend([f"D_{i}{j}^1", f"D_{i}{j}^2", f"D_{i}{j}^3", f"D_{i}{j}^4"])
        for instance_indices, s1, s2, s3, s4 in it.product(possible_datasets, d_ensemble_members, d_ensemble_members, d_ensemble_members, d_ensemble_members):
            row = list(instance_indices) + [s1, s2, s3, s4]
            for i in instance_indices:
                for j in d_targets:
                    row.extend([deviations[s1, i, j], deviations[s2, i, j], deviations[s3, i, j], deviations[s4, i, j]])
            pbar.update(1)
            rows.append(row)
        pbar.close()
        df = pd.DataFrame(rows, columns=cols)

        for i in range(1, n + 1):
            for s1, s2 in [(1, 1), (1, 2), (2, 2), (1, 3), (3, 3), (3, 4)]:
                df[f"xi_{i}^{s1}{s2}"] = sum(df[f"D_{i}{j}^{s1}"] * df[f"D_{i}{j}^{s2}"] for j in d_targets)
        return df

    def get_ground_truth_table_under_sample_iid_assumption(self, max_entries=None, seed=0, logger=None):
        """
            Assume that the instances given in `deviations` are the only ones available and that we draw a certain number of instances of those i.i.d.

            if we have iid samples, then the ground truth table just requires two instances, no matter the actual number of application instances considered later
        """
        return self.get_ground_truth_table(n=2, max_entries=max_entries, seed=seed, logger=logger)
    
    def get_conditional_ground_truth_table(self):
        """
            Assumes that the instances given in `deviations` are the only ones available and that the dataset will look exactly like those
        """
        deviations = self.deviations
        d_ensemble_members = range(deviations.shape[0])
        d_targets = range(deviations.shape[2])
        n = deviations.shape[1]
        dataset_instances = list(range(n))

        # create dataframe with possible ensembles indices of size 4, and, for each of them, the deviations
        n_total = len(d_ensemble_members)**4
        pbar = tqdm(total=n_total)
        rows = []
        cols = [f"i_{i}" for i in range(1, n + 1)] + ["s_1", "s_2", "s_3", "s_4"]
        for i in range(1, n + 1):
            for j in d_targets:
                cols.extend([f"D_{i}{j}^1", f"D_{i}{j}^2", f"D_{i}{j}^3", f"D_{i}{j}^4"])
        for s1, s2, s3, s4 in it.product(*(4 * [d_ensemble_members])):
            row = dataset_instances + [s1, s2, s3, s4]
            for i in range(n):
                for j in d_targets:
                    row.extend([deviations[s1, i, j], deviations[s2, i, j], deviations[s3, i, j], deviations[s4, i, j]])
            pbar.update(1)
            rows.append(row)
        pbar.close()
        df = pd.DataFrame(rows, columns=cols)

        ext_dict = {}
        for i in range(1, n + 1):
            for s1, s2 in [(1, 1), (1, 2), (2, 2), (1, 3), (3, 3), (3, 4)]:
                ext_dict[f"xi_{i}^{s1}{s2}"] = sum(df[f"D_{i}{j}^{s1}"] * df[f"D_{i}{j}^{s2}"] for j in d_targets)
        df = pd.concat([df, pd.DataFrame(ext_dict)], axis=1)
        return df

    def get_covariance_terms_for_each_instance_pair(self, ground_truth_table, ddof=0):
        instances_cols = [c for c in ground_truth_table.columns if c.startswith("i_")]
        df = ground_truth_table

        rows = []
        for (i1, i2) in it.product(* (2 * [list(range(1, 1 + len(instances_cols)))])):
            rows.append([i1, i2,
                df[[f"xi_{i1}^11", f"xi_{i2}^11"]].cov(ddof=ddof).values[0, 1],
                df[[f"xi_{i1}^12", f"xi_{i2}^12"]].cov(ddof=ddof).values[0, 1],
                df[[f"xi_{i1}^11", f"xi_{i2}^12"]].cov(ddof=ddof).values[0, 1],
                df[[f"xi_{i1}^11", f"xi_{i2}^22"]].cov(ddof=ddof).values[0, 1],
                df[[f"xi_{i1}^12", f"xi_{i2}^33"]].cov(ddof=ddof).values[0, 1],
                df[[f"xi_{i1}^12", f"xi_{i2}^13"]].cov(ddof=ddof).values[0, 1],
                df[[f"xi_{i1}^12", f"xi_{i2}^34"]].cov(ddof=ddof).values[0, 1]
            ])
        return pd.DataFrame(rows, columns=["i_1", "i_2", "cov^1111", "cov^1212", "cov^1112", "cov^1122", "cov^1233", "cov^1213", "cov^1234"])
