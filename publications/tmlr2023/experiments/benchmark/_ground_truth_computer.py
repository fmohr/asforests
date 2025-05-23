import itertools as it
import pandas as pd
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed

import logging


def int_to_vector(num, base, d):
    vec = [0] * d
    for i in reversed(range(d)):
        num, vec[i] = divmod(num, base)
    return vec

def draw_unique_vectors_floyd(rs, num_samples, vector_length, max_index):
    base = max_index + 1
    N = base ** vector_length
    if num_samples > N:
        raise ValueError(f"Cannot draw {num_samples} unique vectors: only {N} possible.")

    # Floyd's algorithm for sampling without replacement
    selected = {}
    result = []
    for i in tqdm(range(N - num_samples, N)):
        t = rs.randint(0, i + 1)
        x = selected.get(t, t)
        selected[i] = selected.get(i, i)
        result.append(x)

    vectors = np.array([int_to_vector(num, base, vector_length) for num in result], dtype=np.int32)
    return vectors


class GroundTruthComputer:

    def __init__(self, deviations, logger=None):
        self.deviations = deviations
        self.logger = logger if logger is not None else logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")

    def get_true_parameter(self, param, t, n=None, bias=True):
        if param == "E[Z_nt]":
            if n is not None:
                Warning(f"n should be None for E[Z_nt] but is {n=}")
            return self.get_all_ensemble_data_combinations(n=2, t=t)["z"].mean()
        if param == "V[Z_nt]":
            if n is None:
                raise ValueError(f"n must be specified when querying the true value of v[Z_nt]")
            return self.get_all_ensemble_data_combinations(n=n, t=t)["z"].var(ddof=0 if bias else 1)
        if param == "E[Z_nt|D_val]":
            return self.get_all_ensemble_combinations_on_deviations(t=t)["z"].mean()
        if param == "V[Z_nt|D_val]":
            return self.get_all_ensemble_combinations_on_deviations(t=t)["z"].var(ddof=0 if bias else 1)

    
    def approximate_true_parameters_in_iid_setting_by_sampling(self, t_checkpoints, n_checkpoints=2, num_samples=10**6, num_samples_per_job=10**5, n_jobs=1, max_entries_in_batch_matrix = 10**8):
        """
            This method approximates the true parameters in the iid setting by creating random samples of BOTH datasets and ensembles.
            This is a crucial difference to bootstrapping, which samples only in the ensemble space. 
        
        Args:
            deviation_matrices (_type_): _description_
            n (_type_): _description_
            t (_type_): _description_
            num_samples (_type_, optional): _description_. Defaults to 10**4.
        """
        if num_samples_per_job is None:
            num_samples_per_job = num_samples
        num_sub_jobs = int(np.ceil(num_samples / num_samples_per_job))

        if not isinstance(n_checkpoints, (list, np.ndarray)):
            if not isinstance(n_checkpoints, (int, np.integer)):
                raise ValueError(f"n_checkpoints must be an int or a list/np.ndarray thereof but is {type(n_checkpoints)}")
            n_checkpoints = [n_checkpoints]
        if not isinstance(t_checkpoints, (list, np.ndarray)):
            if not isinstance(t_checkpoints, (int, np.integer)):
                raise ValueError(f"t_checkpoints must be an int or a list/np.ndarray thereof but is {type(t_checkpoints)}")
            t_checkpoints = [t_checkpoints]
        
        # check biggest values for n and t requested
        max_t = max(t_checkpoints)
        max_n = max(n_checkpoints)
        
        # define bookkeeping variables to manage the batch size
        batch_size = max(1, max_entries_in_batch_matrix // max(max_t * self.deviations.shape[1] * self.deviations.shape[2], max_n))
        num_batches = int(np.ceil(num_samples_per_job / batch_size))
        n_bar = num_batches * len(t_checkpoints) * len(n_checkpoints)

        self.logger.info(
            f"Approximating ground truth on world with {self.deviations.shape[0]} ensemble members on {self.deviations.shape[1]} instances. "
            f"We will use {num_samples} samples of Z_nt for each out of {len(t_checkpoints) * len(n_checkpoints)} n-t-combinations. "
            f"Sample values will be determine {'sequentially' if n_jobs == 1 else 'in parallelized manner (' + str(n_jobs) + ' jobs)'} "
            f"in {num_batches} batches of size {batch_size}, leading to a total of {n_bar} operations.")
        

        def collect_scores_for_job(seed, n_checkpoints, t_checkpoints):

            # create random state
            random_state = np.random.RandomState(seed)

            # outer loop over batches
            score_matrix = np.zeros((num_batches, batch_size, len(n_checkpoints), len(t_checkpoints)))
            pbar = tqdm(total=n_bar)
            for batch_idx in range(num_batches):

                # extract datasets and ensemble definitions for this batch
                ensembles_in_batch = random_state.randint(0, self.deviations.shape[0], size=(batch_size, max_t))
                datasets_in_batch = random_state.randint(0, self.deviations.shape[1], size=(batch_size, max_n))

                # compute the instance-wise value of Z_nt for all ensemble sizes simultaneously
                for j, t in enumerate(t_checkpoints):
                    ensemble_errors_on_instances = (self.deviations[ensembles_in_batch[:, :t]].mean(axis=1)**2).sum(axis=2)
                    for i, n in enumerate(n_checkpoints):
                        row_indices = np.arange(batch_size)[:, None]
                        score_matrix[batch_idx, :, i, j] = ensemble_errors_on_instances[row_indices, datasets_in_batch[:, :n]].mean(axis=1)
                        pbar.update(1)
            pbar.close()
            return score_matrix.reshape(-1, *score_matrix.shape[2:])
        
        if n_jobs != 1 and num_sub_jobs > 1:
            results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(collect_scores_for_job)(x, n_checkpoints, t_checkpoints) for x in range(num_sub_jobs))
        else:
            results = [collect_scores_for_job(x, n_checkpoints, t_checkpoints) for x in range(num_sub_jobs)]
        scores = np.concatenate(results)
        return np.mean(scores, axis=0), np.var(scores, axis=0, ddof=0)
    
    def approximate_true_parameters_in_cond_setting_by_sampling(self, t_checkpoints, num_samples=10**6, num_samples_per_job=10**5, n_jobs=1, max_entries_in_batch_matrix = 10**8):
        """
            This method approximates the true parameters in the iid setting by creating random samples of BOTH datasets and ensembles.
            This is a crucial difference to bootstrapping, which samples only in the ensemble space. 
        
        Args:
            deviation_matrices (_type_): _description_
            n (_type_): _description_
            t (_type_): _description_
            num_samples (_type_, optional): _description_. Defaults to 10**4.
        """
        if num_samples_per_job is None:
            num_samples_per_job = num_samples
        num_sub_jobs = int(np.ceil(num_samples / num_samples_per_job))
        
        if not isinstance(t_checkpoints, (list, np.ndarray)):
            if not isinstance(t_checkpoints, (int, np.integer)):
                raise ValueError(f"t_checkpoints must be an int or a list/np.ndarray thereof but is {type(t_checkpoints)}")
            t_checkpoints = [t_checkpoints]
        
        # check biggest values for n and t requested
        max_t = max(t_checkpoints)
        
        # define bookkeeping variables to manage the batch size
        batch_size = max(1, max_entries_in_batch_matrix // (max_t * self.deviations.shape[1] * self.deviations.shape[2]))
        num_batches = int(np.ceil(num_samples_per_job / batch_size))
        n_bar = num_batches * len(t_checkpoints)

        self.logger.info(
            f"Approximating ground truth on world with {self.deviations.shape[0]} ensemble members on {self.deviations.shape[1]} instances. "
            f"We will use {num_samples} samples of Z_nt for each out of {len(t_checkpoints)} t-checkpoints. "
            f"Sample values will be determine {'sequentially' if n_jobs == 1 else 'in parallelized manner (' + str(n_jobs) + ' jobs)'} "
            f"in {num_batches} batches of size {batch_size}, leading to a total of {n_bar} operations.")

        def collect_scores_for_job(seed, t_checkpoints):

            # create random state
            random_state = np.random.RandomState(seed)

            # outer loop over batches
            score_matrix = np.zeros((num_batches, batch_size, len(t_checkpoints)))
            pbar = tqdm(total=n_bar)
            for batch_idx in range(num_batches):

                # extract datasets and ensemble definitions for this batch
                ensembles_in_batch = random_state.randint(0, self.deviations.shape[0], size=(batch_size, max_t))

                # compute the instance-wise value of Z_nt for all ensemble sizes simultaneously
                for j, t in enumerate(t_checkpoints):
                    score_matrix[batch_idx, :, j] = (self.deviations[ensembles_in_batch[:, :t]].mean(axis=1)**2).mean(axis=1).sum(axis=1)
                    pbar.update(1)
            pbar.close()
            return score_matrix.reshape(-1, *score_matrix.shape[2:])
        
        if n_jobs != 1 and num_sub_jobs > 1:
            results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(collect_scores_for_job)(x, t_checkpoints) for x in range(num_sub_jobs))
        else:
            results = [collect_scores_for_job(x, t_checkpoints) for x in range(num_sub_jobs)]
        scores = np.concatenate(results)
        return np.mean(scores, axis=0), np.var(scores, axis=0, ddof=0)

    def get_all_ensemble_combinations_on_deviations(self, t, compute_deviations=False):
        """
            Computes a table with one row for every possible ensemble if `t` ensemble members are allowed.
            The last column `z` contains the score of that ensemble on the previously given deviations.

        Returns:
            _type_: _description_
        """
        
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

        # determine number of possible datasets
        num_possible_datasets = len(d_instances)**n
        num_allowed_datasets = (max_entries // len(d_ensemble_members)**4) if max_entries is not None else np.inf
        if logger is not None:
            logger.info(
                f"Identified that {num_possible_datasets} are possible for {n} validation instances sampled with replacement from a pool of {len(d_instances)}. "
                f"{num_allowed_datasets} are ALLOWED to not overstep the maximum table size."
            )

        # if the number of possible datasets is not at least 100 times as big as what we can accomodate with max entries (without thinking of ensemble members)
        if num_allowed_datasets == np.inf or num_possible_datasets <= num_allowed_datasets:
            if logger is not None:
                logger.info(f"Explicitly computing list of all possible datasets.")
            possible_datasets = np.array(list(it.product(*(n * [d_instances]))) if n > 1 else [(i, ) for i in d_instances])
            if num_possible_datasets > num_allowed_datasets:
                rs = np.random.RandomState(seed)
                possible_datasets = possible_datasets[rs.choice(range(possible_datasets.shape[0]), num_allowed_datasets, replace=False)]
        else:
            rs = np.random.RandomState(seed)
            num_datasets = max_entries // len(d_ensemble_members)**4
            if logger is not None:
                logger.info(f"Sampling {num_datasets} among all possible datasets.")
            possible_datasets = draw_unique_vectors_floyd(rs, num_datasets, n, len(d_instances) - 1)
        
        n_total = len(possible_datasets) * len(d_ensemble_members)**4
        if logger is not None:
            logger.info(f"Identified a base set of {len(possible_datasets)} datasets. Now building GT table with {n_total} entries.")
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
    
    def get_conditional_ground_truth_table(self, max_table_size=10**7):
        """
            Assumes that the instances given in `deviations` are the only ones available and that the dataset will look exactly like those
        """
        deviations = self.deviations
        d_ensemble_members = range(deviations.shape[0])
        d_targets = range(deviations.shape[2])
        n = deviations.shape[1]
        dataset_instances = list(range(n))

        # create dataframe with possible ensembles indices of size 4, and, for each of them, the deviations
        num_possible_ensembles = len(d_ensemble_members)**4
        rows = []
        cols = [f"i_{i}" for i in range(1, n + 1)] + ["s_1", "s_2", "s_3", "s_4"]
        for i in range(1, n + 1):
            for j in d_targets:
                cols.extend([f"D_{i}{j}^1", f"D_{i}{j}^2", f"D_{i}{j}^3", f"D_{i}{j}^4"])
        
        blow_up_factor = n * len(d_targets)
        num_expected_entries = num_possible_ensembles * blow_up_factor
        if num_expected_entries <= max_table_size:
            possible_ensemble_combinations = len(d_ensemble_members) ** 4
            relevant_combinations_of_ensemble_members = it.product(*(4 * [d_ensemble_members]))
        else:
            possible_ensemble_combinations = max_table_size // blow_up_factor
            expected_entries_after_update = possible_ensemble_combinations * blow_up_factor
            print(
                f"Expecting {num_expected_entries} many entries in table which is too large. "
                f"We will sample {possible_ensemble_combinations} ensembles to obtain a table with {expected_entries_after_update} entries."
            )
            relevant_combinations_of_ensemble_members = draw_unique_vectors_floyd(
                rs=np.random.RandomState(seed=0),
                num_samples=possible_ensemble_combinations,
                vector_length=4,
                max_index=len(d_ensemble_members) - 1
            )
            num_expected_entries = possible_ensemble_combinations * blow_up_factor

        print(f"Table will have {num_expected_entries} rows and {len(cols)} columns.")

        pbar = tqdm(total=possible_ensemble_combinations)
        for s1, s2, s3, s4 in relevant_combinations_of_ensemble_members:
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
