import pandas as pd
import numpy as np
from scipy.special import binom
from asforests.momenter_ import Momenter, MixedMomentBuilder
import itertools as it
import time
import logging
import matplotlib.pyplot as plt 

from collections import defaultdict
from typing import Callable, List

from numba import njit


def compute_relation(df1, df2, compatible_fn):
    result = []
    for _, row1 in df1.iterrows():
        for _, row2 in df2.iterrows():
            if compatible_fn(row1, row2):
                result.append(list(row1) + list(row2))

    return pd.DataFrame(result, columns=[f"{c}_x" for c in df1.columns] + [f"{c}_y" for c in df2.columns])


@njit
def fast_mask_k_uniques(arr, k):
    n = arr.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        seen = set()
        for j in range(4):
            seen.add(arr[i, j])
            if len(seen) > k:
                break
        mask[i] = len(seen) <= k
    return mask

def compute_xi_term_pairs_with_at_most_unique_members(df_terms_1, df_terms_2, instances_must_match, unique_members, symmetric=False):

    # if instances must match, we directly perform a merge in pandas since this is the faster solution
    if instances_must_match:
        
        df_combined = df_terms_1.merge(df_terms_2, on="i")
        unique_counts = [len(set(row)) for row in df_combined[["s1_x", "s2_x", "s1_y", "s2_y"]].values]
        df_combined = df_combined[np.array(unique_counts) <= unique_members]
    
    else:

        # Extract numpy arrays
        s1 = df_terms_1['s1'].values
        s2 = df_terms_1['s2'].values
        s3 = df_terms_2['s1'].values
        s4 = df_terms_2['s2'].values

        # get all possible index pairs
        t_start = time.time()
        ii, jj = np.meshgrid(np.arange(len(df_terms_1)), np.arange(len(df_terms_2)), indexing='ij')
        index_pairs = np.stack([ii.ravel(), jj.ravel()], axis=1)
        runtime = time.time() - t_start
        

        # Extract s1 and s2 values for all pairs
        s1_i = s1[index_pairs[:, 0]]
        s2_i = s2[index_pairs[:, 0]]
        s3_i = s3[index_pairs[:, 1]]
        s4_i = s4[index_pairs[:, 1]]

        # Combine and count unique values per row
        combined = np.stack([s1_i, s2_i, s3_i, s4_i], axis=1)

        # Filter pairs with the allowed number of distinct ensemble members
        mask = fast_mask_k_uniques(combined, k=unique_members)
        valid_pairs = index_pairs[mask]
        df_combined = pd.concat([
            df_terms_1.iloc[valid_pairs[:, 0]].reset_index(drop=True).rename(columns=lambda x: f"{x}_x"),
            df_terms_2.iloc[valid_pairs[:, 1]].reset_index(drop=True).rename(columns=lambda x: f"{x}_y")
        ], axis=1)

    if symmetric:

        # Find matching _x and _y columns
        x_cols = [col for col in df_combined.columns if col.endswith('_x')]
        y_cols = [col.replace('_x', '_y') for col in x_cols]

        # Swap the values
        df_mirror = df_combined[y_cols + x_cols].set_axis(x_cols + y_cols, axis=1)
        df_combined = pd.concat([df_combined, df_mirror])

    # Output as DataFrame
    return df_combined



#def compute_xi_term_pairs_with_two_unique_members(existing_xi_terms, new_xi_terms, instances_must_match):

class EnsemblePerformanceAssessor:
    """
        This is the main class for estimating the ensemble performance curve from individual members
    """

    def __init__(
            self,
            threshold_for_number_of_samples_to_exclude_param,
            population_mode,
            estimate_deviation_mean=True,
            estimate_deviation_var=True,
            estimate_deviation_covs=True,
            estimate_performance_var_for_iid_case=True,
            estimate_performance_var_for_conditional_case=True,
            max_number_of_recent_members_to_combine_with=np.inf,
            random_state=None,
            execute_asserts=False,
            enable_asserts=False, # only for debug mode since this slows down the code
            callbacks=[],
            logger=None
    ):
        """
        :param upper_bound_for_sample_size: constant or function of `t`. The maximum number of elements to be considered in any database used for estimation
        :param population_mode: `stream`, `resample_no_replacement`, `resample_with_replacement`
            In case of `stream`, the databases are filled until the maximum sample size is reached.
            In case of `resample_no_replacement`, the database is sampled uniformly without replacement from scratch whenever new data becomes available; at most `upper_bound_for_sample_size` elements are being sampled (withtout replacement). Is identical to `stream` if the number of possible samples is smaller than `upper_bound_for_sample_size`
            In case of `resample_with_replacement`, the database is sampled uniformly with replacement from scratch whenever new data becomes available; in this case, the upper bound is filled up exactly
        """

        # configuration
        self.population_mode = population_mode
        self.threshold_for_number_of_samples_to_exclude_param = threshold_for_number_of_samples_to_exclude_param
        if isinstance(self.threshold_for_number_of_samples_to_exclude_param, np.ndarray):
            if self.threshold_for_number_of_samples_to_exclude_param.shape != (7, ):
                raise ValueError(f"threshold_for_number_of_samples_to_exclude_param must be of shape (7, ) but is {self.threshold_for_number_of_samples_to_exclude_param.shape}.")

        elif not isinstance(self.threshold_for_number_of_samples_to_exclude_param, (np.number, int, float)):
            raise ValueError(f"threshold_for_number_of_samples_to_exclude_param must be a number or a numpy array of 7 entries but is {type(self.threshold_for_number_of_samples_to_exclude_param)}.")

        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state
        self.estimate_deviation_mean = estimate_deviation_mean
        self.estimate_deviation_var = estimate_deviation_var
        self.estimate_deviation_covs = estimate_deviation_covs
        self.estimate_performance_var_for_iid_case = estimate_performance_var_for_iid_case
        self.estimate_performance_var_for_conditional_case = estimate_performance_var_for_conditional_case
        self.max_number_of_recent_members_to_combine_with = max_number_of_recent_members_to_combine_with
        self.execute_asserts = execute_asserts
        self.enable_asserts = enable_asserts
        for i, callback in enumerate(callbacks):
            if not isinstance(callback, Callback):
                raise ValueError(f"Callback #{i} should be of type {__file__}.Callback but is {type(callback)}")
            callback.set_epa(self)

        self.callbacks = callbacks
        self.logger = logging.getLogger("EnsemblePerformanceEstimator") if logger is None else logger

        # sanity check
        accepted_modes = ["stream", "resample_no_replacement", "resample_with_replacement"]
        if population_mode not in accepted_modes:
            raise ValueError(f"population_mode must be in {accepted_modes} but is {population_mode}")

        # state variables
        self._basic_moment_builder_active = self.estimate_deviation_mean or self.estimate_deviation_var
        self._cov_moment_builder_active = self.estimate_deviation_covs
        self._var_estimator_active = self.estimate_performance_var_for_conditional_case or self.estimate_performance_var_for_iid_case
        self._active = self._basic_moment_builder_active or self._cov_moment_builder_active or self._var_estimator_active
        self.deviation_matrices = []
        self.masks_for_valid_instances = []
        self.n = None
        self.k = None
        self.moment_builder = None
        self.mixed_moment_builder = None
        #self.mixed_moment_builders_for_conditional_xi_covs = None
        #self.mixed_moment_builders_for_iid_xi_covs = None
        self.cov_updater_for_conditional_case = None
        self.cov_updater_for_iid_case_equal_instances = None
        self.cov_updater_for_iid_case_arbitrary_instances = None
        self.xi_terms = None
        self.xi_term_means = None
        self.data_points_processed_for_cov_estimate = 0

    @property
    def t(self):
        return len(self.deviation_matrices)
    
    @property
    def active(self):
        return self._active

    @property
    def gap_mean_point(self):
        if not self.estimate_deviation_mean:
            raise ValueError(f"Estimator is not configured to estimate deviation means, so these cannot be retrieved.")
        return self.moment_builder.means

    @property
    def gap_var_point(self):
        if not self.estimate_deviation_var:
            raise ValueError(f"Estimator is not configured to estimate variances, so these cannot be retrieved.")
        return self.moment_builder.central_moments[1]

    @property
    def gap_cov_across_members_point(self):
        if not self.estimate_deviation_covs:
            raise ValueError(f"Estimator is not configured to estimate covariances, so these cannot be retrieved.")
        if self.t < 2:
            return 0
        return self.mixed_moment_builder.cov

    @property
    def gap_mean_cb(self):
        div = self.n * self.t
        return np.sqrt(self.moment_builder.central_moments[1] / div)

    @property
    def gap_var_cb(self):
        div = self.n * self.t
        return np.sqrt((self.moment_builder.central_moments[3] - (div - 3) / (div - 1) * self.moment_builder.central_moments[
            1] ** 2) / div)

    @property
    def gap_cov_across_members_cb(self):
        div = self.n * self.t
        return np.sqrt((self.mixed_moment_builder.m4_ - (div - 3) / (div - 1) * self.moment_builder.central_moments[
            1] ** 2) / div)

    @property
    def expected_performance(self):
        return lambda t: (
            np.sum(self.gap_mean_point ** 2) +
            np.sum(self.gap_var_point) / t +
            (1 - 1 / t) * np.sum(self.gap_cov_across_members_point)
        )

    def add_deviation_matrix(self, d: np.ndarray) -> None:
        """

        :param d: an n x k matrix (np array) where n is the number of validation instances and k the number of targets
        :return: None
        """

        if not self.active:
            self.logger.debug(f"Assessor is inactive, ignoring invocation to add_deviation_matrix")
            return

        start = time.time()
        self.logger.info(
            f"Adding deviation matrix of shape {d.shape} to Ensemble Performance Estimator. "
            f"Computing estimates based on {len(self.deviation_matrices) + 1} ensemble members."
        )
        for callback in self.callbacks:
            callback.on_round_start()

        # format check
        if self.n is None:
            self.n, self.k = d.shape
        elif (self.n, self.k) != d.shape:
            raise ValueError(f"Expected gap format is ({self.n}, {self.k}) but observed {d.shape}")
        self.deviation_matrices.append(d)
        self.masks_for_valid_instances.append(np.all(~np.isnan(d), axis=1))

        # append observations to datasets if the population mode is "stream"
        if self.population_mode == "stream":

            # initialize moment builder
            if self.moment_builder is None:
                self.moment_builder = Momenter(input_dims=(self.k, 1), max_p=2)
                self.mixed_moment_builder = MixedMomentBuilder()

            # update estimates of E[D^1] and V[D^1]
            if self._basic_moment_builder_active:
                self.logger.info("Updating estimates of E[D^1] and V[D^1]")
                allowed_observations = np.max(self.threshold_for_number_of_samples_to_exclude_param) if self.moment_builder.n is None else max([0, min(np.max(self.threshold_for_number_of_samples_to_exclude_param) - self.moment_builder.n)])
                if allowed_observations > 0:
                    self.moment_builder.add_batch(d[:allowed_observations])
                    if np.all(self.moment_builder.n >= self.threshold_for_number_of_samples_to_exclude_param):
                        self.logger.info(f"Used {self.moment_builder.n} instances for all targets, now disabling moment builder.")
                        self._basic_moment_builder_active = False
                    if self.enable_asserts:
                        if self.execute_asserts and allowed_observations >= len(d):
                            assert np.all(np.isclose(self.moment_builder.means_, np.mean(self.deviation_matrices, axis=(0, 1))))
                            if self.t > 1:
                                assert np.all(np.isclose(self.moment_builder.central_moments[1], np.var(self.deviation_matrices, axis=(0, 1))))
            else:
                self.logger.debug("Skipping update of estimates of E[D^1] and V[D^1] since these estimates are disabled or inactive due to saturation.")

            # update estimate of Cov[D^1, D^2]
            if self._cov_moment_builder_active:
                mask_for_valid_instances_s1 = self.masks_for_valid_instances[-1]
                
                for i, (d_s2, mask_for_valid_instances_s2) in enumerate(zip(self.deviation_matrices, self.masks_for_valid_instances)): # include the new one as well for this
                    allowed_observations = max([0, np.max(self.threshold_for_number_of_samples_to_exclude_param) - self.mixed_moment_builder.n])
                    
                    if allowed_observations <= 0:
                        break

                    valid_instances_for_comparison = mask_for_valid_instances_s1 & mask_for_valid_instances_s2

                    # create data frame with all new pairs of deviations on any symmetric pair of instances and ensemble members
                    m1 = d[valid_instances_for_comparison]
                    m2 = d_s2[valid_instances_for_comparison]

                    # if we are not combining a matrix with itself, also add the asynchronous complement
                    self.mixed_moment_builder.add_observations(m1, m2, axis=0)
                    if i < len(self.deviation_matrices) - 1:
                        self.mixed_moment_builder.add_observations(m2, m1, axis=0)
                
                # check whether to disabled this update from now on
                if np.any(self.threshold_for_number_of_samples_to_exclude_param <= self.mixed_moment_builder.n):
                    self.logger.info(
                        f"Considered {self.threshold_for_number_of_samples_to_exclude_param} samples for the cov-estimator of deviations. "
                        "Now disabling the mixed_moment_builder that estimates the deviation covariances."
                    )
                    self._cov_moment_builder_active = False
            
            # estimate V[Z_nt]
            if self._var_estimator_active:
                self.update_estimates_of_covs_of_xi_terms_based_on_last_added_deviation_matrix()
                if (
                    (self.cov_updater_for_conditional_case is None or not self.cov_updater_for_conditional_case.is_active) and
                    (self.cov_updater_for_iid_case_arbitrary_instances is None or not self.cov_updater_for_iid_case_arbitrary_instances.is_active) and
                    (self.cov_updater_for_iid_case_equal_instances is None or not self.cov_updater_for_iid_case_equal_instances.is_active) 
                ):
                    self._var_estimator_active = False
                    self.logger.info("All cov-updaters for variance estimation are saturated and now inactive. Disabling update of variance estimation.")
            else:
                self.logger.debug("Not updating estimate of covariance terms for variance estimation since this is not configured or inactive due to saturation.")
            
            # check whether we should disable the whole estimation unit
            if not self._basic_moment_builder_active and not self._cov_moment_builder_active and not self._var_estimator_active:
                self.logger.info("Deactivating the whole estimator since no more estimation components are active.")
                self._active = False
        
        # update estimates by resampling
        else:

            # create new moment builders from scratch
            self.moment_builder = Momenter(input_dims=(self.k, 1), max_p=2)
            self.mixed_moment_builder = MixedMomentBuilder()

            # sample entries for the mean and variance estimates
            if self.estimate_deviation_mean or self.estimate_deviation_var:
                deviation_matrices = np.array(self.deviation_matrices)
                observations_unified_across_members = deviation_matrices.reshape((-1, deviation_matrices.shape[-1]))
                if self.population_mode == "resample_no_replacement" and len(observations_unified_across_members) < self.threshold_for_number_of_samples_to_exclude_param:
                    self.moment_builder.add_batch(observations_unified_across_members)
                else:
                    indices = [int(i) for i in self.random_state.choice(
                        range(len(observations_unified_across_members)),
                        size=self.threshold_for_number_of_samples_to_exclude_param,
                        replace=(self.population_mode == "resample_with_replacement")
                    )]
                    self.moment_builder.add_batch(observations_unified_across_members[indices])

            # sample entries for the covariance estimates
            if self.estimate_deviation_covs and len(self.deviation_matrices) > 1:

                num_possible_entries = self.n * int(binom(self.t, 2))
                if self.population_mode == "resample_no_replacement":
                    cnt = 0
                    for dm1, dm2 in it.combinations(self.deviation_matrices, 2):
                        self.mixed_moment_builder.add_observations(dm1, dm2, axis=0)
                        cnt += 1
                        if cnt >= self.threshold_for_number_of_samples_to_exclude_param:
                            break
                elif self.population_mode == "resample_with_replacement":
                    instance_indices = self.random_state.choice(range(self.n), size=self.threshold_for_number_of_samples_to_exclude_param, replace=True)
                    possible_pairs = list(it.combinations(range(len(self.deviation_matrices)), 2))
                    pair_indices = self.random_state.choice(
                        range(len(possible_pairs)),
                        size=self.threshold_for_number_of_samples_to_exclude_param,
                        replace=True
                    )
                    assert not self.execute_asserts or len(instance_indices) == len(pair_indices)
                    col1 = []
                    col2 = []
                    for i_instance, i_pair in zip(instance_indices, pair_indices):
                        col1.append(self.deviation_matrices[possible_pairs[i_pair][0]][i_instance])
                        col2.append(self.deviation_matrices[possible_pairs[i_pair][1]][i_instance])
                    self.mixed_moment_builder.add_observations(np.array(col1), np.array(col2), axis=0)
                else:
                    raise ValueError(f"Uncovered case for population mode: {self.population_mode}")
        
        # execute callbacks
        for callback in self.callbacks:
            callback.on_round_end()

    def update_estimates_of_covs_of_xi_terms_based_on_last_added_deviation_matrix(self):
        if (
            (self.estimate_performance_var_for_iid_case and (
                (self.cov_updater_for_iid_case_equal_instances is not None and not self.cov_updater_for_iid_case_equal_instances.is_active) and
                (self.cov_updater_for_iid_case_arbitrary_instances is not None and not self.cov_updater_for_iid_case_arbitrary_instances.is_active)
            )) or
            (self.estimate_performance_var_for_conditional_case and self.cov_updater_for_conditional_case is not None and not self.cov_updater_for_conditional_case.is_active)
        ):
            self.logger.debug("Skipping cov updates since no cov updater is active anymore.")
            return
        self.logger.info("Updating estimate of covariance terms for variance estimation.")

        # initialize moment builders
        if self.estimate_performance_var_for_conditional_case and self.cov_updater_for_conditional_case is None:
            self.logger.debug("Initializing DynamicCovUpdaters for conditional case")
            self.cov_updater_for_conditional_case = DynamicCovUpdater(
                name="cond",
                require_identical_instances_on_both_sides=False,
                threshold_for_number_of_samples_to_exclude_param=self.threshold_for_number_of_samples_to_exclude_param,
                max_number_of_recent_members_to_combine_with=self.max_number_of_recent_members_to_combine_with,
                random_state=self.random_state,
                callbacks=self.callbacks,
                logger=self.logger
            )
        
        if self.estimate_performance_var_for_iid_case and self.cov_updater_for_iid_case_equal_instances is None:
            self.logger.debug("Initializing DynamicCovUpdaters for iid case")
            self.cov_updater_for_iid_case_equal_instances = DynamicCovUpdater(
                name="iid_eq",
                require_identical_instances_on_both_sides=True,
                threshold_for_number_of_samples_to_exclude_param=self.threshold_for_number_of_samples_to_exclude_param,
                max_number_of_recent_members_to_combine_with=self.max_number_of_recent_members_to_combine_with,
                random_state=self.random_state,
                callbacks=self.callbacks,
                logger=self.logger
            )
            self.cov_updater_for_iid_case_arbitrary_instances = DynamicCovUpdater(
                name="iid_uneq",
                require_identical_instances_on_both_sides=False,
                threshold_for_number_of_samples_to_exclude_param=self.threshold_for_number_of_samples_to_exclude_param,
                max_number_of_recent_members_to_combine_with=self.max_number_of_recent_members_to_combine_with,
                skipped_cases=[3, 4, 6], # we know these cases have 0 values by theory, so we skip them
                random_state=self.random_state,
                callbacks=self.callbacks,
                logger=self.logger
            )
        
        # compute new xi-terms that can be shaped thanks to the newly added ensemble member
        need_xi_terms_for_pairs = (
            (self.cov_updater_for_conditional_case is not None and self.cov_updater_for_conditional_case.will_use_xi_terms_for_pairs) |
            (self.cov_updater_for_iid_case_equal_instances is not None and self.cov_updater_for_iid_case_equal_instances.will_use_xi_terms_for_pairs) |
            (self.cov_updater_for_iid_case_arbitrary_instances is not None and self.cov_updater_for_iid_case_arbitrary_instances.will_use_xi_terms_for_pairs)
        )
        new_xi_terms = []
        n, t = self.n, self.t
        if need_xi_terms_for_pairs:
            self.logger.debug(f"Computing {n * (2*t - 1)} new xi-terms") # for each instance, it is one \xi_i^tt for the new ensemble member and (t-1) \xi_i^st for each previous ensemble member
            for i in range(n):
                other_s = self.t - 1
                if np.any(np.isnan(self.deviation_matrices[other_s][i])):
                    continue
                for s in range(other_s + 1):
                    if np.any(np.isnan(self.deviation_matrices[s][i])):
                        continue
                    xi = np.dot(self.deviation_matrices[s][i], self.deviation_matrices[other_s][i])
                    involved_members = set([s, other_s])
                    new_xi_terms.append((i, s, other_s, involved_members, xi))
                    if s != other_s:
                        new_xi_terms.append((i, other_s, s, involved_members, xi))
        else:
            self.logger.debug(f"Computing {n} new xi-terms") # for each instance, it is one \xi_i^tt for the new ensemble member and (t-1) \xi_i^st for each previous ensemble member
            s = self.t - 1
            for i in range(n):
                xi = np.dot(self.deviation_matrices[s][i], self.deviation_matrices[s][i])
                new_xi_terms.append((i, s, s, set([s, s]), xi))

        df_new_xi_terms = pd.DataFrame(new_xi_terms, columns=["i", "s1", "s2", "involved_members", "xi"])
        df_new_xi_terms["same_member"] = df_new_xi_terms["s1"] == df_new_xi_terms["s2"]
        df_new_xi_terms["diff_member"] = ~df_new_xi_terms["same_member"]

        # if we are estimating V[Z_nt|D_val], update the covariance estimates for the mean xi pairs
        if self.estimate_performance_var_for_conditional_case:
            new_xi_term_means = df_new_xi_terms.groupby(["s1", "s2"]).mean(numeric_only=True).drop(columns="i").reset_index().astype({"same_member": bool, "diff_member": bool})
            self.logger.debug(f"Updating covariance estimates for conditional case based on {len(new_xi_term_means)} new xi mean terms.")
            self.cov_updater_for_conditional_case.update_covs(new_xi_terms=new_xi_term_means)

        # if we are estimating V[Z_nt], update the covariance estimates for the xi pairs (raw over instances)
        if self.estimate_performance_var_for_iid_case:
            self.logger.debug(f"Updating covariance estimates for iid case for identical instances scenario based on {len(df_new_xi_terms)} new xi terms.")
            if self.cov_updater_for_iid_case_equal_instances.is_active:
                self.cov_updater_for_iid_case_equal_instances.update_covs(new_xi_terms=df_new_xi_terms)
            if self.cov_updater_for_iid_case_arbitrary_instances.is_active:
                self.logger.debug(f"Updating covariance estimates for iid case for arbitrary instances scenario based on {len(df_new_xi_terms)} new xi terms.")
                self.cov_updater_for_iid_case_arbitrary_instances.update_covs(new_xi_terms=df_new_xi_terms)

class Callback(object):

    def __init__(self):
        self.epa = None

    def set_epa(self, epa):
        self.epa = epa
    
    def on_round_start(self):
        pass

    def on_xi_term_pair_computation(self, cov_updater):
        """

        Args:
            cov_updater (CovUpdate): Most importantly, this has a property `all_new_xi_pairs_available`, which is the dataframe with all newly available pairs
        """
        pass

    def on_round_end(self):
        pass

class DynamicCovUpdater:

    def __init__(
            self,
            name,
            require_identical_instances_on_both_sides,
            threshold_for_number_of_samples_to_exclude_param,
            max_number_of_recent_members_to_combine_with,
            random_state,
            logger,
            upper_bound_for_new_xi_pairs=np.inf,
            callbacks=[],
            skipped_cases=[]
        ):
        self.name = name
        self.require_identical_instances_on_both_sides = require_identical_instances_on_both_sides
        self.threshold_for_number_of_samples_to_exclude_param = threshold_for_number_of_samples_to_exclude_param
        self.skipped_cases = skipped_cases
        self.max_number_of_recent_members_to_combine_with = max_number_of_recent_members_to_combine_with
        self.upper_bound_for_new_xi_pairs = upper_bound_for_new_xi_pairs
        self.cov_builders = [MixedMomentBuilder() for _ in range(7)]
        self.random_state = random_state
        self.callbacks = callbacks
        self.logger = logger
        
        self.num_of_xi_terms = 0
        self.instance_multiplier = 1
        self.old_xi_terms = None
        self.new_xi_terms = None
        self.finished_rounds = 0

        self.new_xi_pairs = None # dataframe with all pairs of xi terms that are to be used for the next round
        self.all_new_xi_pairs_available = False
    
    @property
    def num_used_samples_per_cov_estimate(self):
        return np.array([b.n for b in self.cov_builders])
    
    @property
    def mask_of_active_params(self):
        thresholds = self.threshold_for_number_of_samples_to_exclude_param if isinstance(self.threshold_for_number_of_samples_to_exclude_param, np.ndarray) else self.threshold_for_number_of_samples_to_exclude_param * np.ones(7)
        return np.array([(b.n < threshold) and i not in self.skipped_cases for i, (b, threshold) in enumerate(zip(self.cov_builders, thresholds))])
    
    @property
    def cov(self):
        return np.array([cb.cov for cb in self.cov_builders])
    
    def get_highest_order_of_member_combinations_required(self):
        if self.mask_of_active_params[-1]:
            return 4
        if any(self.mask_of_active_params[-3:]):
            return 3
        if any(self.mask_of_active_params[1:]):
            return 2
        if any(self.mask_of_active_params):
            return 1
        return 0
    
    @property
    def is_active(self):
        return self.get_highest_order_of_member_combinations_required() > 0
    
    @property
    def will_use_xi_terms_for_pairs(self):
        return self.get_highest_order_of_member_combinations_required() > 1
    
    def get_number_of_xi_pairs_for_current_round(self):
        
        # determine number of base combinations for ensemble members
        highest_required_degree = self.get_highest_order_of_member_combinations_required()
        b = len(self.indices_of_considered_previous_ensemble_members) + 1
        if highest_required_degree == 4:
            additional_possible_terms = 4 * b**3  - 6 * b**2 + 4*b - 1 # this is the number of increase in possible pairs of pairs (for last case)
        elif highest_required_degree == 3:
            additional_possible_terms = 18 * b**2 - 40 * b + 23 # this is the number of increase in pairs with at most 3 unique ensemble members
        elif highest_required_degree == 2:
            additional_possible_terms = 14 * b - 13  # this is the number of increase in possible pairs with at most 2 unique ensemble members
        elif highest_required_degree == 1:
            additional_possible_terms = 1
        else:
            additional_possible_terms = 0
        
        # if we can choose between several instances, multiply the number of possible terms by this number
        additional_possible_terms *= self.instance_multiplier
        return additional_possible_terms
    
    def _start_new_round(self, new_xi_terms):
        if not self.is_active:
            raise ValueError("Cannot start new round if the updater is inactive.")
        if self.new_xi_terms is not None:
            raise ValueError("Cannot start new round since there are already new xi terms available. Please call `finish_round` first.")
        max_t = new_xi_terms["s1"].max()

        # check which order would be required in this round. If it is 0, we can already stop
        required_order = self.get_highest_order_of_member_combinations_required()

        # check whether there are multiple instances in these xi-terms
        self.instance_multiplier = len(pd.unique(new_xi_terms["i"])) if "i" in new_xi_terms.columns else 1
        if not self.require_identical_instances_on_both_sides:
            self.instance_multiplier = self.instance_multiplier**2

        # forget old xi terms that connect the new one with which we pair the new ensemble member 
        if self.max_number_of_recent_members_to_combine_with < np.inf:
            self.indices_of_considered_previous_ensemble_members = [
                int(i) for i in sorted(
                    self.random_state.choice(range(max_t), size=self.max_number_of_recent_members_to_combine_with, replace=False)
                    if max_t > self.max_number_of_recent_members_to_combine_with
                    else range(max_t)
                )
            ]
            self.logger.debug(f"Considering only combinations of new ensemble member {max_t} with ensemble members {self.indices_of_considered_previous_ensemble_members}")
        else:
            self.indices_of_considered_previous_ensemble_members = range(max_t)

        # get xi terms that comply with the considered previous ensemble members (usually all possible new xi-terms unless restricted explicitly above)
        self.new_xi_terms = new_xi_terms[
            (new_xi_terms["s1"].isin(list(self.indices_of_considered_previous_ensemble_members) + [max_t])) &
            (new_xi_terms["s2"].isin(list(self.indices_of_considered_previous_ensemble_members) + [max_t]))
        ]
        self.considered_old_xi_terms = None if self.old_xi_terms is None else self.old_xi_terms[
            (self.old_xi_terms["s1"].isin(self.indices_of_considered_previous_ensemble_members)) &
            (self.old_xi_terms["s2"].isin(self.indices_of_considered_previous_ensemble_members))
        ]

        # determine how many new xi pairs will be required (depends on which of the 7 covariances are still active)
        num_of_complete_new_xi_pairs_to_update_active_params = self.get_number_of_xi_pairs_for_current_round()
        
        # if we expect more new xi pairs than we can handle, we will not compute them
        if self.upper_bound_for_new_xi_pairs < np.inf and num_of_complete_new_xi_pairs_to_update_active_params > self.upper_bound_for_new_xi_pairs:
            self.all_new_xi_pairs_available = None
            new_xi_terms = self.new_xi_terms.copy()
            new_xi_terms["exists"] = False
            if self.considered_old_xi_terms is not None:
                self.considered_old_xi_terms["exists"] = True    
                self.merged_xi_terms = pd.concat([self.considered_old_xi_terms, new_xi_terms], ignore_index=True)
            else:
                self.merged_xi_terms = new_xi_terms
            self.logger.warning(f"Expected {num_of_complete_new_xi_pairs_to_update_active_params} new xi pairs but only {self.upper_bound_for_new_xi_pairs} can be handled. xi-pairs will be sampled.")
        
        # otherwise compute a daframe of all new (relevant) xi pairs
        else:

            if self.considered_old_xi_terms is not None:

                # if we need complete coverage of all possible combinations
                t_start_xi_pair_computation = time.time()

                self.logger.debug(f"Computing all xi term pairs with up to {required_order} distinct ensemble members.")
                if required_order == 4:
                    if self.require_identical_instances_on_both_sides:
                        self.all_new_xi_pairs_available = pd.concat([
                            self.considered_old_xi_terms.merge(self.new_xi_terms, on="i"),
                            self.new_xi_terms.merge(self.considered_old_xi_terms, on="i"),
                            self.new_xi_terms.merge(self.new_xi_terms, on="i")
                        ])
                        self.logger.debug(f"Join on instances took {np.round(time.time() - t_start_xi_pair_computation, 4)}s")
                    else:
                        self.all_new_xi_pairs_available = pd.concat([
                            self.considered_old_xi_terms.merge(self.new_xi_terms, how="cross"),
                            self.new_xi_terms.merge(self.considered_old_xi_terms, how="cross"),
                            self.new_xi_terms.merge(self.new_xi_terms, how="cross")
                        ])
                        self.logger.debug(f"Cartesian product computation took {np.round(time.time() - t_start_xi_pair_computation, 4)}s")
                
                # if we need only up to 3 distinct ensemble members
                elif required_order in [2, 3]:
                    
                    # in this case, in the existing xi terms we can only have those where both members are identical (because we also need a new one in the new term)
                    if required_order == 2:
                        relevant_old_terms = self.considered_old_xi_terms[self.considered_old_xi_terms["s1"] == self.considered_old_xi_terms["s2"]]
                    else:
                        relevant_old_terms = self.considered_old_xi_terms
                        
                    # first case: the new ensemble member occurs only in one of the xi-terms
                    c1 = compute_xi_term_pairs_with_at_most_unique_members(self.new_xi_terms, relevant_old_terms, instances_must_match=self.require_identical_instances_on_both_sides, unique_members=required_order, symmetric=True)
                        
                    # second case: the new ensemble member occurs in both xi-terms
                    c2 = compute_xi_term_pairs_with_at_most_unique_members(self.new_xi_terms, self.new_xi_terms, instances_must_match=self.require_identical_instances_on_both_sides, unique_members=required_order, symmetric=False)
                        
                    # merge everything
                    self.all_new_xi_pairs_available = pd.concat([c1, c2], ignore_index=True)
                    
                # in this case, only add the line with the new member with itself
                elif required_order == 1:

                    relevant_xi_terms = self.new_xi_terms[self.new_xi_terms["s1"] == self.new_xi_terms["s2"]]

                    if self.require_identical_instances_on_both_sides:
                        self.all_new_xi_pairs_available = relevant_xi_terms.merge(relevant_xi_terms, on="i")
                    else:
                        self.all_new_xi_pairs_available = relevant_xi_terms.merge(relevant_xi_terms, how="cross")
                
                time_xi_pair_computation = time.time() - t_start_xi_pair_computation
                #assert num_of_complete_new_xi_pairs_to_update_active_params == len(self.all_new_xi_pairs_available), f"Expected {num_of_complete_new_xi_pairs_to_update_active_params} new xi pairs for CovUpdater {self.name} on degree {required_order} but got {len(self.all_new_xi_pairs_available)}."
                achieved_pairs_per_s = len(self.all_new_xi_pairs_available) / time_xi_pair_computation
                self.logger.debug(f"Finished computation of {len(self.all_new_xi_pairs_available)} xi-pairs after {np.round(time_xi_pair_computation, 4)}s ({achieved_pairs_per_s} pairs/s).")
                if time_xi_pair_computation > 0.1 and achieved_pairs_per_s < 10**5:
                    self.logger.warning(f"Computation of {len(self.all_new_xi_pairs_available)} xi-pairs in updater {self.name} for degree {required_order} was slow and finished only after {np.round(time_xi_pair_computation, 4)}s ({achieved_pairs_per_s} pairs/s).")
            else:
                if self.require_identical_instances_on_both_sides:
                    self.all_new_xi_pairs_available = self.new_xi_terms.merge(self.new_xi_terms, on="i")
                else:
                    self.all_new_xi_pairs_available = self.new_xi_terms.merge(self.new_xi_terms, how="cross")
            for callback in self.callbacks:
                callback.on_xi_term_pair_computation(self)
        
        # memorize important masks
        if self.all_new_xi_pairs_available is not None:
            #if num_of_complete_new_xi_pairs_to_update_active_params is not None:
             #   assert num_of_complete_new_xi_pairs_to_update_active_params == len(self.all_new_xi_pairs_available), f"Expected {num_of_complete_new_xi_pairs_to_update_active_params} new xi pairs but got {len(self.all_new_xi_pairs_available)}."
            assert not np.any(np.isnan(self.all_new_xi_pairs_available[["xi_x", "xi_y"]].values))
            self.same_left = self.all_new_xi_pairs_available["same_member_x"]
            self.same_right = self.all_new_xi_pairs_available["same_member_y"]
            self.first_shared = self.all_new_xi_pairs_available["s1_x"] == self.all_new_xi_pairs_available["s1_y"]
            self.second_shared = self.all_new_xi_pairs_available["s2_x"] == self.all_new_xi_pairs_available["s2_y"]
    
    


    def update_covs(self, new_xi_terms):
        self._start_new_round(new_xi_terms=new_xi_terms)
        for case in range(7):
            if case not in self.skipped_cases and self.mask_of_active_params[case]:
                self._update_cov_for_case(case)
            else:
                self.logger.debug(f"Skipping case {case + 1} since it is not relevant for this xi term covariance estimation.")
        
        # finish round
        self.old_xi_terms = pd.concat([self.old_xi_terms, new_xi_terms]) if self.old_xi_terms is not None else new_xi_terms # important, NOT self.new_xi_terms
        self.num_of_xi_terms += len(self.new_xi_terms)
        self.new_xi_terms = None
        self.new_xi_pairs = None
        self.finished_rounds += 1
    
    def _update_cov_for_case(self, case):
        """
            All seven cases can be characterized by three boolean flags:
            - same_left: whether the ensemble members in the left xi term must be identical
            - same_right: whether the ensemble members in the right xi term must be identical
            - first_shared: whether the first ensemble member in the left xi term must be identical to the first ensemble member in the right xi term

        Args:
            case (_type_): _description_
        """
        if case in self.skipped_cases:
            raise RuntimeError(f"Asked to update case {case}, which is marked to be skipped.")
        if not self.mask_of_active_params[case]:
            raise RuntimeError(f"Asked to update case {case}, which is inactive.")

        # if we know all available xi pairs, we directly compute the relevant pairs for the case
        if self.all_new_xi_pairs_available is not None:
            self.logger.debug(f"Computing covariances for case {case} based on all {len(self.all_new_xi_pairs_available)} available xi pairs.")
            # get dataframe of xi pairs usable for this case
            if case == 0:
                mask = self.same_left & self.same_right & self.first_shared # case 11,11
            elif case == 1:
                mask = self.first_shared & self.second_shared # case 12, 12
            elif case == 2:
                mask = self.same_left & self.first_shared # case 11, 12
            elif case == 3:
                mask = self.same_left & self.same_right # case 11, 22
            elif case == 4:
                mask = self.same_right # case 12,33
            elif case == 5:
                mask = self.first_shared # case 12,13
            elif case == 6:
                mask = np.ones(len(self.all_new_xi_pairs_available)).astype(bool) # case 12,34
            else:
                raise ValueError(f"Unknown case {case} for xi term covariance estimation.")
            df_case = self.all_new_xi_pairs_available[mask]
            self.logger.debug(f"Computed all {len(df_case)} data points to update covariances for case {case}.")
        
        # otherwise, we need to sample the pairs from the new xi terms
        else:
            self.logger.debug(f"Updating covariance estimates for case {case} based on {self.upper_bound_for_new_xi_pairs} samples of new xi pairs.")
            if case == 0:
                df_case = self.sample_xi_term_pairs_between_existing_and_new_terms(same_in_left=True, same_in_right=True, first_shared=True)
            elif case == 1:
                df_case = self.sample_xi_term_pairs_between_existing_and_new_terms(first_shared=True, second_shared=True)
            elif case == 2:
                df_case = self.sample_xi_term_pairs_between_existing_and_new_terms(same_in_left=True, first_shared=True)
            elif case == 3:
                df_case = self.sample_xi_term_pairs_between_existing_and_new_terms(same_in_left=True, same_in_right=True)
            elif case == 4:
                df_case = self.sample_xi_term_pairs_between_existing_and_new_terms(same_in_right=True)
            elif case == 5:
                df_case = self.sample_xi_term_pairs_between_existing_and_new_terms(first_shared=True)
            elif case == 6:
                df_case = self.sample_xi_term_pairs_between_existing_and_new_terms()
                

        # update covariances for this case        
        t_start_update = time.time()
        self.cov_builders[case].add_observations(df_case["xi_x"], df_case["xi_y"])
        t_end_update = time.time()
        self.logger.debug(f"Update of covs for conditional case took {np.round(t_end_update - t_start_update, 6)}s")

    def sample_xi_term_pairs_between_existing_and_new_terms(self, same_in_left=False, same_in_right=False, first_shared=False, second_shared=False):
            
        # constraints potential candidates for left and right
        if same_in_left:
            left_xi_candidates = self.merged_xi_terms[self.merged_xi_terms["same_member"]]
        else:
            left_xi_candidates = self.merged_xi_terms
        if same_in_right:
            right_xi_candidates = self.merged_xi_terms[self.merged_xi_terms["same_member"]]
        else:
            right_xi_candidates = self.merged_xi_terms

        # determine fields that should be shared by the two xi terms
        share_set = []
        if first_shared:
            share_set.append("s1")
        if second_shared:
            share_set.append("s2")
        if self.require_identical_instances_on_both_sides:
            share_set.append("i")

        self.logger.debug(f"Computing xi-term pairs considering shared columns {share_set}")
        
        # if something is shared between the two xi terms we will see whether the sample base can be simplified
        if share_set:

            # determine the portion of xi-terms in left and right candidates that already exists
            portion_of_existing_xi_terms_left = np.count_nonzero(left_xi_candidates["exists"]) / len(left_xi_candidates)
            portion_of_existing_xi_terms_right = np.count_nonzero(right_xi_candidates["exists"]) / len(right_xi_candidates)
            portion_of_existing_xi_terms = (portion_of_existing_xi_terms_left * len(left_xi_candidates) + portion_of_existing_xi_terms_right * len(right_xi_candidates)) / (len(left_xi_candidates) + len(right_xi_candidates))
            assert portion_of_existing_xi_terms < 1, f"Expected portion of existing xi terms to be smaller than 1 but got {portion_of_existing_xi_terms}."

            # if less than self.upper_bound_for_new_xi_pairs combinations are possible, just compute them all
            if len(left_xi_candidates) * len(right_xi_candidates) <= self.upper_bound_for_new_xi_pairs:
                all_possible_candidates = left_xi_candidates.merge(right_xi_candidates, on=share_set)
                considered_candidates = all_possible_candidates[~(all_possible_candidates["exists_x"] & all_possible_candidates["exists_y"])]  # only keep pairs where at least one xi term is new
                self.logger.debug(f"Computed all {len(considered_candidates)} possible pairs of xi terms with at least one existing under the shared set {share_set}.")

            else:

                self.logger.debug(f"Cannot effectively compute all pairs of left and right even under a shared set. Sampling candidates instead.")
            
                # sample xi pairs consisting exclusively of new xi terms
                num_of_pairs_with_only_new_xi_terms = int(self.upper_bound_for_new_xi_pairs * (1 - portion_of_existing_xi_terms))
                assert num_of_pairs_with_only_new_xi_terms > 0
                left_xi_candidates_that_are_new = left_xi_candidates[left_xi_candidates["exists"] == False]
                right_xi_candidates_that_are_new = right_xi_candidates[right_xi_candidates["exists"] == False]
                considered_candidates = left_xi_candidates_that_are_new.merge(right_xi_candidates_that_are_new, on=share_set)
                if min(num_of_pairs_with_only_new_xi_terms, self.upper_bound_for_new_xi_pairs) < len(considered_candidates):
                    considered_candidates = considered_candidates.sample(
                        min(num_of_pairs_with_only_new_xi_terms, self.upper_bound_for_new_xi_pairs),
                        replace=False,
                        random_state=self.random_state
                    )
                
                num_missing_samples = self.upper_bound_for_new_xi_pairs - len(considered_candidates)
                assert num_missing_samples >= 0, f"Number of missing candidates is negative, which cannot be the case because this means that we already have more samples drawn than necessary."
                if num_missing_samples > 0:
                    
                    num_samples_where_left_is_new = num_missing_samples // 2
                    num_samples_where_right_is_new = num_missing_samples - num_samples_where_left_is_new
                    assert num_missing_samples == num_samples_where_left_is_new + num_samples_where_right_is_new
                    assert self.upper_bound_for_new_xi_pairs == len(considered_candidates) + num_missing_samples

                    cols = self.merged_xi_terms.columns

                    # sample pairs where left xi term is new and right xi term exists
                    right_xi_candidates_that_exist = right_xi_candidates[right_xi_candidates["exists"]]
                    samples_where_left_is_new = pd.concat([
                        left_xi_candidates_that_are_new.rename(columns={c: f"{c}_x" for c in cols}).sample(num_samples_where_left_is_new, replace=True, random_state=self.random_state).reset_index(drop=True),
                        right_xi_candidates_that_exist.rename(columns={c: f"{c}_y" for c in cols}).sample(num_samples_where_left_is_new, replace=True, random_state=self.random_state).reset_index(drop=True)
                    ], axis=1)
                    assert num_samples_where_left_is_new == len(samples_where_left_is_new)

                    # sample pairs where right xi term is new and left xi term exists
                    left_xi_candidates_that_exist = left_xi_candidates[left_xi_candidates["exists"]]
                    samples_where_right_is_new = pd.concat([
                        left_xi_candidates_that_exist.rename(columns={c: f"{c}_x" for c in cols}).sample(num_samples_where_right_is_new, replace=True, random_state=self.random_state).reset_index(drop=True),
                        right_xi_candidates_that_are_new.rename(columns={c: f"{c}_y" for c in cols}).sample(num_samples_where_right_is_new, replace=True, random_state=self.random_state).reset_index(drop=True)
                    ], axis=1)
                    assert num_samples_where_right_is_new == len(samples_where_right_is_new)

                    considered_candidates = pd.concat([considered_candidates, samples_where_left_is_new, samples_where_right_is_new], axis=0)
                assert len(considered_candidates) == self.upper_bound_for_new_xi_pairs, f"Expected {self.upper_bound_for_new_xi_pairs} considered candidates but got {len(considered_candidates)}."
                return considered_candidates
                    
            
            if len(considered_candidates) < self.upper_bound_for_new_xi_pairs:
                return considered_candidates
            else:
                #raise Exception("This code shouldn't be reached, since we already checked that the number of pairs is smaller than the upper bound.")
                if portion_of_existing_xi_terms > 0:
                    num_pairs_with_existing_xi = int(self.upper_bound_for_new_xi_pairs * portion_of_existing_xi_terms)
                    sampled_candidates_with_existing_xi = considered_candidates[considered_candidates["exists_x"] | considered_candidates["exists_y"]].sample(num_pairs_with_existing_xi, replace=len(considered_candidates[considered_candidates["exists_x"] | considered_candidates["exists_y"]]) < num_pairs_with_existing_xi, random_state=self.random_state)
                else:
                    num_pairs_with_existing_xi = 0
                sampled_candidates_without_existing_xi = considered_candidates[~considered_candidates["exists_x"] & ~considered_candidates["exists_y"]].sample(self.upper_bound_for_new_xi_pairs - num_pairs_with_existing_xi, replace=len([~considered_candidates["exists_x"] & ~considered_candidates["exists_y"]]) < self.upper_bound_for_new_xi_pairs - num_pairs_with_existing_xi, random_state=self.random_state)
                if num_pairs_with_existing_xi > 0:
                    out = pd.concat([sampled_candidates_with_existing_xi, sampled_candidates_without_existing_xi])
                else:
                    out = sampled_candidates_without_existing_xi
                return out

        # otherwise just randomly sample pairs of xi terms
        else:

            # first create samples where one is existing and the other is new
            portion_of_existing_xi_terms = np.count_nonzero(self.merged_xi_terms["exists"]) / len(self.merged_xi_terms)
            if portion_of_existing_xi_terms > 0:
                num_pairs_with_existing_xi = int(self.upper_bound_for_new_xi_pairs * portion_of_existing_xi_terms)
            else:
                num_pairs_with_existing_xi = 0
            left_xis = left_xi_candidates[left_xi_candidates["exists"]].sample(num_pairs_with_existing_xi, replace=True, random_state=self.random_state).reset_index(drop=True)
            right_xis = right_xi_candidates[~right_xi_candidates["exists"]].sample(len(left_xis), replace=True).reset_index(drop=True)
            df_xi_pairs_with_existing_one = pd.concat([
                left_xis.rename(columns={c: f"{c}_x" for c in left_xis.columns}),
                right_xis.rename(columns={c: f"{c}_y" for c in right_xis.columns})
            ], axis=1)

            # create samples where both are new
            remaining_samples = self.upper_bound_for_new_xi_pairs - len(df_xi_pairs_with_existing_one)
            left_xis = self.new_xi_terms.sample(remaining_samples, replace=True).reset_index(drop=True)
            right_xis = self.new_xi_terms.sample(remaining_samples, replace=True).reset_index(drop=True)
            df_xi_pairs_with_both_new = pd.concat([
                left_xis.rename(columns={c: f"{c}_x" for c in left_xis.columns}),
                right_xis.rename(columns={c: f"{c}_y" for c in right_xis.columns})
            ], axis=1)
            
            # merge dataframes
            return pd.concat([df_xi_pairs_with_existing_one, df_xi_pairs_with_both_new], axis=0)