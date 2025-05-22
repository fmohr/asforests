import pandas as pd
import numpy as np
from scipy.special import binom
from asforests.momenter_ import Momenter, MixedMomentBuilder
import itertools as it
import time
import logging


class EnsemblePerformanceAssessor:
    """
        This is the main class for estimating the ensemble performance curve from individual members
    """

    def __init__(
            self,
            upper_bound_for_sample_size,
            population_mode,
            estimate_deviation_mean=True,
            estimate_deviation_var=True,
            estimate_deviation_covs=True,
            estimate_performance_var_for_iid_case=True,
            estimate_performance_var_for_conditional_case=True,
            max_number_of_xi_terms_to_include_in_update=10**5,
            rs=None,
            execute_asserts=False,
            enable_asserts=False, # only for debug mode since this slows down the code
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
        self.upper_bound_for_sample_size = upper_bound_for_sample_size
        if rs is None:
            rs = np.random.RandomState()
        self.rs = rs
        self.estimate_deviation_mean = estimate_deviation_mean
        self.estimate_deviation_var = estimate_deviation_var
        self.estimate_deviation_covs = estimate_deviation_covs
        self.estimate_performance_var_for_iid_case = estimate_performance_var_for_iid_case
        self.estimate_performance_var_for_conditional_case = estimate_performance_var_for_conditional_case
        self.execute_asserts = execute_asserts
        self.enable_asserts = enable_asserts
        self.max_number_of_xi_terms_to_include_in_update = max_number_of_xi_terms_to_include_in_update
        self.logger = logging.getLogger("EnsemblePerformanceEstimator") if logger is None else logger

        # sanity check
        accepted_modes = ["stream", "resample_no_replacement", "resample_with_replacement"]
        if population_mode not in accepted_modes:
            raise ValueError(f"population_mode must be in {accepted_modes} but is {population_mode}")

        # state variables
        self.deviation_matrices = []
        self.masks_for_valid_instances = []
        self.n = None
        self.k = None
        self.moment_builder = None
        self.mixed_moment_builder = None
        self.mixed_moment_builders_for_conditional_xi_covs = None
        self.mixed_moment_builders_for_iid_xi_covs = None
        self.xi_database = None
        self.num_included_term_pairs = 0
        self.data_points_processed_for_cov_estimate = 0

    @property
    def t(self):
        return len(self.deviation_matrices)

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
        start = time.time()
        self.logger.info(f"Adding deviation matrix of shape {d.shape} to Ensemble Performance Estimator")

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
            if self.estimate_deviation_mean or self.estimate_deviation_var:
                self.logger.info(f"Updating estimates of E[D^1] and V[D^1]")
                allowed_observations = self.upper_bound_for_sample_size if self.moment_builder.n is None else max([0, min(self.upper_bound_for_sample_size - self.moment_builder.n)])
                if allowed_observations > 0:
                    self.moment_builder.add_batch(d[:allowed_observations])
                    if self.enable_asserts:
                        if self.execute_asserts and allowed_observations >= len(d):
                            assert np.all(np.isclose(self.moment_builder.means_, np.mean(self.deviation_matrices, axis=(0, 1))))
                            if self.t > 1:
                                assert np.all(np.isclose(self.moment_builder.central_moments[1], np.var(self.deviation_matrices, axis=(0, 1))))
            else:
                self.logger.info(f"Skipping update of estimates of E[D^1] and V[D^1] since this is not configured")

            # update estimate of Cov[D^1, D^2]
            if self.estimate_deviation_covs and self.upper_bound_for_sample_size > self.mixed_moment_builder.n:
                mask_for_valid_instances_s1 = self.masks_for_valid_instances[-1]
                
                for i, (d_s2, mask_for_valid_instances_s2) in enumerate(zip(self.deviation_matrices, self.masks_for_valid_instances)): # include the new one as well for this
                    allowed_observations = max([0, self.upper_bound_for_sample_size - self.mixed_moment_builder.n])
                    
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
            
            # estimate V[Z_nt]
            if self.estimate_performance_var_for_iid_case or self.estimate_performance_var_for_conditional_case:
                self.update_estimates_of_covs_of_xi_terms_based_on_last_added_deviation_matrix()
            else:
                self.logger.info(f"Not updating estimate of covariance terms for variance estimation since this is not configured.")
                
        # update estimates by resampling
        else:

            # create new moment builders from scratch
            self.moment_builder = Momenter(input_dims=(self.k, 1), max_p=2)
            self.mixed_moment_builder = MixedMomentBuilder()

            # sample entries for the mean and variance estimates
            if self.estimate_deviation_mean or self.estimate_deviation_var:
                deviation_matrices = np.array(self.deviation_matrices)
                observations_unified_across_members = deviation_matrices.reshape((-1, deviation_matrices.shape[-1]))
                if self.population_mode == "resample_no_replacement" and len(observations_unified_across_members) < self.upper_bound_for_sample_size:
                    self.moment_builder.add_batch(observations_unified_across_members)
                else:
                    indices = [int(i) for i in self.rs.choice(
                        range(len(observations_unified_across_members)),
                        size=self.upper_bound_for_sample_size,
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
                        if cnt >= self.upper_bound_for_sample_size:
                            break
                elif self.population_mode == "resample_with_replacement":
                    instance_indices = self.rs.choice(range(self.n), size=self.upper_bound_for_sample_size, replace=True)
                    possible_pairs = list(it.combinations(range(len(self.deviation_matrices)), 2))
                    pair_indices = self.rs.choice(
                        range(len(possible_pairs)),
                        size=self.upper_bound_for_sample_size,
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

    def update_estimates_of_covs_of_xi_terms_based_on_last_added_deviation_matrix(self):

        if self.num_included_term_pairs > self.max_number_of_xi_terms_to_include_in_update:
            self.logger.info(f"Reached maximum number of estimates, ignoring new data.")
            return
        
        self.logger.info(f"Updating estimate of covariance terms for variance estimation.")

        # initialize moment builders
        if self.mixed_moment_builders_for_conditional_xi_covs is None:
            self.logger.debug("Initializing MixedMomentBuilders for conditional covs")
            self.mixed_moment_builders_for_conditional_xi_covs = np.array([
                [
                    [
                        MixedMomentBuilder()
                        for _ in range(7)
                    ]
                    for i1 in range(self.n)
                ]
                for i2 in range(self.n)
            ])
        
        if self.mixed_moment_builders_for_iid_xi_covs is None:
            self.logger.debug("Initializing MixedMomentBuilders for iid covs")
            self.mixed_moment_builders_for_iid_xi_covs = np.array([
                [
                    MixedMomentBuilder()
                    for predictor_pairs in ["1111", "1212", "1112", "1122", "1233", "1213", "1234"]
                ]
                for instance_pair in ["11", "12"]
            ])
            self.xi_terms = None
        
        # compute new xi-terms
        new_xi_terms = []
        n, t = self.n, self.t
        self.logger.debug(f"Computing {n} new xi-terms")
        for i in range(n):
            other_s = self.t - 1
            if np.any(np.isnan(self.deviation_matrices[other_s][i])):
                continue
            for s in range(other_s + 1):
                if np.any(np.isnan(self.deviation_matrices[s][i])):
                    continue
                xi = np.dot(self.deviation_matrices[s][i], self.deviation_matrices[other_s][i])
                new_xi_terms.append((i, s, other_s, xi))
                if s != other_s:
                    new_xi_terms.append((i, other_s, s, xi))
        df_new_xi_terms = pd.DataFrame(new_xi_terms, columns=["i", "s1", "s2", "xi"])
        df_new_xi_terms["same_member"] = df_new_xi_terms["s1"] == df_new_xi_terms["s2"]
        df_new_xi_terms["diff_member"] = ~df_new_xi_terms["same_member"]
        self.logger.info(f"Identified {len(df_new_xi_terms)} new xi-terms for the estimation. Creating combinations of those to estimate cov terms.")

        # create table of all relevant xi terms, combining all seen (old and new) with the new ones    
        if self.xi_terms is None:
            if len(df_new_xi_terms)**2 > self.upper_bound_for_sample_size:
                raise ValueError(f"Cannot create even initial xi-terms since these would be {len(df_new_xi_terms)**2} but the upper bound for the sample size is {self.upper_bound_for_sample_size}")
            df_full = df_new_xi_terms.merge(df_new_xi_terms, how="cross")
        else:
            num_of_new_datapoints = (2 * len(self.xi_terms) + len(df_new_xi_terms)) * len(df_new_xi_terms)
            if num_of_new_datapoints <= self.upper_bound_for_sample_size:
                self.logger.info(f"Computing {num_of_new_datapoints} new datapoints to support covariance estimates.")
                df_full = pd.concat([
                    self.xi_terms.merge(df_new_xi_terms, how="cross"),
                    df_new_xi_terms.merge(self.xi_terms, how="cross"),
                    df_new_xi_terms.merge(df_new_xi_terms, how="cross")
                ])
            else:

                # first create only xi-pairs where the instance is identical (since there are not so many of those)
                df_full = pd.concat([
                    self.xi_terms.merge(df_new_xi_terms, on="i"),
                    df_new_xi_terms.merge(self.xi_terms, on="i"),
                    df_new_xi_terms.merge(df_new_xi_terms, on="i")
                ])
                df_full["i_x"] = df_full["i"]
                df_full["i_y"] = df_full["i"]
                df_full.drop(columns="i", inplace=True)
                self.logger.info(f"Combined each of the {len(df_new_xi_terms)} new xi terms twice with each existing one on a matching instance and once with each other new xi-term on a matching instance.")
                self.logger.warning(f"Updating cov estimates not using all {num_of_new_datapoints} but only {len(df_full)} additional xi-terms for covariance estimation of V[Z_nt] and V[Z_nt|D_val] since the upper bound is {self.upper_bound_for_sample_size} and we have no method yet to effectively select xi-terms.")

                # if the instance is not constrained to be equal, create a non-redundant sample of admissible size for random combinations of ensemble members
                num_resamples = self.upper_bound_for_sample_size - len(df_full)
                if num_resamples > 0:
                    df_full_extension = pd.concat([
                        self.xi_terms.merge(df_new_xi_terms, how="cross"),
                        df_new_xi_terms.merge(self.xi_terms, how="cross"),
                        df_new_xi_terms.merge(df_new_xi_terms, how="cross")
                    ])
                    df_full_extension = df_full_extension.sample(num_resamples)
                    self.logger.debug(f"Filling up with {len(df_full_extension)} xi pairs of unequal instances.")
                    df_full = pd.concat([df_full, df_full_extension])

            
        self.logger.info(f"Created {len(df_full)} data points to add for covariance estimations. Checking sanity.")
        assert not np.any(np.isnan(df_full[["xi_x", "xi_y"]].values))
        self.logger.info(f"No nan entries found. Now updating covariance estimates")
        self.data_points_processed_for_cov_estimate += len(df_full)
        
        # run over the 7 cases
        # note that different indices do not REQUIRE but only ALLOW that the ensemble members are different!
        same_in_left = df_full["s1_x"] == df_full["s2_x"]
        same_in_right = df_full["s1_y"] == df_full["s2_y"]
        first_shared = df_full["s1_x"] == df_full["s1_y"]
        for c, mask in enumerate([
            same_in_left & same_in_right & first_shared, # case 11,11
            first_shared & (df_full["s2_x"] == df_full["s2_y"]), # case 12,12
            same_in_left & first_shared, # case 11,12
            same_in_left & same_in_right, # case 11,22
            same_in_right, # case 12,33
            first_shared, # case 12,13
            np.ones(len(df_full)).astype(bool) # case 12,34
        ]):
            
            self.logger.debug(f"Applying mask to dataframe to determine the relevant rows for the case.")
            df_case = df_full[mask]
            self.logger.debug(f"Identified {len(df_case)} data points to update covariances for case {c}.")
            assert not np.any(np.isnan(df_case[["xi_x", "xi_y"]].values))

            if len(df_case) <= 1:
                self.logger.warning(f"Skipping update of covariances for case {c} since less or equal than on data points are available, but we need at least two.")
                continue
            
            # add data for iid case (make two sub-cases for shared or differing instance)
            if self.estimate_performance_var_for_iid_case:
                t_start_update = time.time()
                equal_instance_mask = df_case["i_x"] == df_case["i_y"]

                # first add observations for the case of equal instances
                if np.count_nonzero(equal_instance_mask) > 1:
                    self.mixed_moment_builders_for_iid_xi_covs[0, c].add_observations(df_case[equal_instance_mask]["xi_x"], df_case[equal_instance_mask]["xi_y"])
                else:
                    self.logger.warning("not updating IID cov estimates for equal instances since we have not data for at least two xi terms.")
                
                # now add observations for the case of unequal instances, but skip 3 of the cases, which we know have 0 values by theory
                if c not in [3, 4, 6]:
                    if len(df_case) > self.max_number_of_xi_terms_to_include_in_update:
                        df_case = df_case.sample(replace=False, n=self.max_number_of_xi_terms_to_include_in_update)
                        self.logger.debug(f"Using {len(df_case)} data points to update covariances for case {c}.")

                    self.mixed_moment_builders_for_iid_xi_covs[1, c].add_observations(df_case["xi_x"], df_case["xi_y"])
                    t_end_update = time.time()
                    self.logger.debug(f"Update of covs for iid case took {np.round(t_end_update - t_start_update, 6)}s")

            # add data for conditional case
            if self.estimate_performance_var_for_conditional_case:
                t_start_update = time.time()
                for (i1, i2), df_sub in df_case.groupby(["i_x", "i_y"]):
                    self.mixed_moment_builders_for_conditional_xi_covs[i1, i2, c].add_observations(df_sub["xi_x"], df_sub["xi_y"])
                t_end_update = time.time()
                self.logger.debug(f"Update of covs for conditional case took {np.round(t_end_update - t_start_update, 6)}s")

        self.num_included_term_pairs += len(df_full)
        self.xi_terms = df_new_xi_terms if self.xi_terms is None else pd.concat([self.xi_terms, df_new_xi_terms], ignore_index=True)
        self.logger.info(f"Finished update of covariance estimates.")
