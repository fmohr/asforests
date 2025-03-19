import numpy as np
from scipy.special import binom
from asforests.momenter_ import Momenter, MixedMomentBuilder
import itertools as it
import time


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
            rs=None,
            execute_asserts=False
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
        self.execute_asserts = execute_asserts

        # sanity check
        accepted_modes = ["stream", "resample_no_replacement", "resample_with_replacement"]
        if population_mode not in accepted_modes:
            raise ValueError(f"population_mode must be in {accepted_modes} but is {population_mode}")

        # state variables
        self.deviation_matrices = []
        self.n = None
        self.k = None
        self.moment_builder = None
        self.mixed_moment_builder = None

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

        # format check
        if self.n is None:
            self.n, self.k = d.shape
        elif (self.n, self.k) != d.shape:
            raise ValueError(f"Expected gap format is ({self.n}, {self.k}) but observed {d.shape}")
        self.deviation_matrices.append(d)

        # append observations to datasets if the population mode is "stream"
        if self.population_mode == "stream":

            # initialize moment builder
            if self.moment_builder is None:
                self.moment_builder = Momenter(input_dims=(self.k, 1), max_p=2)
                self.mixed_moment_builder = MixedMomentBuilder()

            # update ensemble member mean and variance
            if self.estimate_deviation_mean or self.estimate_deviation_var:
                allowed_observations = self.upper_bound_for_sample_size if self.moment_builder.n is None else max([0, min(self.upper_bound_for_sample_size - self.moment_builder.n)])
                if allowed_observations > 0:
                    self.moment_builder.add_batch(d[:allowed_observations])
                    if self.execute_asserts and allowed_observations >= len(d):
                        assert np.all(np.isclose(self.moment_builder.means_, np.mean(self.deviation_matrices, axis=(0, 1))))
                        if self.t > 1:
                            assert np.all(np.isclose(self.moment_builder.central_moments[1], np.var(self.deviation_matrices, axis=(0, 1))))

            # mixed central moment of degree 4 (for covariance CB)
            if len(self.deviation_matrices) > 1 and self.estimate_deviation_covs:
                for gap_matrix_of_ensemble_member2 in self.deviation_matrices[:-1]:
                    allowed_observations = max([0, self.upper_bound_for_sample_size - self.mixed_moment_builder.n])
                    if allowed_observations > 0:
                        self.mixed_moment_builder.add_observations(d, gap_matrix_of_ensemble_member2, axis=0)

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

        #print(f"Added batch in {int(1000 * (time.time() - start))}ms.")