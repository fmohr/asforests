import scipy.stats
from scipy.special import binom
import numpy as np
from tqdm import tqdm
import warnings
import time


class Momenter:

    def __init__(self, input_dims=None, max_p=4, keep_history=False):

        self.input_dims = input_dims
        self.max_p = max_p
        self.keep_history = keep_history

        # state variables
        self.n = 0
        self.counts = None
        self.M = None
        self.means_ = None
        self.moments_ = None
        self.means_over_time = []
        self.moments_over_time_ = []
        if input_dims is not None:
            self.reset_()

    def reset_(self):
        self.counts = np.zeros(self.input_dims)
        self.M = np.zeros(tuple([self.max_p] + (list(self.input_dims))))

    def add_entry(self, new_obs):

        if not isinstance(new_obs, np.ndarray):
            new_obs = np.array(new_obs)

        if self.input_dims is None:
            self.input_dims = new_obs.shape
            self.reset_()

        elif new_obs.shape != self.input_dims:
            raise ValueError(f"Expected shape {self.input_dims} but is {new_obs.shape}")

        # compute updated moment
        mask = ~np.isnan(new_obs)
        self.counts += mask

        M_masked = self.M[:, mask]

        N = self.counts[mask]
        mu = self.means_.copy() if self.means_ is not None else np.zeros(self.input_dims)
        delta = new_obs[mask] - mu[mask]
        for p in range(1, self.max_p + 1):

            t1 = M_masked[p - 1]

            t2 = ((N - 1) / (-N) ** p + ((N - 1) / N) ** p) * delta ** p

            t3 = 0
            for k in range(1, p - 1):
                t3 += binom(p, k) * M_masked[p - k - 1] * (-delta / N) ** k

            self.M[p - 1][mask] = t1 + t2 + t3

        # update means
        mu[mask] = ((self.counts[mask] - 1) * mu[mask] + new_obs[mask]) / self.counts[mask]

        # store updated means and moments
        self.means_ = mu
        self.moments_ = self.M / np.maximum(1, self.counts)
        if self.keep_history:
            self.moments_over_time_.append(self.moments_)
            self.means_over_time.append(self.means_)

    def add_batch_one_by_one(self, new_obs_batch, axis=0):
        if axis != 0:
            order = list(range(len(new_obs_batch.shape)))
            order[0] = axis
            for i in range(1, axis + 1):
                order[i] -= 1
            order = tuple(order)
            new_obs_batch = new_obs_batch.transpose(order)

        for obs in new_obs_batch:
            self.add_entry(obs)

    def get_mean_and_moments_for_batch(self, batch, axis=0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(batch, axis=axis)
            mean[np.isnan(mean)] = 0
            centralized_batch = (batch - mean)
            moments = [np.zeros(mean.shape)]
            even_relevant = np.nansum(batch, axis=axis) > 0

            for m in range(2, self.max_p + 1):
                even = m % 2 == 0
                if even:
                    moment = np.zeros(even_relevant.shape)
                    moment[even_relevant] = np.nanmean(centralized_batch[:, even_relevant]**m, axis=0)
                else:
                    moment = np.nanmean(centralized_batch ** m, axis=0)
                moment[np.isnan(moment)] = 0
                if m == 2:
                    even_relevant = moment > 0
                moments.append(moment)
            return mean, np.array(moments)

    def add_batch(self, new_obs_batch, axis=0):
        N_B = np.sum(~np.isnan(new_obs_batch), axis=axis)
        start = time.time()
        mu_B, moments = self.get_mean_and_moments_for_batch(new_obs_batch, axis=axis)
        M_B = N_B * moments

        if np.all(self.n == 0):
            self.n = N_B
            self.moments_ = moments
            self.M = M_B
            self.means_ = mu_B
        else:
            M_A = self.M
            N_A = self.n
            mu_A = self.means_
            delta = mu_B - mu_A

            N = N_A + N_B

            new_M = M_A + M_B
            updated_indices = N_B > 0
            for i in range(self.max_p):
                p = i + 1
                t2 = N_A[updated_indices] * (-N_B[updated_indices]/N[updated_indices] * delta[updated_indices])**p
                t3 = N_B[updated_indices] * (N_A[updated_indices]/N[updated_indices] * delta[updated_indices])**p
                t4 = np.sum([
                    scipy.special.binom(p, k) * (
                        M_A[p - k - 1][updated_indices] * (-N_B[updated_indices]/N[updated_indices] * delta[updated_indices])**k +
                        M_B[p - k - 1][updated_indices] * (N_A[updated_indices]/N[updated_indices] * delta[updated_indices])**k
                    )
                    for k in range(1, p - 1)], axis=0)
                new_M[i][updated_indices] += t2 + t3 + t4
            self.M[:, updated_indices] = new_M[:, updated_indices]
            self.moments_[:, updated_indices] = self.M[:, updated_indices] / N[updated_indices]
            self.means_[updated_indices] = (N_A[updated_indices] * mu_A[updated_indices] + N_B[updated_indices] * mu_B[updated_indices]) / N[updated_indices]
            self.n[updated_indices] = N[updated_indices]

    @property
    def moments(self):
        return np.array(self.moments_)

    @property
    def moments_over_time(self):
        if not self.keep_history:
            raise ValueError("Momenter not configured to keep moments over time.")
        return np.array(self.moments_over_time_).transpose(1, 0, *range(2, len(self.input_dims) + 2))

    @property
    def avg_num_samples(self):
        return np.mean(self.n) if self.n is not None else 0
