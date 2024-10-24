from typing import Callable

import numpy as np
import scipy.stats as stats
import logging
import time
from .momenter_ import Momenter

import pandas as pd
import random
import scipy.special
import itertools as it


class DummyOutputSupplier(Callable):

    def __init__(self, history):
        self.history= history
        self.index = 0
    
    def __call__(self, num_entries):
        out = self.history[self.index:self.index + num_entries]
        self.index += num_entries
        return out


def upper_bound_for_variance_from_momenter(momenter: Momenter, alpha):
    m = momenter.avg_num_samples
    tau = stats.t.ppf(1 - (1 - alpha), df=m - 1)
    moments = momenter.moments
    return moments[1] + tau * np.sqrt(np.maximum(0, (moments[3] - moments[1]**2)) / m)


def upper_bound_for_variance(sample, alpha):
    momenter = Momenter()
    momenter.add_batch(sample)
    return upper_bound_for_variance_from_momenter(momenter, alpha)


class EnsembleGrower:
    
    def __init__(
            self,
            output_supplier,
            y_one_hot,
            oob,
            tolerances,
            n=None,
            max_size=None,
            betas=None,
            ensemble_size_for_initial_estimate_of_vzt=4,
            max_sample_size_for_vzt_estimation=10**5,
            var_zt_estimation_strategy="bound",
            logger=None,
            callbacks=[]
    ):

        # setup variables
        self.output_supplier = output_supplier
        self.y_one_hot = y_one_hot
        self.n = n if n is not None else self.y_one_hot.shape[0]
        self.k = self.y_one_hot.shape[1]
        self.oob = oob
        t_start = time.time()
        sample_estimate_for_vzt_pairs = np.array([t * (t - 1) / 2 * (self.y_one_hot.shape[0] * (0.366**2 if oob else 1)) for t in range(1, 1000)])
        sample_estimate_for_vzt_triplets = np.array(
            [t * (t - 1) * (t - 2) / 6 * (self.y_one_hot.shape[0] * (0.366 ** 3 if oob else 1)) for t in range(1, 1000)]
        )
        self.ensemble_size_required_to_estimate_vzt = max([
            np.min(np.where(sample_estimate_for_vzt_pairs >= max_sample_size_for_vzt_estimation)[0]),
            np.min(np.where(sample_estimate_for_vzt_triplets >= max_sample_size_for_vzt_estimation)[0])
        ]) + 1
        self.ensemble_size_for_initial_estimate_of_vzt = ensemble_size_for_initial_estimate_of_vzt
        self.min_size = min([
            self.ensemble_size_required_to_estimate_vzt,
            self.ensemble_size_for_initial_estimate_of_vzt
        ])
        self.final_estimate_created = False

        accepted_var_zt_estimation_strategies = ["bound", "sample"]
        if var_zt_estimation_strategy not in accepted_var_zt_estimation_strategies:
            raise ValueError(
                f"var_zt_estimation_strategy must be one of {accepted_var_zt_estimation_strategies} "
                f"but is {var_zt_estimation_strategy}."
            )
        self.var_zt_estimation_strategy = var_zt_estimation_strategy
        self.max_sample_size_for_vzt_estimation = max_sample_size_for_vzt_estimation

        self.output_diffs_per_ensemble_member = []

        self.tolerances = tolerances
        self.alphas = np.array([alpha for _, alpha in self.tolerances])
        self.epsilons = np.array([eps for eps, _ in self.tolerances])
        self.max_size = max_size
        self.betas = betas if betas is not None else np.ones(5) / 5
        self.logger = logger if logger is not None else logging.getLogger("asbe.grower")

        # compute minimum size
        ### TODO COMPUTE THIS

        # state variables
        self.covariances_relevant_for_var_of_zt = None
        self.counts = np.zeros((self.n, self.k))
        self.c_history, self.v_history, self.size = [], [], 0
        self.momenter_outputs = Momenter()
        self.momenter_lambdas_equals = Momenter()
        self.momenter_lambdas_unequals = Momenter()
        self.step_times = []
        self.callbacks = callbacks

    def grow(self):
        
        # now start training
        self.logger.info(f"Start training.")
        while self.step() > 0:
            pass
        self.logger.info("Forest grown completely. Stopping routine!")

    def get_lambdas(self, equals):
        deltas = np.array(self.output_diffs_per_ensemble_member)

        def get_lambda(delta_1, delta_2):
            if delta_1.shape != delta_2.shape:
                raise ValueError()
            return np.sum(delta_1 * delta_2, axis=1)

        if equals:
            lambdas = np.array([get_lambda(delta_s, delta_s) for delta_s in deltas])
        else:
            lambdas = np.array([
                get_lambda(delta_s, delta_sp)
                for i, delta_s in enumerate(deltas) for j, delta_sp in enumerate(deltas[i+1:], start=i+1)
            ])
        return lambdas

    """
    def get_unique_lambda_combos(self, n_combos, k):

        deltas = np.array(self.output_diffs_per_ensemble_member)

        def get_lambda(delta_1, delta_2):
            if delta_1.shape != delta_2.shape:
                raise ValueError()
            return np.sum(delta_1 * delta_2, axis=1)

        def get_unique_random_k_subsets(s, k):

            s = list(s)

            # existing subsets
            num_existing_subsets = int(scipy.special.binom(len(s), k))

            if num_existing_subsets > 10 ** 4:
                unique_subsets = set()

                while len(unique_subsets) < num_existing_subsets:
                    subset = tuple(sorted(random.sample(s, k)))
                    yield subset

            else:
                all_combos = list(it.combinations(s, k))
                full_sample = random.sample(all_combos, len(all_combos))
                for s in full_sample:
                    yield s

        ensemble_member_indices = list(range(len(deltas)))
        rows = []
        for index_combo in get_unique_random_k_subsets(ensemble_member_indices, k=k):
            if k == 2:
                delta1 = deltas[index_combo[0]]
                delta2 = deltas[index_combo[1]]
                lambda1 = get_lambda(delta1, delta1)
                lambda2 = get_lambda(delta1, delta2)
            if k == 3:
                delta1 = deltas[index_combo[0]]
                delta2 = deltas[index_combo[1]]
                delta3 = deltas[index_combo[2]]
                lambda1 = get_lambda(delta1, delta2)
                lambda2 = get_lambda(delta1, delta3)

            new_rows = np.array([lambda1, lambda2]).T
            new_rows = new_rows[~np.isnan(new_rows).any(axis=1)]
            rows.extend(new_rows)
            if len(rows) > n_combos:
                break
        rows = np.array(rows)
        return rows

    def estimate_and_memorize_covariances_for_zt(self):
        t_start = time.time()
        deltas = np.array(self.output_diffs_per_ensemble_member)

        def get_lambda(delta_1, delta_2):
            if delta_1.shape != delta_2.shape:
                raise ValueError()
            return np.sum(delta_1 * delta_2, axis=1)

        # estimate V[l_i^ss]
        lambdas = np.array([get_lambda(delta_s, delta_s) for delta_s in deltas])
        V_lambda_ss = np.nanvar(lambdas.flatten())
        assert not np.isnan(V_lambda_ss), "Estimate of v_lambda_ss is nan"

        # estimate V[l_i^ss']
        lambdas = np.array([
            get_lambda(delta_s, delta_sp)
            for i, delta_s in enumerate(deltas) for j, delta_sp in enumerate(deltas) if i != j
        ])
        V_lambda_ssp = np.nanvar(lambdas.flatten())

        # estimate cov[l_i^ss, l_i^ss']
        lambda_pairs_s_sp = self.get_unique_lambda_combos(n_combos=self.max_sample_size_for_vzt_estimation, k=2)
        cov_lambda_s_sp = np.cov(lambda_pairs_s_sp, rowvar=False)[0, 1]

        # estimate cov[l_i^ss', l_i^ss'']
        lambda_pairs_s_sp_spp = self.get_unique_lambda_combos(n_combos=self.max_sample_size_for_vzt_estimation, k=3)
        cov_lambda_s_sp_spp = np.cov(lambda_pairs_s_sp_spp, rowvar=False)[0, 1]

        runtime = time.time() - t_start

        self.covariances_relevant_for_var_of_zt = {
            "basis": self.size,
            "sample_size_s_sp": len(lambda_pairs_s_sp),
            "sample_size_s_sp_spp": len(lambda_pairs_s_sp_spp),
            "v_lambda_ss": V_lambda_ss,
            "v_lambda_ssp": V_lambda_ssp,
            "cov_lambda_s_sp": cov_lambda_s_sp,
            "cov_lambda_s_sp_spp": cov_lambda_s_sp_spp
        }
        print(f"Total runtime for estimation: {runtime}.")
        print(self.covariances_relevant_for_var_of_zt)

    def get_estimate_for_performance_variance(self, t, n):

        # if we have only three trees, use the general and data-independent bound
        if self.var_zt_estimation_strategy == "bound" or self.size < self.ensemble_size_required_to_estimate_vzt:
            return 4 / (n * t)

        # create ingredients to estimate V[Z_t]
        if self.covariances_relevant_for_var_of_zt is None or (
                self.size >= self.ensemble_size_required_to_estimate_vzt and
                not self.final_estimate_created
        ):
            self.estimate_and_memorize_covariances_for_zt()
            if self.size >= self.ensemble_size_required_to_estimate_vzt:
                self.final_estimate_created = True
            assert not np.any(np.isnan(list(self.covariances_relevant_for_var_of_zt.values()))), f"Nan values is {self.covariances_relevant_for_var_of_zt}"

        # compute estimate for V[Z_t]
        var_zt = (
                self.covariances_relevant_for_var_of_zt["v_lambda_ss"] +
                self.covariances_relevant_for_var_of_zt["v_lambda_ssp"] * 2 * (t - 1) +
                self.covariances_relevant_for_var_of_zt["cov_lambda_s_sp"] * 4 * (t - 1) +
                self.covariances_relevant_for_var_of_zt["cov_lambda_s_sp_spp"] * 4 * (t - 1) * (t - 2)
        ) / (t ** 3 * n)

        if var_zt < 0:
            return 4 / (n * t)

        return var_zt

    def get_bound_for_deviation_from_mean(self, alpha, t, n):
        estimated_var_at_t = self.get_estimate_for_performance_variance(t, n)
        assert not np.isnan(estimated_var_at_t), f"Estimate of V[Z_t] is nan for size t={t}"
        assert estimated_var_at_t >= 0, f"Estimate of V[Z_t] is negative ({estimated_var_at_t}) for size t={t}"
        if estimated_var_at_t == 0:
            return 0
        return stats.norm.ppf(
            q=1 - self.beta * (1 - alpha),
            loc=0,
            scale=np.sqrt(estimated_var_at_t)
        )

    def get_bound_for_expected_gap(self, alpha, t):
        v_hat = self.v_history[-1] if self.size > 0 else 1
        c_hat = self.c_history[-1] if self.size > 0 else 1
        tau = stats.t.ppf(1 - (1 - self.beta) * (1 - alpha) / self.k, df=t - 1)
        return (v_hat + tau * c_hat / np.sqrt(t * (0.366 if self.oob else 1))) / t

    def get_upper_confidence_bound_for_performance_gap(self, t, n, alpha):
        kappa = stats.norm.ppf(q=1 - self.beta * (1 - alpha), loc=0, scale=1)
        tau = stats.t.ppf(1 - (1 - alpha), df=self.size - 1)

        moments = self.momenter.moments
        self.ubv_tree_predictions = ((moments[1] + tau * (np.sqrt(np.maximum(0, moments[3] - moments[1] ** 2) / t))).sum() / self.n) / t
        self.ubv_lambda_equal = upper_bound_for_variance(self.get_lambdas(True).reshape(-1, 1), alpha)[0]
        self.ubv_lambda_uneqal = upper_bound_for_variance(self.get_lambdas(False).reshape(-1, 1), alpha)[0]

        ub1 = self.ubv_tree_predictions

        t0 = self.size * 0.366
        ub2 = kappa * np.sqrt(
            (self.ubv_lambda_equal + 4 * (t-1) * ((t - 2) * self.ubv_lambda_uneqal + np.sqrt(self.ubv_lambda_equal * self.ubv_lambda_uneqal)))
            /
            (self.n * t**3 * t0)
        )
        ub = ub1 + ub2
        print(t, ub1, ub2)
        return ub

        bound_1 = self.get_bound_for_deviation_from_mean(alpha=alpha, t=t, n=n)
        bound_2 = self.get_bound_for_expected_gap(alpha=alpha, t=t)
        assert not np.isnan(bound_1), f"The upper bound for the deviation of Z_t to its mean is nan for t = {t}"
        assert not np.isnan(bound_2), f"The upper bound for the gap of E[Z_t] and the limit performance is nan for t ={t}"
        return bound_1 + bound_2
        """

    def estimate_performance_gap(
            self,
            t,
            n_test,
            alpha,
            betas,
            include_epistemic_bound_of_mu=True,
            include_aleatoric_bound_of_sigma=True,
            include_epistemic_bound_of_sigma=True
    ):

        # estimate mu
        if include_epistemic_bound_of_mu:
            v = upper_bound_for_variance_from_momenter(self.momenter_outputs, alpha=1 - (betas[0] * (1 - alpha)))
        else:
            v = self.momenter_outputs.moments[1]
        mu = np.nansum(v, axis=1).mean() / t
        assert not np.isnan(mu), "Estimate of mu is nan!"

        # estimate sigma^2, but only if it will be used
        if include_aleatoric_bound_of_sigma:
            if include_epistemic_bound_of_sigma:
                b1 = upper_bound_for_variance_from_momenter(self.momenter_lambdas_equals, alpha=1 - (betas[2] * (1 - alpha)))[0]
                b2 = upper_bound_for_variance_from_momenter(self.momenter_lambdas_unequals, alpha=1 - (betas[2] * (1 - alpha)))[0]
                b3 = np.sqrt(
                    upper_bound_for_variance_from_momenter(self.momenter_lambdas_equals, alpha=1 - np.sqrt((betas[3] * (1 - alpha))))[0]
                    *
                    upper_bound_for_variance_from_momenter(self.momenter_lambdas_unequals,
                                                           alpha=1 - np.sqrt(betas[3] * (1 - alpha)))[0]
                )
            else:
                b1 = self.momenter_lambdas_equals.moments[1]
                b2 = self.momenter_lambdas_unequals.moments[1]
                b3 = np.sqrt(b1 * b2)

            sigma = (b1 + 4 * (t - 1) * ((t - 2) * b2 + b3)) / (n_test * t**3)
        else:
            if include_epistemic_bound_of_sigma:
                raise ValueError(f"It does not make sense to include the epistemic uncertainty on sigma^2 if the aleatoric uncertainty is not considered!")
            sigma = 0

        # if aleatoric uncertainty is quantified, apply it; otherwise just return the mean
        if sigma > 0:
            kappa = stats.norm.ppf(
                q=1 - betas[1] * (1 - alpha),
                loc=0,
                scale=1
            )
            return mu + kappa * np.sqrt(sigma / n_test)
        else:
            return mu

    def get_gaps(self, t):

        """
        computes the delta as defined in Corollary 5. t0 is given implicitly by the size of the current ensemble.
        :param t:
        :return:
        """
        gaps = np.zeros(len(self.alphas))
        for i, alpha in enumerate(self.alphas):
            gap = self.estimate_performance_gap(
                t=t,
                n_test=self.n,
                alpha=alpha,
                betas=self.betas
            )
            gaps[i] = gap
        return gaps

    def get_min_t_that_satisfies_conditions(self, max_additional_members=64):
        t = np.max([self.min_size, self.size])
        while np.any(self.get_gaps(t) > self.epsilons):
            t += 1
            if t >= self.size + max_additional_members:
                break
        return t

    def step(self, t_p=None):
        t_start = time.time()
        if self.max_size is not None and self.size >= self.max_size:
            return 0

        self.logger.debug(f"Starting Iteration.")

        # determine number of additional members
        if t_p is None:
            if self.size == 0:
                t_p = self.min_size
            else:
                t_p = self.get_min_t_that_satisfies_conditions() - self.size
                if (
                        self.var_zt_estimation_strategy != "bound" and
                        self.size < self.ensemble_size_required_to_estimate_vzt < self.size + t_p
                ):
                    t_p = self.ensemble_size_required_to_estimate_vzt - self.size
        if t_p <= 0:
            return 0

        # train new base learners and update model
        new_outputs = self.output_supplier(t_p)

        # determine diffs to target (necessary for estimation of V[Z_t])
        deltas = new_outputs - self.y_one_hot

        self.size += len(new_outputs)

        # remember these deltas for computations of lambdas
        self.output_diffs_per_ensemble_member.extend(deltas)

        # update momenters
        self.momenter_outputs.add_batch(new_outputs)
        self.momenter_lambdas_equals.add_batch(np.sum(deltas * deltas, axis=2).reshape(-1, 1))
        for s1, deltas_s1 in enumerate(deltas):
            for s2, deltas_s2 in enumerate(self.output_diffs_per_ensemble_member):
                if s1 != len(self.output_diffs_per_ensemble_member) - s2:
                    self.momenter_lambdas_unequals.add_batch(np.sum(deltas_s1 * deltas_s2, axis=1).reshape(-1, 1))

        for momenter in [self.momenter_outputs, self.momenter_lambdas_equals, self.momenter_lambdas_unequals]:
            assert not np.any(np.isnan(momenter.moments)), "There are nan entries in the moments!"

        if False:
            moments = self.momenter_outputs.moments
            assert not np.any(np.isnan(moments)), "There are nan entries in the moments!"

            self.v_history.append(moments[1].sum(axis=1).mean())

            """
                when computing c, we need to take into account that we must not simple form the average of sums.
                this is because, in the OOB case, we actually only have 36.6% of the observations available.
            """
            self.c_history.append(np.sqrt(np.maximum(0, moments[3] - moments[1] ** 2)).sum() / self.n)

        self.step_times.append(time.time() - t_start)
        for cb in self.callbacks:
            cb(self)
        return len(new_outputs)
