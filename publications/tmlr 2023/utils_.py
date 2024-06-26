import itertools as it
from tqdm import tqdm
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import json
import zlib
from asforests.momenter_ import Momenter
import gzip


class Analyzer:

    def __init__(self,
                 openmlid,
                 seed,
                 prob_history_oob,
                 prob_history_val,
                 Y_train,
                 Y_val,
                 times_fit,
                 times_predict_train,
                 times_predict_val,
                 times_update,
                 scores_of_single_trees,
                 single_tree_scores_mean_ests,
                 single_tree_scores_std_ests,
                 scores_of_forests,
                 correction_terms_for_t1_per_time,
                 confidence_term_for_correction_term_per_time,
                 **kwargs
                 ):
        self.openmlid = openmlid
        self.seed = seed

        self.prob_history_oob = np.array(prob_history_oob)
        self.prob_history_val = np.array(prob_history_val)
        self.Y_train = np.array(Y_train)
        self.Y_val = np.array(Y_val)
        self.num_trees = self.prob_history_oob.shape[0]
        if self.prob_history_oob.shape[0] != self.prob_history_val.shape[0]:
            raise ValueError("Inconsistent tree count!")
        self.times_fit = times_fit
        self.times_predict_train = times_predict_train
        self.times_predict_val = times_predict_val
        self.times_update = times_update

        self.scores_of_single_trees = scores_of_single_trees
        self.single_tree_scores_mean_ests = single_tree_scores_mean_ests
        self.single_tree_scores_std_ests = single_tree_scores_std_ests
        self.scores_of_forests = scores_of_forests
        self.correction_terms_for_t1_per_time = correction_terms_for_t1_per_time
        self.confidence_term_for_correction_term_per_time = confidence_term_for_correction_term_per_time

        # store all the rest in info dict
        self.info = kwargs
        print(self.info.keys())

        # things to compute lazy
        self.ci_histories = {
            "oob": {},
            "val": {}
        }

    @staticmethod
    def for_dataset(openmlid, seed=0, folder="data"):

        # read in data
        with open(f"{folder}/{openmlid}_{seed}.json", "r") as f:
            return Analyzer(**json.load(f))

    def sample_rf_behavior_from_probs(self, probs_orig, Y, forest_size=None):
        if forest_size is None:
            forest_size = probs_orig.shape[0]
        permutation = np.random.choice(range(len(probs_orig)), forest_size, replace=False)
        permuted_probs = probs_orig[permutation]

        sums_per_timestep = np.zeros(Y.shape)
        counts_per_timestep = np.zeros(Y.shape)
        mean_distributions_per_forest_size = []
        for t, probs_in_t in enumerate(permuted_probs):
            mask = ~np.isnan(probs_in_t[:, 0])
            covered_indices = list(np.where(mask)[0])
            sums_per_timestep[covered_indices] += probs_in_t[covered_indices]
            counts_per_timestep += np.tile(mask, (Y.shape[1], 1)).T
            mean_distributions_per_forest_size.append(sums_per_timestep / counts_per_timestep)
        mean_distributions_per_forest_size = np.array(mean_distributions_per_forest_size)

        estimated_limit_performance_per_timestep = np.nanmean(
            ((mean_distributions_per_forest_size[:] - Y) ** 2).sum(axis=2), axis=1)
        brier_score_per_forest_size = np.nanmean(((mean_distributions_per_forest_size[:] - Y) ** 2).sum(axis=2), axis=1)
        return permuted_probs, brier_score_per_forest_size, estimated_limit_performance_per_timestep

    def get_ci_for_expected_performance_at_size_t_based_on_normality(self,
                                                                     forest_size,
                                                                     alpha,
                                                                     sub_forest_size=1,
                                                                     oob=True
                                                                     ):
        key = "oob" if oob else "val"
        trees_to_be_considered_for_ci = self.get_num_trees_used_on_avg_for_oob_estimates_at_forest_size(forest_size) if oob else forest_size  # this is always t, even for OOB
        est_mean = self.single_tree_scores_mean_ests[key][forest_size - 1]
        est_std = self.single_tree_scores_std_ests[key][forest_size - 1]
        ci = stats.norm.interval(alpha, loc=est_mean, scale=est_std / np.sqrt(trees_to_be_considered_for_ci))
        return ci

    def compute_ci_sequence_for_expected_performance_at_size_t_based_on_normality(self, alpha, sub_forest_size=1,
                                                                                  offset=10, oob=True):
        max_forest_size = self.num_trees
        cis = []
        ci_intersection = [-np.inf, np.inf]
        ci_intersection_history = []
        for forest_size in range(offset, max_forest_size + 1):
            ci = self.get_ci_for_expected_performance_at_size_t_based_on_normality(
                forest_size=forest_size,
                alpha=alpha,
                sub_forest_size=sub_forest_size,
                oob=oob
            )
            cis.append(ci)

            ci_intersection[0] = max([ci_intersection[0], ci[0]])
            ci_intersection[1] = min([ci_intersection[1], ci[1]])
            if ci_intersection[0] > ci_intersection[1]:
                # print(f"WARNING: INCONSISTENT CI")
                # raise ValueError("INCONSISTENT")
                pass

            ci_intersection_history.append([i for i in ci_intersection])

        ci_intersection_history = np.array(ci_intersection_history)
        first_key = "oob" if oob else "val"
        second_key = alpha
        self.ci_histories[first_key][second_key] = {
            "orig": cis,
            "intersect": ci_intersection_history
        }
    def get_num_trees_used_on_avg_for_oob_estimates_at_forest_size(self, forest_size):
        return int(np.ceil(forest_size * 0.366))

    def get_num_trees_required_for_stable_correction_term_estimate(self, max_iterations_without_new_max=5,
                                                                   do_discount=True, oob=True):
        new_max_indices = []
        first_plateau_max_index = -1
        cur_max = 0
        cnt_no_new_max = 0
        key = "oob" if oob else "val"
        for t, v in enumerate(self.correction_terms_for_t1_per_time[key], start=1):
            if do_discount:
                v /= t
            if v < cur_max:
                cnt_no_new_max += 1
            else:
                cur_max = v
                cnt_no_new_max = 0

            if cnt_no_new_max >= max_iterations_without_new_max:
                return t
        return np.nan

    def create_ci_plot(self, alpha, eps, ci_offset, oob=True, ax=None):
        raise NotImplementedError

        first_key = "oob" if oob else "val"
        second_key = alpha
        cis = self.ci_histories[first_key][second_key]["orig"]
        ci_intersections = self.ci_histories[first_key][second_key]["intersect"]
        times = range(1, self.prob_history_oob.shape[0] + 1)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        expected_limit_performance = sum(cis[-1]) / 2
        ax.fill_between(times[ci_offset - 1:], [e[0] for e in cis], [e[1] for e in cis], alpha=0.2)
        ax.fill_between(times[ci_offset - 1:], [e[0] for e in ci_intersections], [e[1] for e in ci_intersections],
                        color="red", alpha=0.2)
        ax.fill_between(times, expected_limit_performance - eps, expected_limit_performance + eps, alpha=0.2)
        ax.set_ylim([expected_limit_performance - 5 * eps, expected_limit_performance + 5 * eps])

        if fig is not None:
            plt.show()

    def plot_estimates_of_correction_term(self, axs=None):
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(20, 4))
        else:
            fig = None

        times = np.arange(1, self.num_trees + 1)

        # correction term without t
        ax = axs[0]
        ax.plot(times, self.correction_terms_for_t1_per_time["oob"], label="$\\mathbb{V}_{oob}$")
        ax.plot(times, self.correction_terms_for_t1_per_time["val"], label="$\\mathbb{V}_{val}$")
        x_limit = 150
        for (key, oob), do_discount in it.product(zip(["oob", "val"], [True, False]), [True, False]):
            color = f"C{0 if oob else 1}"
            linestyle = "--" if do_discount else "dotted"
            num_tree_count_for_stable_correction_term_estimate = self.get_num_trees_required_for_stable_correction_term_estimate(
                oob=oob, do_discount=do_discount)
            ax.axvline(num_tree_count_for_stable_correction_term_estimate, linestyle=linestyle, color=color,
                       label="num trees for stable estimate of $\\mathbb{V}_{" + key + "}$ " + (
                           "" if do_discount else "not") + " discounting")
            x_limit = max([x_limit, num_tree_count_for_stable_correction_term_estimate + 10])

        ax.set_xlim([1, x_limit])
        ax.set_xscale("log")

        # correction term with t
        ax = axs[1]
        ax.plot(times, self.correction_terms_for_t1_per_time["oob"] / times, label="$\\mathbb{V}_{oob}$")
        ax.plot(times, self.correction_terms_for_t1_per_time["val"] / times, label="$\\mathbb{V}_{val}$")
        x_limit = 150
        for (key, oob), do_discount in it.product(zip(["oob", "val"], [True, False]), [True, False]):
            color = f"C{0 if oob else 1}"
            linestyle = "--" if do_discount else "dotted"
            num_tree_count_for_stable_correction_term_estimate = self.get_num_trees_required_for_stable_correction_term_estimate(
                oob=oob, do_discount=do_discount)
            ax.axvline(num_tree_count_for_stable_correction_term_estimate, linestyle=linestyle, color=color,
                       label="num trees for stable estimate of $\\mathbb{V}_{" + key + "}$ " + (
                           "" if do_discount else "not") + " discounting")
            x_limit = max([x_limit, num_tree_count_for_stable_correction_term_estimate + 10])

        ax.set_xlim([1, x_limit])
        ax.set_xscale("log")

        if fig is not None:
            ax.legend()
            ax.set_title("Estimates of correction term $\\varphi(1)$ over time.")
            plt.show()

    def plot_gap_over_time(self, alpha, eps, oob, min_trees=2, ax=None):
        if ax is None:
            fig, axs = plt.subplots(2, 1, figsize=(16, 8))
        else:
            fig = None

        key = "oob" if oob else "val"

        times = np.arange(1, self.num_trees + 1)  # here we need to use times for both OOB and validation curves
        num_trees_for_estimates_oob = np.array([
            self.get_num_trees_used_on_avg_for_oob_estimates_at_forest_size(t) for t in times
        ])
        correction_terms_over_time_real_oob = self.correction_terms_for_t1_per_time[key] / num_trees_for_estimates_oob
        correction_terms_over_time_real_estimated_for_val = self.correction_terms_for_t1_per_time[key] / times
        # correction_terms_over_time_omniscient = self.correction_terms_for_t1_per_time[key][-1] / num_trees_for_estimates
        # correction_terms_over_time_pessimistic = (self.Y_train.shape[1] / (4 * num_trees_for_estimates))
        confidence_band_sizes = np.array([(e[1] - e[0]) / 2 for e in self.ci_histories[key][alpha]["orig"]])

        # print(correction_terms_over_time_real[:10])
        # print(correction_terms_over_time_omniscient[:10])

        offset = len(times) - len(confidence_band_sizes)
        gaps_real = confidence_band_sizes + correction_terms_over_time_real_oob[offset:]
        gaps_anticipated_for_val = confidence_band_sizes + correction_terms_over_time_real_estimated_for_val[offset:]
        # gaps_known_variance = confidence_band_sizes + correction_terms_over_time_omniscient[offset:]
        # gaps_pessimistic_estimate = confidence_band_sizes + correction_terms_over_time_pessimistic[offset:]

        # real plot
        max_xlim = 0
        max_gap = np.max([np.max(gaps_real), np.max(gaps_anticipated_for_val)])
        for ax, gaps in zip(axs, [gaps_real, gaps_anticipated_for_val]):
            ax.fill_between(times[offset:], 0, confidence_band_sizes, color=f"C{0 if oob else 1}", alpha=0.2,
                            label="uncertainty about $\\mu$")
            ax.fill_between(times[offset:], confidence_band_sizes, gaps, color="purple", alpha=0.2,
                            label="variance deviation")
            ax.plot(times[offset:], gaps, color="red", alpha=0.5, label="Total Gap to $\\mu$.")
            ax.axhline(0, linestyle="-", color="black", linewidth=1)
            ax.fill_between(times, 0, eps, alpha=0.2, color="green",
                            label=f"acceptance area for $\\varepsilon = {eps}$")

            accepting_indices = [t + 1 for t in np.where(gaps < eps)[0] if t >= min_trees]

            num_trees_required_for_stable_correction_term_estimate = self.get_num_trees_required_for_stable_correction_term_estimate()
            ax.fill_between([1, num_trees_required_for_stable_correction_term_estimate], [0, 0], [1, 1], color="red",
                            alpha=0.2)

            if len(accepting_indices) > 0:
                stopping_point = np.max(
                    [min_trees, offset + accepting_indices[0], num_trees_required_for_stable_correction_term_estimate])
                ax.axvline(stopping_point, color="black", linestyle="--", label="Oracle Stopping Point")
            else:
                stopping_point = np.nan

            if not np.isnan(stopping_point):
                max_xlim = np.max([max_xlim, min([2 * stopping_point, len(gaps_real)])])

            ax.set_title(f"Stopping point: {stopping_point} trees")

            ax.legend()
            ax.set_ylim([0, 2 * eps])

        for ax in axs:
            ax.set_xlim([offset, max_xlim])

        if fig is not None:
            plt.show()

    def get_results_for_different_alphas_on_dataset(self, eps, pareto_profiles, alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]):

        offset = 2

        # get statistics for 100 trees
        times_fit = np.array(self.times_fit)
        times_overhead = np.array(self.times_predict_train) + np.array(self.times_update)
        val_curve = self.scores_of_forests["val"]
        oob_curve = self.scores_of_forests["oob"]
        val_gap_100 = np.abs(val_curve[100] - np.mean(val_curve[-10:]))
        time_100 = times_fit[:100].sum()

        # check regrets for different alphas
        out = {
            "openmlid": self.openmlid,
            "val_gap_100": val_gap_100,
            "ok_100": val_gap_100 <= eps,
            "time_100": time_100
        }
        for alpha in alphas:
            if False:
                for oob in [False, True]:
                    analyzer.compute_ci_sequence_for_expected_performance_at_size_t_based_on_normality(
                        alpha=alpha,
                        offset=offset,
                        oob=oob
                    )

            stopping_point = get_stopping_point(
                analyzer,
                alpha=alpha,
                eps=eps,
                mode="realistic",
                min_trees=2,
                oob=True
            )
            if stopping_point is not None and not np.isnan(stopping_point) and stopping_point < len(val_curve):
                val_gap = np.abs(val_curve[stopping_point] - np.mean(val_curve[-10:]))
                oob_gap = np.abs(oob_curve[stopping_point] - np.mean(oob_curve[-10:]))
                time_we = (times_fit[:stopping_point] + times_overhead[:stopping_point]).sum()
            else:
                val_gap = oob_gap = 1
                time_we = np.nan

            out.update({
                f"t_{alpha}": stopping_point,
                f"gap_oob_{alpha}": oob_gap,
                f"gap_val_{alpha}": val_gap,
                f"ok_{alpha}": val_gap <= eps,
                f"time_asrf_{alpha}": time_we
            })

        # PARETO APPROACH
        if False:
            stopping_point = get_stopping_point(
                analyzer,
                profiles=pareto_profiles,
                mode="realistic",
                min_trees=2,
                oob=True
            )
            if stopping_point is not None and not np.isnan(stopping_point) and stopping_point < len(val_curve):
                val_gap = np.abs(val_curve[stopping_point] - val_curve).max()
                oob_gap = np.abs(oob_curve[stopping_point] - oob_curve).max()
                time_we = (times_fit[:stopping_point] + times_overhead[:stopping_point]).sum()
            else:
                val_gap = oob_gap = 1
                time_we = np.nan

            out.update({
                f"t_pareto": stopping_point,
                f"oob_val_pareto": oob_gap,
                f"gap_val_pareto": val_gap,
                f"ok_pareto": val_gap <= eps,
                f"time_asrf_pareto": time_we
            })
        return out

    def get_num_trees_required_for_stable_correction_term_estimate(self,
                                                                   max_trees=np.inf,
                                                                   max_iterations_without_new_max=5,
                                                                   do_discount=True,
                                                                   oob=True
                                                                   ):
        new_max_indices = []
        first_plateau_max_index = -1
        cur_max = 0
        cnt_no_new_max = 0
        key = "oob" if oob else "val"
        correction_terms = self.correction_terms_for_t1_per_time[key]
        if max_trees < np.inf:
            correction_terms = correction_terms[:max_trees]
        for t, v in enumerate(correction_terms, start=1):
            if do_discount:
                v /= t
            if v < cur_max:
                cnt_no_new_max += 1
            else:
                cur_max = v
                cnt_no_new_max = 0

            if cnt_no_new_max >= max_iterations_without_new_max:
                return t
        return np.nan

    def estimate_var_t_sampled(self, X_val, y_val, t=None, sample_size=10):

        # one-hot-encoding of label vector
        label_list = list(forest.classes_)
        y_val_oh = np.zeros((len(y_val), len(np.unique(y_val))))
        for i, label in enumerate(y_val):
            y_val_oh[i,label_list.index(label)] = 1

        # determine deltas for all tree-instance-label combination
        deltas = []
        for tree in forest.estimators_[:t]:
            deltas.append(tree.predict_proba(X_val) - y_val_oh)
        deltas = np.array(deltas)

        def get_lamba(delta_1, delta_2):
            if delta_1.shape != delta_2.shape:
                raise ValueError()
            return np.sum(delta_1*delta_2, axis=1)

        # estimate V[l_i^ss]
        lambdas = np.array([get_lamba(delta_s, delta_s) for delta_s in deltas])
        V_lambda_ss = np.var(lambdas.flatten())

        # estimate V[l_i^ss']
        lambdas = np.array([get_lamba(delta_s, delta_sp) for i, delta_s in enumerate(deltas) for j, delta_sp in enumerate(deltas) if i != j])
        V_lambda_sp = np.var(lambdas.flatten())

        # estimate cov[l_i^ss, l_i^ss']
        indices_for_s = np.arange(t)
        lambda_pairs = []
        for s in np.random.choice(indices_for_s, size=min([sample_size, len(indices_for_s)]), replace=False):
            indices_for_sp = list(np.arange(t))
            indices_for_sp.remove(s)

            lambda_ss = get_lamba(deltas[s], deltas[s]).copy()

            for sp in np.random.choice(indices_for_sp, size=min([sample_size, len(indices_for_sp)]), replace=False):
                lambda_ssp = get_lamba(deltas[s], deltas[sp]).copy()
                lambda_pairs.append([lambda_ss, lambda_ssp])
        lambda_pairs = np.array(lambda_pairs)
        lambda_pairs = lambda_pairs.transpose(1, 2, 0)
        lambda_pairs = lambda_pairs.reshape(2, -1)
        cov_lambda_s_sp = np.cov(lambda_pairs, rowvar=True, bias=True)[0,1]

        # estimate cov[l_i^ss', l_i^ss'']
        indices_for_s = np.arange(t)
        lambda_pairs = []
        for s in np.random.choice(indices_for_s, size=min([sample_size, len(indices_for_s)]), replace=False):
            indices_for_sp = list(np.arange(t))
            indices_for_sp.remove(s)

            for sp in np.random.choice(indices_for_sp, size=min([sample_size, len(indices_for_sp)]), replace=False):
                lambda_ssp = get_lamba(deltas[s], deltas[sp])

                indices_for_spp = indices_for_sp.copy()
                indices_for_spp.remove(sp)

                for spp in np.random.choice(indices_for_spp, size=min([sample_size, len(indices_for_spp)]), replace=False):
                    lambda_sspp = get_lamba(deltas[s], deltas[spp])
                    lambda_pairs.append([lambda_ssp, lambda_sspp])
        lambda_pairs = np.array(lambda_pairs)
        lambda_pairs = lambda_pairs.transpose(1, 2, 0)
        lambda_pairs = lambda_pairs.reshape(2, -1)
        cov_lambda_s_sp_spp = np.cov(lambda_pairs, rowvar=True, bias=True)[0,1]

        n = X_val.shape[0]

        return (
                V_lambda_ss * t +
                V_lambda_sp * 2 * t * (t - 1) +
                cov_lambda_s_sp * 4 * t * (t - 1) +
                cov_lambda_s_sp_spp * 4 * t * (t - 1) * (t - 2)
        ) / (t ** 4 * n)

    def get_stopping_point(self, alpha, eps, mode="realistic", min_trees=2, beta = 0.5, oob=True):
        if mode == "oracle":
            num_trees_used_for_forecast = len(self.scores_of_single_trees["oob"])
        elif mode == "realistic":
            num_trees_used_for_forecast = None
        else:
            raise ValueError("mode must be 'oracle' or 'realistic'")

        if num_trees_used_for_forecast is not None and min_trees > num_trees_used_for_forecast:
            raise ValueError(
                f"num_trees_used_for_forecast is  {num_trees_used_for_forecast} but must not be bigger than min_trees, which is {min_trees}")


        key = "oob" if oob else "val"
        Y = self.Y_train if oob else self.Y_val
        k = Y.shape[1]
        n = Y.shape[0]  # TODO: this is not really correct; it should be the number of *test* instances.

        vbar_per_t = self.correction_terms_for_t1_per_time[key]
        cbar_per_t = self.confidence_term_for_correction_term_per_time[key]

        t = min_trees  # minimum number of trees, usually one per CPU, at least 2

        while True:
            vbar = vbar_per_t[t - 1]
            cbar = cbar_per_t[t - 1]
            tau = stats.t.ppf(1 - (1 - beta) * (1 - alpha) / k, df=t-1)
            kappa = stats.norm.ppf(1 - beta * (1 - alpha))

            # first criterion: bound on expected regret is smaller than eps.
            delta = eps - (vbar + tau * cbar / np.sqrt(t)) / t
            if delta > 0:
                if kappa * np.sqrt(2 / (n * t)) <= delta:
                    return t
            t += 1

    def create_full_belief_plot(
            self,
            alpha=0.95,
            eps=0.01,
            ci_offset=2,
            min_trees=5,
            decision_oob=True,
            scoring_oob=True,
            eps_limit_multiplier=3,
            ax=None
    ):
        second_key = alpha

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 5))
        else:
            fig = None

        decision_lines = {}
        x_lim = 400
        expected_limit_performances = {}
        for i, key in enumerate(["oob", "val"]):
            oob = key == "oob"

            if key not in self.ci_histories or second_key not in [key]:
                analyzer.compute_ci_sequence_for_expected_performance_at_size_t_based_on_normality(
                    alpha=alpha,
                    offset=ci_offset,
                    oob=oob
                )

            color = f"C{i}"
            scores = self.scores_of_forests[key]
            correction_terms_over_time = np.array(self.correction_terms_for_t1_per_time[key])
            cis = np.array([list(e) for e in self.ci_histories[key][second_key]["orig"]])

            times = np.arange(1, len(scores) + 1)

            # actual performance
            ax.plot(times, scores, color=color, label="$Z_t^{" + key + "}$")

            # expected performance (approximated)
            final_belief_of_correction_term_at_t1 = correction_terms_over_time[-1]
            correction_terms_over_time_based_on_final_belief = final_belief_of_correction_term_at_t1 / (times * (0.366 if oob else 1))
            expected_limit_performance = sum(cis[-1]) / 2 - final_belief_of_correction_term_at_t1  # take the last estimate (as best estimate), but do NOT divide by t here, because this is inferred via the estimate on t = 1
            expected_limit_performance_believed = np.mean(cis, axis=1) - correction_terms_over_time[ci_offset - 1:]
            ax.axhline(expected_limit_performance, linestyle="--", color=color, linewidth=1,
                       label="$\mathbb{E}[Z_\infty]$")
            expected_limit_performances[key] = expected_limit_performance

            expected_performance_curve = expected_limit_performance + correction_terms_over_time_based_on_final_belief
            expected_performance_curve_believed = expected_limit_performance_believed + correction_terms_over_time[ci_offset - 1:] / times[ci_offset - 1:]
            if oob:
                diff_curve = (self.scores_of_forests[key] - expected_performance_curve)

            ax.plot(times, expected_performance_curve, color=color,
                    linestyle="dotted", label="$\mathbb{E}[Z_t]$")
            if key == "val":
                ax.fill_between(times, expected_limit_performance - eps, expected_limit_performance + eps,
                                color="green", alpha=0.2)

            # add confidence intervals for what is believed to be the true final performance
            ax.plot(times[ci_offset - 1:], expected_limit_performance_believed, color=color, linestyle="--")
            ax.fill_between(
                times[ci_offset - 1:],
                [e[0] - final_belief_of_correction_term_at_t1 for e in cis],
                [e[1] - final_belief_of_correction_term_at_t1 for e in cis],
                alpha=0.2
            )

            # plot decision lines
            decision_line = self.get_stopping_point(alpha, eps, mode="realistic" if oob else "oracle", min_trees=min_trees, oob=oob)
            decision_lines[key] = decision_line
            ax.axvline(decision_line, color=color, linestyle="--", linewidth=1, label="$t^{" + key + "}$")
            if oob:
                decision_line = self.get_stopping_point(alpha, eps, "realistic" if oob else "oracle", min_trees=min_trees, oob=oob)
                decision_lines[key + "_conservative"] = decision_line
                ax.axvline(decision_line, color=color, linestyle="dotted", linewidth=1,
                           label="$t^{" + key + "}$ OOB pure")
                x_lim = np.max([x_lim, decision_line * 1.2])

        #ax.set_xlim([1, x_lim])
        ax.set_xscale("log")
        ax.set_ylim([min(expected_limit_performances.values()) - eps_limit_multiplier * eps,
                     max(expected_limit_performances.values()) + eps_limit_multiplier * eps])
        #ax.set_ylim([0.25, 0.36])

        ax.legend()
        ax.set_title(
            f"{self.openmlid} - "
            f"Stopping after {decision_lines['oob']} trees (OOB)."
            f"Perfect decision would be {decision_lines['val']} (VAL orcale)"
        )

        if fig is not None:
            return fig


def create_analysis_file_from_result_file(
        openmlid,
        seed,
        result_folder="./results",
        analysis_folder="../analysis",
        show_progress=True
):

    # read in results
    filename = f"{result_folder}/{openmlid}_{seed}.json"
    print(f"Opening file {filename}")
    analysis_data = {
        "openmlid": openmlid,
        "seed": seed
    }
    with open(filename) as f:
        encoded_results = json.loads(json.load(f))

        # decode results
        print("Unpacking results")
        Y_train = np.array(encoded_results[1])
        Y_val = np.array(encoded_results[2])
        prob_history_oob = np.array(
            json.loads(zlib.decompress(eval(encoded_results[3])).decode().replace("nan", "-1")))
        prob_history_val = np.array(
            json.loads(zlib.decompress(eval(encoded_results[4])).decode().replace("nan", "-1")))

        prob_history_oob[prob_history_oob < 0] = np.nan
        prob_history_val[prob_history_val < 0] = np.nan

        # times
        times_fit = encoded_results[-4]
        times_predict_train = encoded_results[-3]
        times_predict_val = encoded_results[-2]
        times_update = encoded_results[-1]

    # create analysis data
    print("Adding dedicated analysis data")
    if prob_history_oob.shape[0] != prob_history_val.shape[0]:
        raise ValueError("Inconsistent tree count!")

    # compute probabilistic behavior of forest
    scores_of_single_trees = {}
    single_tree_scores_mean_ests = {}
    single_tree_scores_std_ests = {}
    scores_of_forests = {}
    correction_terms_for_t1_per_time = {}
    confidence_term_for_correction_term_per_time = {}
    variances_of_zt = {}
    for key, probs_orig, Y in zip(["oob", "val"], [prob_history_oob, prob_history_val], [Y_train, Y_val]):

        # compute distribution per forest size
        counter = np.zeros((probs_orig.shape[1], probs_orig.shape[2]))  # count how many trees voted on each instance (also use classes for convenience)
        probs_forest = np.zeros(counter.shape)
        probs_forest[:] = np.nan
        prob_vars_forest = np.zeros(counter.shape)
        prob_vars_forest[:] = np.nan
        scores_of_single_trees_k = []
        single_tree_scores_mean_ests_k = []
        single_tree_scores_std_ests_k = []
        forest_scores_k = []
        correction_terms_k = []
        confidence_term_for_correction_term_per_time_k = []
        variances_of_zt_k = {}

        momenter = Momenter(input_dims=Y.shape, max_p=4, keep_history=True)

        iterable = tqdm(probs_orig) if show_progress else probs_orig
        for t, probs_tree in enumerate(iterable, start=1):

            # estimate variance V[Z_t] with many bootstraps
            sample_size_to_estimate_VZt = 100
            if False and t <= 500 and t <= probs_orig.shape[0] * 0.25:
                scores = []
                for _ in range(sample_size_to_estimate_VZt):
                    permutation = np.random.choice(range(len(probs_orig)), t, replace=False)
                    probs = probs_orig[permutation]
                    brier_score = np.nanmean(((Y - np.nanmean(probs, axis=0)) ** 2).sum(axis=1))
                    scores.append(brier_score)

                variances_of_zt_k[t] = np.var(scores)

            # tell momenter
            momenter.add_entry(probs_tree)

            probs_forest = momenter.means_over_time[-1]
            moments = momenter.moments_over_time_[-1]
            prob_vars_forest = moments[1]  # index 1 is for variances
            correction_term = np.nanmean(prob_vars_forest.sum(axis=1))
            confidence_term = np.nanmean(np.sqrt(np.maximum(0, moments[3] - moments[1] ** 2)).sum(axis=1))
            correction_terms_k.append(correction_term)
            confidence_term_for_correction_term_per_time_k.append(confidence_term)

            # compute actual scores for this tree and the forest including this tree
            score_tree = np.nanmean(((probs_tree - Y) ** 2).sum(axis=1))
            score_forest = np.nanmean(((probs_forest - Y) ** 2).sum(axis=1))
            scores_of_single_trees_k.append(score_tree)
            forest_scores_k.append(score_forest)

            # compute empirical mean and std of performance of a single tree per forest size
            # mu = single_tree_scores_mean_ests[-1] if single_tree_scores_mean_ests else 0
            single_tree_scores_mean_ests_k.append(np.nanmean(scores_of_single_trees_k))  # ((t - 1) * mu + score_tree) / t)
            single_tree_scores_std_ests_k.append(np.nanstd(scores_of_single_trees_k))

        # sanity check of correction terms
        assert np.isclose(correction_terms_k[-1], np.nanvar(probs_orig, axis=0).sum(axis=1).mean())

        # compute probabilities and scores
        scores_of_single_trees[key] = tuple(scores_of_single_trees_k)
        single_tree_scores_mean_ests[key] = single_tree_scores_mean_ests_k
        single_tree_scores_std_ests[key] = single_tree_scores_std_ests_k
        scores_of_forests[key] = tuple(forest_scores_k)
        correction_terms_for_t1_per_time[key] = tuple(correction_terms_k)
        confidence_term_for_correction_term_per_time[key] = confidence_term_for_correction_term_per_time_k
        variances_of_zt[key] = variances_of_zt_k

    analysis_data.update({
        "prob_history_oob": prob_history_oob.tolist(),
        "prob_history_val": prob_history_val.tolist(),
        "Y_train": Y_train.tolist(),
        "Y_val": Y_val.tolist(),
        "num_trees": prob_history_oob.shape[0],
        "times_fit": times_fit,
        "times_predict_train": times_predict_train,
        "times_predict_val": times_predict_val,
        "times_update": times_update,
        "scores_of_single_trees": scores_of_single_trees,
        "single_tree_scores_mean_ests": single_tree_scores_mean_ests,
        "single_tree_scores_std_ests": single_tree_scores_std_ests,
        "scores_of_forests": scores_of_forests,
        "correction_terms_for_t1_per_time": correction_terms_for_t1_per_time,
        "confidence_term_for_correction_term_per_time": confidence_term_for_correction_term_per_time,
        "variances_of_zt": variances_of_zt
    })

    # dump analysis data into file
    analysis_file = f"{analysis_folder}/{openmlid}_{seed}.json"
    analysis_file_gz = f"{analysis_file}.gz"
    with open(analysis_file, "w") as f:
        json.dump(analysis_data, f)

    # gzip file
    with open(analysis_file, 'rb') as f_in, gzip.open(analysis_file_gz, 'wb') as f_out:
        f_out.writelines(f_in)
