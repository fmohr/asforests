import itertools as it
from tqdm import tqdm
from scipy import stats
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import zlib
import multiprocessing.pool


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
                 show_progress_on_init=False
                 ):
        self.openmlid = openmlid
        self.seed = seed
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.num_trees = prob_history_oob.shape[0]
        if prob_history_oob.shape[0] != prob_history_val.shape[0]:
            raise ValueError("Inconsistent tree count!")
        self.times_fit = times_fit
        self.times_predict_train = times_predict_train
        self.times_predict_val = times_predict_val
        self.times_update = times_update

        # compute probabilistic behavior of forest
        self.scores_of_single_trees = {}
        self.single_tree_scores_mean_ests = {}
        self.single_tree_scores_std_ests = {}
        self.scores_of_forests = {}
        self.correction_terms_for_t1_per_time = {}
        self.num_trees_used_on_avg_for_oob_estimates_at_forest_size = []
        for key, probs_orig, Y in zip(["oob", "val"], [prob_history_oob, prob_history_val],
                                      [self.Y_train, self.Y_val]):

            # compute distribution per forest size
            counter = np.zeros((probs_orig.shape[1], probs_orig.shape[
                2]))  # count how many trees voted on each instance (also use classes for convenience)
            probs_forest = np.zeros(counter.shape)
            probs_forest[:] = np.nan
            prob_vars_forest = np.zeros(counter.shape)
            probs_forests = []
            single_tree_scores = []
            single_tree_scores_mean_ests = []
            single_tree_scores_std_ests = []
            forest_scores = []
            correction_terms = []

            iterable = tqdm(probs_orig) if show_progress_on_init else probs_orig
            for t, probs_tree in enumerate(iterable, start=1):
                mask = ~np.isnan(probs_tree)[:, 0]
                mask_insert = mask & np.isnan(probs_forest)[:, 0]
                mask_update = mask & ~np.isnan(probs_forest)[:, 0]
                old_counter = counter.copy()
                counter[mask] += 1
                probs_forest[mask_insert] = probs_tree[mask_insert]
                probs_forest[mask_update] = (old_counter[mask_update] * probs_forest[mask_update] + probs_tree[
                    mask_update]) / counter[mask_update]
                probs_forests.append(probs_forest.copy())

                # compute probability variances
                if t == 1:
                    correction_terms.append(0)  # no variance estimate for first tree
                else:
                    prob_vars_forest[mask] = (prob_vars_forest[mask] * old_counter[mask] + (
                                probs_tree[mask] - probs_forests[-2][mask]) * (
                                                          probs_tree[mask] - probs_forests[-1][mask])) / counter[
                                                 mask]  # this has slight bias
                    correction_term = np.nanmean(prob_vars_forest.sum(axis=1))
                    correction_terms.append(correction_term)

                # compute actual scores for this tree and the forest including this tree
                score_tree = np.nanmean(((probs_tree - Y) ** 2).sum(axis=1))
                score_forest = np.nanmean(((probs_forest - Y) ** 2).sum(axis=1))
                single_tree_scores.append(score_tree)
                forest_scores.append(score_forest)

                # update average number of trees used for an assessment
                if key == "oob":
                    self.num_trees_used_on_avg_for_oob_estimates_at_forest_size.append(
                        int((~np.isnan(probs_orig))[:t, :, 0].sum(axis=0).mean()))

                # compute empirical mean and std of performance of a single tree per forest size
                # mu = single_tree_scores_mean_ests[-1] if single_tree_scores_mean_ests else 0
                single_tree_scores_mean_ests.append(np.nanmean(single_tree_scores))  # ((t - 1) * mu + score_tree) / t)
                single_tree_scores_std_ests.append(np.nanstd(single_tree_scores))

            # compute probabilities and scores
            self.scores_of_single_trees[key] = tuple(single_tree_scores)
            self.single_tree_scores_mean_ests[key] = single_tree_scores_mean_ests
            self.single_tree_scores_std_ests[key] = single_tree_scores_std_ests
            self.scores_of_forests[key] = tuple(forest_scores)
            self.correction_terms_for_t1_per_time[key] = tuple(correction_terms)

        # things to compute lazy
        self.ci_histories = {
            "oob": {},
            "val": {}
        }

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
        trees_to_be_considered_for_ci = self.num_trees_used_on_avg_for_oob_estimates_at_forest_size[
            forest_size - 1] if oob else forest_size  # this is always t, even for OOB
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
        return self.num_trees_used_on_avg_for_oob_estimates_at_forest_size[forest_size]

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

    def get_stopping_point(self, alpha, eps, mode, min_trees=5, use_conservative_correction_term=False, oob=True,
                           use_oob_forest_size_estimate_for_correction_term=False):

        if mode not in ["oracle", "realistic"]:
            raise ValueError("mode must be 'oracle' or 'realistic'")

        key = "oob" if oob else "val"

        tree_scores = self.scores_of_single_trees[key]
        correction_terms = self.correction_terms_for_t1_per_time[key]
        num_tree_count_for_stable_correction_term_estimate = self.get_num_trees_required_for_stable_correction_term_estimate(
            max_iterations_without_new_max=5,
            do_discount=True,
            oob=oob
        )

        if mode == "oracle":
            std_of_scores = np.std(tree_scores)
            correction_term = correction_terms[-1]
        for t in range(min_trees, len(tree_scores) + 1):

            # define base of operation if no oracle is used
            if mode != "oracle":
                std_of_scores = np.std(tree_scores[:t])
                correction_term = (self.Y_train.shape[1] / 4) if use_conservative_correction_term else correction_terms[
                    t - 1]

            # compute pessimistic gap to asymptotic performance (uncertainty + noise)
            # for VAL curves, one uses t. For OOB curves, one pretends a smaller forest, of the avg number of trees used for predictions
            trees_to_be_considered_for_ci = self.num_trees_used_on_avg_for_oob_estimates_at_forest_size[
                t - 1] if oob else t  # this is always t, even for OOB
            ci = stats.norm.interval(alpha, loc=0, scale=std_of_scores / np.sqrt(trees_to_be_considered_for_ci))
            gap = ci[1] + correction_term / (
                trees_to_be_considered_for_ci if use_oob_forest_size_estimate_for_correction_term else t)

            # accept t if the gap is small enough
            if t >= num_tree_count_for_stable_correction_term_estimate and gap <= eps:
                return t

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

    def create_full_belief_plot(self, alpha, eps, ci_offset, min_trees=5, decision_oob=True, scoring_oob=True,
                                eps_limit_multiplier=3, ax=None):
        first_key_decision = "oob" if decision_oob else "val"
        first_key_scoring = "oob" if scoring_oob else "val"
        second_key = alpha

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 5))
        else:
            fig = None

        decision_lines = {}
        x_lim = 400
        for i, key in enumerate(["oob", "val"]):
            color = f"C{i}"
            scores = self.scores_of_forests[key]
            correction_terms_over_time = self.correction_terms_for_t1_per_time[key]
            cis = np.array([list(e) for e in self.ci_histories[key][second_key]["orig"]])

            times = range(1, len(scores) + 1)

            # actual performance
            ax.plot(times, scores, color=color, label="$Z_t^{" + key + "}$")

            # expected performance (approximated)
            final_belief_of_correction_term_at_t1 = correction_terms_over_time[-1]
            correction_terms_over_time_based_on_final_belief = final_belief_of_correction_term_at_t1 / times
            expected_limit_performance = sum(cis[
                                                 -1]) / 2 - final_belief_of_correction_term_at_t1  # take the last estimate (as best estimate), but do NOT divide by t here, because this is inferred via the estimate on t = 1
            ax.axhline(expected_limit_performance, linestyle="--", color=color, linewidth=1,
                       label="$\mathbb{E}[Z_\infty]$")
            ax.plot(times, expected_limit_performance + correction_terms_over_time_based_on_final_belief, color=color,
                    linestyle="dotted", label="$\mathbb{E}[Z_t]$")
            if key == "val":
                ax.fill_between(times, expected_limit_performance - eps, expected_limit_performance + eps,
                                color="green", alpha=0.2)

            # add confidence intervals for what is believed to be the true final performance
            ax.fill_between(
                times[ci_offset - 1:],
                [e[0] - final_belief_of_correction_term_at_t1 for e in cis],
                [e[1] - final_belief_of_correction_term_at_t1 for e in cis],
                alpha=0.2
            )

            # plot decision lines
            oob = key == "oob"
            decision_line = self.get_stopping_point(alpha, eps, "realistic" if oob else "oracle", min_trees=min_trees,
                                                    oob=oob)
            decision_lines[key] = decision_line
            ax.axvline(decision_line, color=color, linestyle="--", linewidth=1, label="$t^{" + key + "}$")
            if oob:
                decision_line = self.get_stopping_point(alpha, eps, "realistic" if oob else "oracle",
                                                        min_trees=min_trees, oob=oob,
                                                        use_oob_forest_size_estimate_for_correction_term=True)
                decision_lines[key + "_conservative"] = decision_line
                ax.axvline(decision_line, color=color, linestyle="dotted", linewidth=1,
                           label="$t^{" + key + "}$ OOB pure")
                x_lim = np.max([x_lim, decision_line * 1.2])

        ax.set_xlim([1, x_lim])
        ax.set_xscale("log")
        ax.set_ylim([expected_limit_performance - eps_limit_multiplier * eps,
                     expected_limit_performance + eps_limit_multiplier * eps])

        ax.legend()
        ax.set_title(
            f"Stopping after {decision_lines['oob']} trees (OOB)."
            f"Perfect decision would be {decision_lines['val']} (VAL orcale)"
        )

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
        num_trees_for_estimates_oob = np.array(self.num_trees_used_on_avg_for_oob_estimates_at_forest_size)
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


def get_analyzer_from_row(row, show_progress_on_init=False):
    openmlid = int(row["openmlid"])
    seed = int(row["seed"])
    results = json.loads(row["scores"])

    Y_train = np.array(results[1])
    Y_test = np.array(results[2])
    prob_history_oob = np.array(json.loads(zlib.decompress(eval(results[3])).decode().replace("nan", "-1")))
    prob_history_val = np.array(json.loads(zlib.decompress(eval(results[4])).decode().replace("nan", "-1")))

    num_classes = Y_train.shape[1]

    prob_history_oob[prob_history_oob < 0] = np.nan
    prob_history_val[prob_history_val < 0] = np.nan

    # times
    times_fit = results[-4]
    times_predict_train = results[-3]
    times_predict_val = results[-2]
    times_update = results[-1]

    analyzer = Analyzer(
        openmlid,
        seed,
        prob_history_oob,
        prob_history_val,
        Y_train,
        Y_test,
        times_fit,
        times_predict_train,
        times_predict_val,
        times_update,
        show_progress_on_init=show_progress_on_init
    )
    return analyzer


def write_analyzer_to_disk(analyzer):
    with open(f"analyzers/{analyzer.openmlid}_{analyzer.seed}.ana", "wb") as f:
        pickle.dump(analyzer, f)


def get_analyzer_from_disk(openmlid, seed, analysis_folder="."):
    with open(f"{analysis_folder}/analyzers/{openmlid}_{seed}.ana", "rb") as f:
        return pickle.load(f)


def is_analyzer_serialized(openmlid, seed, analysis_folder="."):
    return os.path.exists(f"{analysis_folder}/analyzers/{openmlid}_{seed}.ana")


def get_results_for_different_alphas_on_dataset(analyzer, eps, pareto_profiles, alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]):

    offset = 2

    # get statistics for 100 trees
    times_fit = np.array(analyzer.times_fit)
    times_overhead = np.array(analyzer.times_predict_train) + np.array(analyzer.times_update)
    val_curve = analyzer.scores_of_forests["val"]
    oob_curve = analyzer.scores_of_forests["oob"]
    val_gap_100 = np.abs(val_curve[100] - np.mean(val_curve[-10:]))
    time_100 = times_fit[:100].sum()

    # check regrets for different alphas
    out = {
        "openmlid": analyzer.openmlid,
        "val_gap_100": val_gap_100,
        "ok_100": val_gap_100 <= eps,
        "time_100": time_100
    }
    for alpha in alphas:
        for oob in [False, True]:
            analyzer.compute_ci_sequence_for_expected_performance_at_size_t_based_on_normality(
                alpha=alpha,
                offset=offset,
                oob=oob
            )

        stopping_point = get_stopping_point(
            analyzer,
            profiles=[(alpha, eps)],
            mode="realistic",
            min_trees=2,
            oob=True,
            use_oob_forest_size_estimate_for_correction_term=False
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
            f"oob_val_{alpha}": oob_gap,
            f"gap_val_{alpha}": val_gap,
            f"ok_{alpha}": val_gap <= eps,
            f"time_asrf_{alpha}": time_we
        })

    # PARETO APPROACH
    stopping_point = get_stopping_point(
        analyzer,
        profiles=pareto_profiles,
        mode="realistic",
        min_trees=2,
        oob=True,
        use_oob_forest_size_estimate_for_correction_term=False
    )
    if stopping_point is not None and not np.isnan(stopping_point) and stopping_point < len(val_curve):
        val_gap = np.abs(val_curve[stopping_point] - np.mean(val_curve[-10:]))
        oob_gap = np.abs(oob_curve[stopping_point] - np.mean(oob_curve[-10:]))
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


def get_results(
        openmlids,
        analyzer_fetcher,
        eps,
        pareto_profiles,
        alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        n_jobs=8,
        show_progress=False
):
    if show_progress:
        pbar = tqdm(total=len(openmlids))

    with multiprocessing.pool.ThreadPool(n_jobs) as pool:

        def f(openmlid):
            analyzer = analyzer_fetcher(openmlid, seed=0)
            result = get_results_for_different_alphas_on_dataset(analyzer, eps, pareto_profiles=pareto_profiles, alphas=alphas)
            if show_progress:
                pbar.update(1)
            return result

        results = pool.map(f, openmlids)
        if show_progress:
            pbar.close()
    return pd.DataFrame(results)


def get_num_trees_required_for_stable_correction_term_estimate(analyzer,
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
    correction_terms = analyzer.correction_terms_for_t1_per_time[key]
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


def get_stopping_point(analyzer, profiles, mode, min_trees=2, num_trees_used_for_forecast=None,
                       use_conservative_correction_term=False, oob=True,
                       use_oob_forest_size_estimate_for_correction_term=False):
    if num_trees_used_for_forecast is not None and min_trees > num_trees_used_for_forecast:
        raise ValueError(
            f"num_trees_used_for_forecast is  {num_trees_used_for_forecast} but must not be bigger than min_trees, which is {min_trees}")

    if mode not in ["oracle", "realistic"]:
        raise ValueError("mode must be 'oracle' or 'realistic'")

    key = "oob" if oob else "val"

    tree_scores = analyzer.scores_of_single_trees[key]
    correction_terms = analyzer.correction_terms_for_t1_per_time[key]

    if num_trees_used_for_forecast is not None:
        tree_scores = tree_scores[:num_trees_used_for_forecast]
        correction_terms = correction_terms[:num_trees_used_for_forecast]
        std_of_scores = np.std(tree_scores)
        correction_term = (analyzer.Y_train.shape[1] / 4) if use_conservative_correction_term else correction_terms[
            num_trees_used_for_forecast - 1]

    if mode == "oracle":
        std_of_scores = np.std(tree_scores)
        correction_term = correction_terms[-1]

    for t in range(min_trees, 10 ** 6):

        # define base of operation if no oracle is used
        if mode != "oracle" and t <= len(tree_scores):
            if num_trees_used_for_forecast is None:
                std_of_scores = np.std(tree_scores[:t])
                correction_term = (analyzer.Y_train.shape[1] / 4) if use_conservative_correction_term else correction_terms[
                    t - 1]

        # compute pessimistic gap to asymptotic performance (uncertainty + noise)
        # for VAL curves, one uses t. For OOB curves, one pretends a smaller forest, of the avg number of trees used for predictions
        trees_to_be_considered_for_ci = np.ceil(t * 0.366) if oob else t  # this is always t, even for OOB

        profiles_ok = len(profiles) * [False]
        for i, (alpha, eps) in enumerate(profiles):
            ci = stats.norm.interval(alpha, loc=0, scale=std_of_scores / np.sqrt(trees_to_be_considered_for_ci))
            gap = ci[1] + correction_term / (
                trees_to_be_considered_for_ci if use_oob_forest_size_estimate_for_correction_term else t)

            # accept t if the gap is small enough
            if gap <= eps:
                num_tree_count_for_stable_correction_term_estimate = get_num_trees_required_for_stable_correction_term_estimate(
                    analyzer,
                    max_trees=t,
                    max_iterations_without_new_max=5,
                    do_discount=True,
                    oob=oob
                )
                if t >= num_tree_count_for_stable_correction_term_estimate:
                    profiles_ok[i] = True

        if all(profiles_ok):
            return t

def create_full_belief_plot(
        analyzer,
        alpha,
        eps,
        ci_offset,
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
    for i, key in enumerate(["oob", "val"]):
        oob = key == "oob"

        if key not in analyzer.ci_histories or second_key not in [key]:
            analyzer.compute_ci_sequence_for_expected_performance_at_size_t_based_on_normality(
                alpha=alpha,
                offset=ci_offset,
                oob=oob
            )

        color = f"C{i}"
        scores = analyzer.scores_of_forests[key]
        correction_terms_over_time = analyzer.correction_terms_for_t1_per_time[key]
        cis = np.array([list(e) for e in analyzer.ci_histories[key][second_key]["orig"]])

        times = np.arange(1, len(scores) + 1)

        # actual performance
        ax.plot(times, scores, color=color, label="$Z_t^{" + key + "}$")

        # expected performance (approximated)
        final_belief_of_correction_term_at_t1 = correction_terms_over_time[-1]
        correction_terms_over_time_based_on_final_belief = final_belief_of_correction_term_at_t1 / (times * (0.366 if oob else 1))
        expected_limit_performance = sum(cis[-1]) / 2 - final_belief_of_correction_term_at_t1  # take the last estimate (as best estimate), but do NOT divide by t here, because this is inferred via the estimate on t = 1
        ax.axhline(expected_limit_performance, linestyle="--", color=color, linewidth=1,
                   label="$\mathbb{E}[Z_\infty]$")
        ax.plot(times, expected_limit_performance + correction_terms_over_time_based_on_final_belief, color=color,
                linestyle="dotted", label="$\mathbb{E}[Z_t]$")
        if key == "val":
            ax.fill_between(times, expected_limit_performance - eps, expected_limit_performance + eps,
                            color="green", alpha=0.2)

        # add confidence intervals for what is believed to be the true final performance
        ax.fill_between(
            times[ci_offset - 1:],
            [e[0] - final_belief_of_correction_term_at_t1 for e in cis],
            [e[1] - final_belief_of_correction_term_at_t1 for e in cis],
            alpha=0.2
        )

        # plot decision lines
        decision_line = analyzer.get_stopping_point(alpha, eps, "realistic" if oob else "oracle", min_trees=min_trees,
                                                oob=oob)
        decision_lines[key] = decision_line
        ax.axvline(decision_line, color=color, linestyle="--", linewidth=1, label="$t^{" + key + "}$")
        if oob:
            decision_line = analyzer.get_stopping_point(alpha, eps, "realistic" if oob else "oracle",
                                                    min_trees=min_trees, oob=oob,
                                                    use_oob_forest_size_estimate_for_correction_term=True)
            decision_lines[key + "_conservative"] = decision_line
            ax.axvline(decision_line, color=color, linestyle="dotted", linewidth=1,
                       label="$t^{" + key + "}$ OOB pure")
            x_lim = np.max([x_lim, decision_line * 1.2])

    ax.set_xlim([1, x_lim])
    ax.set_xscale("log")
    ax.set_ylim([expected_limit_performance - eps_limit_multiplier * eps,
                 expected_limit_performance + eps_limit_multiplier * eps])

    ax.legend()
    ax.set_title(
        f"Stopping after {decision_lines['oob']} trees (OOB)."
        f"Perfect decision would be {decision_lines['val']} (VAL orcale)"
    )

    if fig is not None:
        plt.show()