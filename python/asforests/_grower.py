import numpy as np
import scipy.stats as stats
import logging
import time
from .momenter_ import Momenter


def get_dummy_info_supplier(history):
    
    def iterator():
        for e in history:
            yield e
    return iterator()


class EnsembleGrower:
    
    def __init__(self, output_supplier, k, n, tolerances, min_size=4, max_size=None, beta=0.5, logger=None):

        # setup variables
        self.output_supplier = output_supplier
        self.n = n
        self.k = k
        self.tolerances = tolerances
        self.alphas = np.array([alpha for _, alpha in self.tolerances])
        self.epsilons = np.array([eps for eps, _ in self.tolerances])
        self.max_size = max_size
        self.beta = beta
        self.logger = logger if logger is not None else logging.getLogger("asbe.grower")

        # compute minimum size
        ### TODO COMPUTE THIS

        # state variables
        self.counts = np.zeros((n, k))
        self.c_history, self.v_history, self.size = [], [], 0
        self.momenter = Momenter()
        self.step_times = []

    def grow(self):
        
        # now start training
        self.logger.info(f"Start training.")
        while self.step() > 0:
            pass
        self.logger.info("Forest grown completely. Stopping routine!")

    def get_deltas(self, t):

        """
        computes the delta as defined in Corollary 5. t0 is given implicitly by the size of the current ensemble.
        :param t:
        :return:
        """
        initialized = self.size > 0
        v = self.v_history[-1] if initialized else 0
        c = self.c_history[-1] if initialized else 0

        deltas = np.zeros(len(self.alphas))
        for i, alpha in enumerate(self.alphas):
            kappa = stats.norm.ppf(1 - self.beta * (1 - alpha))
            if not initialized:
                delta = kappa * np.sqrt(2 / (self.n * t))
            else:
                tau = stats.t.ppf(1 - (1 - self.beta) * (1 - alpha) / self.k, df=t - 1)
                delta = kappa * np.sqrt(2 / (self.n * t)) + (v + tau * c / np.sqrt(t)) / t
            deltas[i] = delta
        return deltas

    def get_min_t_that_satisfies_conditions(self):
        t = np.max([4, self.size])
        while np.any(self.get_deltas(t) > self.epsilons):
            t += 1
        return t

    def step(self, t_p=None):
        t_start = time.time()
        if self.max_size is not None and self.size >= self.max_size:
            return 0

        self.logger.debug(f"Starting Iteration.")

        # determine number of additional members
        if t_p is None:
            t_p = self.get_min_t_that_satisfies_conditions() - self.size
        if t_p <= 0:
            return 0

        # train new base learners and update model
        new_outputs = self.output_supplier(t_p)
        self.size += len(new_outputs)
        self.momenter.add_batch(new_outputs)
        moments = self.momenter.moments
        assert not np.any(np.isnan(moments)), "There are nan entries in the moments!"

        self.v_history.append(moments[1].sum(axis=1).mean())
        self.c_history.append(np.sqrt(np.maximum(0, moments[3] - moments[1]**2)).sum(axis=1).mean())
        self.step_times.append(time.time() - t_start)
        return len(new_outputs)
