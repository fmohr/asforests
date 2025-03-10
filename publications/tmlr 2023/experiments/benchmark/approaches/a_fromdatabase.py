from .approach import DeviationBasedApproach
from asforests import EnsemblePerformanceAssessor


class DatabaseWoseApproach(DeviationBasedApproach):

    def __init__(self, upper_bound_for_sample_size=10**8, population_mode="stream"):
        self.epa = EnsemblePerformanceAssessor(
            upper_bound_for_sample_size=upper_bound_for_sample_size,
            population_mode=population_mode,
            execute_asserts=False
        )

    @property
    def deviation_means(self):
        return self.epa.gap_mean_point

    @property
    def deviation_vars(self):
        return self.epa.gap_var_point

    @property
    def deviation_covs(self):
        return self.epa.gap_cov_across_members_point

    def receive_deviations_of_new_ensemble_member(self, deviation_matrix):
        self.epa.add_deviation_matrix(deviation_matrix)
