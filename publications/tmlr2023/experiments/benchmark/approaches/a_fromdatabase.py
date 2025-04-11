import numpy as np

from .approach import DeviationBasedApproach
from asforests import EnsemblePerformanceAssessor


class DatabaseWiseApproach(DeviationBasedApproach):

    def __init__(
            self,
            upper_bound_for_sample_size=10**6,
            population_mode="stream",
            single_data_point_per_ensemble_member=False,
            create_estimates_for_iid_scenario=True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.upper_bound_for_sample_size = upper_bound_for_sample_size
        self.population_mode = population_mode
        self.single_data_point_per_ensemble_member = single_data_point_per_ensemble_member
        self.create_estimates_for_iid_scenario = create_estimates_for_iid_scenario
        if not create_estimates_for_iid_scenario and single_data_point_per_ensemble_member:
            raise ValueError(f"if create_estimates_for_iid_scenario is False, then we have no iid estimates, so single_data_point_per_ensemble_member should also be False")

        # state variables
        self.deviation_matrices = None
        self.epa = None
    
    def reset(self):
        super().reset()
        if self.create_estimates_for_iid_scenario:
            self.epa = EnsemblePerformanceAssessor(
                upper_bound_for_sample_size=self.upper_bound_for_sample_size,
                population_mode=self.population_mode,
                execute_asserts=False
            )
        else:
            self.deviation_matrices = []
        
    @property
    def deviation_means_in_conditional_setting(self):
        matrices = self.epa.deviation_matrices if self.create_estimates_for_iid_scenario else self.deviation_matrices
        return np.mean(matrices, axis=0)
        

    @property
    def deviation_vars_in_conditional_setting(self):
        matrices = self.epa.deviation_matrices if self.create_estimates_for_iid_scenario else self.deviation_matrices
        return np.var(matrices, axis=0)

    @property
    def deviation_means_in_iid_setting(self):
        return self.epa.gap_mean_point

    @property
    def deviation_vars_in_iid_setting(self):
        return self.epa.gap_var_point

    @property
    def deviation_covs_in_iid_setting(self):
        return self.epa.gap_cov_across_members_point

    def receive_deviations_of_new_ensemble_member(self, deviation_matrix):

        if self.create_estimates_for_iid_scenario:
            if self.single_data_point_per_ensemble_member:
                idx = (self.epa.n if self.epa.n is not None else 0) % deviation_matrix.shape[0]
                deviation_row = deviation_matrix[idx].reshape(1, deviation_matrix.shape[1]).copy()
                deviation_matrix[:] = np.nan
                deviation_matrix[idx] = deviation_row
            self.epa.add_deviation_matrix(deviation_matrix)

        else:

            # TODO: create incremental solution for means/vars
            self.deviation_matrices.append(deviation_matrix)
