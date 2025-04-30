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
        self.num_validation_instances = None
        self.epa = None
    
    def reset(self):
        super().reset()
        self.epa = EnsemblePerformanceAssessor(
            upper_bound_for_sample_size=self.upper_bound_for_sample_size,
            population_mode=self.population_mode,
            execute_asserts=False,
            estimate_performance_var_for_iid_case="V[Z_nt]" in self.estimated_parameters,
            estimate_performance_var_for_conditional_case="V[Z_nt|D_val]" in self.estimated_parameters,
            max_number_of_xi_terms_to_include_in_update=10**8,
            logger=self.logger
        )
        if not self.create_estimates_for_iid_scenario:  # maybe we only need this
            self.deviation_matrices = []
        
    @property
    def deviation_means_in_conditional_setting(self):
        matrices = self.epa.deviation_matrices if self.epa is not None else self.deviation_matrices
        return np.mean(matrices, axis=0)
        

    @property
    def deviation_vars_in_conditional_setting(self):
        matrices = self.epa.deviation_matrices if self.epa is not None else self.deviation_matrices
        return np.var(matrices, axis=0)

    @property
    def deviation_means_in_iid_setting(self):
        if not self.create_estimates_for_iid_scenario:
            raise ValueError(f"Approach not configured to estimate iid parameters.")
        return self.epa.gap_mean_point

    @property
    def deviation_vars_in_iid_setting(self):
        if not self.create_estimates_for_iid_scenario:
            raise ValueError(f"Approach not configured to estimate iid parameters.")
        return self.epa.gap_var_point

    @property
    def deviation_covs_in_iid_setting(self):
        if not self.create_estimates_for_iid_scenario:
            raise ValueError(f"Approach not configured to estimate iid parameters.")
        return self.epa.gap_cov_across_members_point
    
    @property
    def xi_covs_in_conditional_setting(self):
        if "V[Z_nt|D_val]" not in self.estimated_parameters:
            raise ValueError(f"Approach not configured to estimate iid parameters.")
        if not self.epa.estimate_performance_var_for_conditional_case and not self.epa.estimate_performance_var_for_iid_case:
            raise ValueError(f"EnsemblePerformance estimator is not configured to estimate variances!")
        f = np.vectorize(lambda obj: obj.cov)  # make estimate biased
        covs = f(self.epa.mixed_moment_builders_for_conditional_xi_covs)
        return covs
    
    @property
    def xi_covs_in_iid_setting(self):
        if not self.create_estimates_for_iid_scenario or "V[Z_nt]" not in self.estimated_parameters:
            raise ValueError(f"Approach not configured to estimate iid parameters.")
        if not self.epa.estimate_performance_var_for_conditional_case and not self.epa.estimate_performance_var_for_iid_case:
            raise ValueError(f"EnsemblePerformance estimator is not configured to estimate variances!")
        f = np.vectorize(lambda obj: obj.cov)  # make estimate biased
        covs = f(self.epa.mixed_moment_builders_for_iid_xi_covs)
        return covs.flatten()

    def receive_deviations_of_new_ensemble_member(self, deviation_matrix):

        self.logger.info("Receiving new deviation matrix.")

        if self.num_validation_instances is None:
            self.num_validation_instances = deviation_matrix.shape[0]

        if self.epa is not None:
            if self.single_data_point_per_ensemble_member:
                idx = (self.epa.n if self.epa.n is not None else 0) % deviation_matrix.shape[0]
                deviation_row = deviation_matrix[idx].reshape(1, deviation_matrix.shape[1]).copy()
                deviation_matrix[:] = np.nan
                deviation_matrix[idx] = deviation_row
            self.logger.debug("Adding deviation matrix to Ensemble Performance Estimator")
            self.epa.add_deviation_matrix(deviation_matrix)

        else:

            # TODO: create incremental solution for means/vars
            self.logger.debug("Adding deviation matrix internally but NOT to Ensemble Performance Estimator")
            self.deviation_matrices.append(deviation_matrix)
        self.logger.info("Inclusion of new deviation matrix finished.")
