from experiments.problem_instance.problem_instance import ProblemInstance
from sklearn.datasets import make_classification
from abc import abstractmethod

from pathlib import Path
import json
import numpy as np
import itertools as it

from unittest import TestCase, skip
from parameterized import parameterized

from tqdm import tqdm

import logging


# define stream handler
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)

# configure logger for tester
logger = logging.getLogger("tester")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


def get_standard_benchmark(**kwargs):
    wrapper = ProblemInstanceWrapperForTesting(ensemble_seed=0)

    from experiments.benchmark.benchmark import Benchmark
    return Benchmark(
        problem_instance=wrapper.pi,
        **kwargs
    )

def get_problem_instance_for_openmlid(openmlid, n_checkpoints, t_checkpoints):

    file = Path(f"{Path(__file__).parent}/assets/pi_{openmlid}.json")

    if not file.exists():
        data_seed = 0
        ensemble_seed = 0
        training_instances_per_class = 50
        validation_size = 20

        pi = ProblemInstance(
            num_samples_allowed_for_ground_truth_approximation=10**2,
            data_description=openmlid,
            is_classification=True,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            num_possible_ensemble_members=5,
            training_instances_per_class=training_instances_per_class,
            n_checkpoints=n_checkpoints,
            t_checkpoints=t_checkpoints,
            validation_size=validation_size
        )

    else:

        # load problem instance file
        with open(file) as f:
            pi = ProblemInstance.from_dict(json.load(f))
    
    return file, pi

class ProblemInstanceWrapperForTesting:
    
    def __init__(
            self,
            ensemble_seed,
            data_gen_seed=0,
            data_seed=0,
            n_samples=100,
            training_instances_per_class=0.1,
            validation_size=20,
            num_possible_ensemble_members=5,
            num_samples_allowed_for_ground_truth_approximation=10**7,
            n_checkpoints=[2],
            t_checkpoints=[10, 100, 1000]
        ):

        self.file = Path(f"{Path(__file__).parent}/assets/pi_synth_{ensemble_seed}.json")

        self.n_checkpoints = n_checkpoints
        self.t_checkpoints = t_checkpoints

        if not self.file.exists():

            X, y = make_classification(n_classes=2, n_samples=n_samples, n_features=20, random_state=data_gen_seed)

            self.pi = ProblemInstance(
                num_samples_allowed_for_ground_truth_approximation=num_samples_allowed_for_ground_truth_approximation,
                data_description=(X, y),
                is_classification=True,
                data_seed=data_seed,
                ensemble_seed=ensemble_seed,
                num_possible_ensemble_members=num_possible_ensemble_members,
                training_instances_per_class=training_instances_per_class,
                n_checkpoints=self.n_checkpoints,
                t_checkpoints=self.t_checkpoints,
                validation_size=validation_size
            )

        else:

            # load problem instance file
            with open(self.file) as f:
                self.pi = ProblemInstance.from_dict(json.load(f))
    
    def dump_problem_instance(self, overwrite=False):
        if not self.file.exists() or overwrite:
            with open(self.file, "w") as f:
                json.dump(self.pi.to_dict(), f)


class ApproachTestClass(TestCase):

    def setUp(self):
        return super().setUp()

    @abstractmethod
    def get_approach(self, seed, estimated_parameters):
        raise NotImplementedError

    def test_that_returned_shapes_are_correct(self):
        if self.__class__ == ApproachTestClass:
            return
        
        logger.info("Test correctness of returned shapes")

        # tell bootstrapping about the prediction matrices
        pi = ProblemInstanceWrapperForTesting(
            ensemble_seed=0,
            num_possible_ensemble_members=8
        )
        a = self.get_approach(seed=None, estimated_parameters=["E[Z_nt]", "V[Z_nt]", "E[Z_nt|D_val]", "V[Z_nt|D_val]"]) # must be set in setup
        a.reset()
        a.tell_ground_truth_labels(pi.pi.y_oh_val)
        for pm in pi.pi.predictions_val:
            a.receive_predictions_of_new_ensemble_member(pm)

        # check whether iid shapes are correct
        pred_mean_iid_as_array = a.estimate_performance_mean_in_iid_setup(t=pi.t_checkpoints)
        pred_var_iid_as_array = a.estimate_performance_var_in_iid_setup(n=pi.n_checkpoints, t=pi.t_checkpoints)
        self.assertEqual((len(pi.t_checkpoints),), pred_mean_iid_as_array.shape)
        self.assertEqual((len(pi.n_checkpoints), len(pi.t_checkpoints)), pred_var_iid_as_array.shape)
        for i, (n, pred_var_for_n_from_array) in enumerate(zip(pi.n_checkpoints, pred_var_iid_as_array)):
            for j, (t, pred_mean_from_array, pred_var_from_array) in enumerate(zip(pi.t_checkpoints, pred_mean_iid_as_array, pred_var_for_n_from_array)):
                pred_mean = a.estimate_performance_mean_in_iid_setup(t=t)
                assert (1,) == pred_mean.shape, f"Wrong shape returned by {self.__class__.__name__} for E[Z_nt]. Should be (1,) but saw {pred_mean.shape}"
                self.assertAlmostEqual(pred_mean_from_array, pred_mean[0], msg=f"Incorrect value recovered by {self.__class__.__name__} for E[Z_nt]. Before: {pred_mean_from_array}. After: {pred_mean[0]}")
                pred_var = a.estimate_performance_var_in_iid_setup(n=n, t=t)
                assert (1, 1) == pred_var.shape, f"Wrong shape returned by {self.__class__.__name__} for V[Z_nt]. Should be (1,1) but saw {pred_var.shape}"
                self.assertAlmostEqual(pred_var_from_array, pred_var[0, 0], msg=f"Incorrect value recovered by {self.__class__.__name__} for V[Z_nt]. Before: {pred_var_from_array}. After: {pred_var[0, 0]}")
            
        # check whether conditional shapes are correct
        pred_mean_cond_as_array = a.estimate_performance_mean_in_conditional_setup(t=pi.t_checkpoints)
        pred_var_cond_as_array = a.estimate_performance_var_in_conditional_setup(t=pi.t_checkpoints)
        self.assertEqual((len(pi.t_checkpoints),), pred_mean_cond_as_array.shape)
        self.assertEqual((len(pi.t_checkpoints), ), pred_var_cond_as_array.shape)
        for t, pred_mean_from_array, pred_var_from_array in zip(pi.t_checkpoints, pred_mean_cond_as_array, pred_var_cond_as_array):
            pred_mean = a.estimate_performance_mean_in_conditional_setup(t=t)
            assert (1,) == pred_mean.shape, f"Wrong shape returned by {self.__class__.__name__} for E[Z_nt|D_val]. Should be (1,) but saw {pred_mean.shape}"
            self.assertAlmostEqual(pred_mean_from_array, pred_mean[0], msg=f"Incorrect value recovered by {self.__class__.__name__} for E[Z_nt|D_val]. Before: {pred_mean_from_array}. After: {pred_mean[0]}")
            pred_var = a.estimate_performance_var_in_conditional_setup(t=t)
            assert (1,) == pred_var.shape, f"Wrong shape returned by {self.__class__.__name__} for V[Z_nt|D_val]. Should be (1,) but saw {pred_var.shape}"
            self.assertAlmostEqual(pred_var_from_array, pred_var[0], msg=f"Incorrect value recovered by {self.__class__.__name__} for V[Z_nt|D_val]. Before: {pred_var_from_array}. After: {pred_var[0]}")

    def test_reproducibility(self):
        if self.__class__ == ApproachTestClass:
            return

        logger.info("Test reproducibility")

        # tell bootstrapping about the prediction matrices
        pi = ProblemInstanceWrapperForTesting(
            ensemble_seed=0,
            num_possible_ensemble_members=8
        )
        a = self.get_approach(seed=0, estimated_parameters=["E[Z_nt]", "V[Z_nt]", "E[Z_nt|D_val]", "V[Z_nt|D_val]"])

        # create two estimates for each parameter
        pred_mean_iid_as_array = []
        pred_var_iid_as_array = []
        pred_mean_cond_as_array = []
        pred_var_cond_as_array = []
        for _ in range(2):
            a.reset()
            a.tell_ground_truth_labels(pi.pi.y_oh_val)
            for pm in pi.pi.predictions_val:
                a.receive_predictions_of_new_ensemble_member(pm)
            pred_mean_iid_as_array.append(a.estimate_performance_mean_in_iid_setup(t=pi.t_checkpoints))
            pred_var_iid_as_array.append(a.estimate_performance_var_in_iid_setup(n=pi.n_checkpoints, t=pi.t_checkpoints))
            pred_mean_cond_as_array.append(a.estimate_performance_mean_in_conditional_setup(t=pi.t_checkpoints))
            pred_var_cond_as_array.append(a.estimate_performance_var_in_conditional_setup(t=pi.t_checkpoints))
        
        # check that estimates are identical
        for param, array in zip(
            ["E[Z_nt]", "V[Z_nt]", "E[Z_nt|D_val]", "V[Z_nt|D_val]"],
            [pred_mean_iid_as_array, pred_var_iid_as_array, pred_mean_cond_as_array, pred_var_cond_as_array]
        ):
            assert np.all(array[0] == array[1]), f"{param} estimate could not be recovered. Was first {array[0]} and then {array[1]}"
        
    @parameterized.expand(
        range(5)
    )
    def test_that_returned_values_are_reasonable_for_iid(self, seed):
        if self.__class__ == ApproachTestClass:
            return
        
        logger.info(f"Test that iid estimated are reasonable for {self.__class__.__name__}")

        # tell bootstrapping about the prediction matrices
        wrapper = ProblemInstanceWrapperForTesting(
            ensemble_seed=0,
            n_samples=20,
            training_instances_per_class=0.2,
            validation_size=0.2,
            num_possible_ensemble_members=8,
            num_samples_allowed_for_ground_truth_approximation=10**7
        )
        assert wrapper.pi.exact_ground_truth_feasible, f"No exact computation of ground truth possible! {wrapper.pi.required_samples_for_exact_ground_truth_computation} samples are required but only {wrapper.pi.num_samples_allowed_for_ground_truth_approximation} are allowed for ground truth approximation."

        # show each prediction matrix exactly once to the model
        a = self.get_approach(seed, estimated_parameters=["E[Z_nt]", "V[Z_nt]"])
        a.reset()
        a.tell_ground_truth_labels(wrapper.pi.y_oh_val)
        logger.info("Feeding model")
        for _ in range(10**0):
            for pm in wrapper.pi.predictions_val:
                a.receive_predictions_of_new_ensemble_member(pm)

        # check whether iid estimates are in a reasonable region
        pred_mean_iid_as_array = a.estimate_performance_mean_in_iid_setup(t=wrapper.t_checkpoints)
        pred_var_iid_as_array = a.estimate_performance_var_in_iid_setup(n=wrapper.n_checkpoints, t=wrapper.t_checkpoints)
        self.assertFalse(np.any(np.isnan(pred_mean_iid_as_array)), f"E[Z_nt] estimates have nan value: {pred_mean_iid_as_array}")
        self.assertFalse(np.any(np.isnan(pred_var_iid_as_array)), f"V[Z_nt] estimates have nan value: {pred_var_iid_as_array}")
        self.assertFalse(np.any(pred_mean_iid_as_array < 0), f"E[Z_nt] estimates have negative values: {pred_mean_iid_as_array}")
        self.assertFalse(np.any(pred_mean_iid_as_array > 2), f"E[Z_nt] estimates have values bigger than 2: {pred_mean_iid_as_array}")
        self.assertFalse(np.any(pred_var_iid_as_array < 0), f"V[Z_nt] estimates have negative value: {pred_var_iid_as_array}")
        for i, (n, pred_var_for_n_from_array) in enumerate(zip(wrapper.n_checkpoints, pred_var_iid_as_array)):
            for j, (t, pred_mean_from_array, pred_var_for_t_n_from_array) in enumerate(zip(wrapper.t_checkpoints, pred_mean_iid_as_array, pred_var_for_n_from_array)):

                # test that estimate for E[Z_nt] is at least somewhere close to the real value (exact estimate impossible since approach has only validation data)
                pred_mean = a.estimate_performance_mean_in_iid_setup(t=[t])[0]
                assert np.isclose(pred_mean_from_array, pred_mean, atol=10**-16), f"Estimating mean for {t=} should be identical to an extracted mean estimate from a call with t={wrapper.t_checkpoints}"
                assert np.abs(wrapper.pi.means_iid[i] - pred_mean) < 0.1, f"Estimates for E[Z_nt] by {a.__class__.__name__} are off the mark. Expected {wrapper.pi.means_iid[j]} but saw {pred_mean} for {t=}"

                # test that estimate for sqrt(V[Z_nt]) is at least somewhere close to the real value (exact estimate impossible since approach has only validation data)
                pred_var = a.estimate_performance_var_in_iid_setup(n=n, t=t)[0, 0]
                assert np.isclose(pred_var_for_t_n_from_array, pred_var, atol=10**-16), f"Estimating var for {n=} and {t=} should be identical to an extracted var estimate from a call with n={wrapper.n_checkpoints} and t={wrapper.t_checkpoints}"
                assert np.abs(wrapper.pi.vars_iid[i, j] - pred_var) < 0.1, f"Estimates for V[Z_nt] by {a.__class__.__name__} are off the mark. Expected {wrapper.pi.vars_iid[i, j]} but saw {pred_var} for {t=}"
    
    def adjust_approach_object_for_evaluation_on_conditional_convergence_test_checkpoint(self, approach, checkpoint):
        pass

    @parameterized.expand(
        it.product(
            [
                "E[Z_nt|D_val]",
                "V[Z_nt|D_val]"
            ],
            range(1)
        )
    )
    def test_that_approach_converges_to_no_error_on_validation_data(self, param, seed):
        """
            All approaches should converge to an estimation error of 0 for both E[Z_nt|D_val] and V[Z_nt|D_val] when conditioning on concrete (known) data

            Args:
                a_name (_type_): _description_
                a_obj (_type_): _description_
        """

        if self.__class__ == ApproachTestClass:
            return
        
        logger.info(f"Test that estimation errors for {param} converge to 0 for {self.__class__.__name__}")

        # configuration of how we do the test
        n = 2 # values for n and t for which we want to check convergence of E[Z_nt|D_val] and V[Z_nt|D_val]
        t = 10 # values for n and t for which we want to check convergence of E[Z_nt|D_val] and V[Z_nt|D_val]
        basis_for_checkpoints = 2
        first_exponent = 1
        exponent_step_size = 3
        num_steps = 3 if self.__class__.__name__ == "TestDatabaseBasedApproach" else 5 # BUT CURRENTLY THE DATABASE BASED APPROACH IS TOO SLOW
        required_factor_of_improvement_by_each_checkpoint = 1.2 # require that the average error at the previous checkpoint was at least 20% higher than the current one
        num_estimates_per_checkpoint_for_average_error_approximation = 10

        # tell bootstrapping about the prediction matrices
        wrapper = ProblemInstanceWrapperForTesting(
            data_gen_seed=seed,
            ensemble_seed=0,
            n_samples=20,
            training_instances_per_class=0.2,
            validation_size=n,
            num_possible_ensemble_members=8,
            num_samples_allowed_for_ground_truth_approximation=10**7,
            n_checkpoints=[n], # irrelevant here
            t_checkpoints=[t]
        )
        assert wrapper.pi.exact_ground_truth_feasible, f"No exact computation of ground truth possible! {wrapper.pi.required_samples_for_exact_ground_truth_computation} samples are required but only {wrapper.pi.num_samples_allowed_for_ground_truth_approximation} are allowed for ground truth approximation."

        # get ground truth parameters
        logger.info(f"Starting estimation of true value of {param}")
        true_param_value = wrapper.pi.means_cond[0] if param == "E[Z_nt|D_val]" else wrapper.pi.vars_cond[0]
        logger.info(f"True value to estimate is {param}={true_param_value}")

        # create approach objects for different seeds
        approaches = [
            self.get_approach(approach_seed, estimated_parameters=[param])
            for approach_seed in range(num_estimates_per_checkpoint_for_average_error_approximation)
        ]
        for a in approaches:
            a.reset()
            a.tell_ground_truth_labels(wrapper.pi.y_oh_val)
        generators = [
            wrapper.pi.get_prediction_matrix_generator(ensemble_sequence_seed=gen_seed, only_validation_data=True)
            for gen_seed in range(num_estimates_per_checkpoint_for_average_error_approximation)
        ]
        
        # get generator for the estimates of the approach on the given problem
        e_checkpoints = [basis_for_checkpoints**(first_exponent + i * exponent_step_size) for i in range(num_steps)]
        logger.info(f"Checkpoints that we will visit in this test: {e_checkpoints=}")

        # no check improvement rates
        avg_error_at_last_checkpoint = np.inf
        b = 0
        for e_checkpoint in e_checkpoints:

            # now stepping all copies of the approach with these matrices
            errors_for_different_seeds_at_checkpoint = []
            logger.info(f"Stepping the approaches until size {e_checkpoint}")
            for i, (a, g) in enumerate(zip(approaches, generators), start=1):

                # get prediction matrices necessary to advance to the next checkpoint
                required_matrices = e_checkpoint - b
                pms = []
                for _ in range(required_matrices):
                    pms.append(next(g))

                # optionally let the approach adjust itself for this test on a higher checkpoint
                self.adjust_approach_object_for_evaluation_on_conditional_convergence_test_checkpoint(a, e_checkpoint)
                
                # advance the approaches until next checkpoint
                for pm in pms:
                    a.receive_predictions_of_new_ensemble_member(pm)
                
                # determine estimation error for this approach copy
                if param == "E[Z_nt|D_val]":
                    pred = a.estimate_performance_mean_in_conditional_setup(t=t)[0]
                else:
                    pred = a.estimate_performance_var_in_conditional_setup(t=t)[0]
                estimation_error = np.abs(pred - true_param_value)
                errors_for_different_seeds_at_checkpoint.append(estimation_error)

                avg_error_at_current_checkpoint = np.mean(errors_for_different_seeds_at_checkpoint)
                actual_average_improvement_rate = avg_error_at_last_checkpoint / avg_error_at_current_checkpoint
                logger.debug(
                    f"Estimate for {param} with {e_checkpoint} ensemble members in {i}-th scenario: {np.round(pred, 4)}. "
                    f"Error: {np.round(estimation_error, 4)}. "
                    f"Avg err of round: {np.round(avg_error_at_current_checkpoint, 4)}. "
                    f"Avg impr. rate of round: {np.round(actual_average_improvement_rate, 2)}"
                )
            
            b = e_checkpoint

            # check that improvement up to here is as expected
            if avg_error_at_last_checkpoint < np.inf:
                logger.info(
                    f"Average improvement rate compared to last step is {np.round(actual_average_improvement_rate, 2)} "
                    f"(required is {required_factor_of_improvement_by_each_checkpoint})"
                )

                # check that average improvement is by the required factor
                self.assertLessEqual(required_factor_of_improvement_by_each_checkpoint, actual_average_improvement_rate)
            else:
                logger.info(
                    "Averge improvement rate not compared at first step since no comparison available. "
                    f"Average error here was {np.round(avg_error_at_current_checkpoint, 2)}"
                )
            avg_error_at_last_checkpoint = avg_error_at_current_checkpoint
