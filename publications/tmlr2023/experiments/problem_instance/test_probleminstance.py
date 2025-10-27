import pytest
from experiments.problem_instance.problem_instance import ProblemInstance
from experiments.problem_instance._ground_truth_computer import GroundTruthComputer
from sklearn.datasets import make_classification
import numpy as np
import itertools as it
import json

def get_standard_problem_instance(
        n_classes=3,
        n_samples=10**4,
        n_features=20,
        n_informative=10,
        data_gen_seed=0,
        data_seed=0,
        ensemble_seed=0,
        portion_validation=0.1,
        portion_training = 0.1
    ):
    X, y = make_classification(n_classes=n_classes, n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=data_gen_seed)

    return ProblemInstance(
        data_description=(X, y),
        is_classification=True,
        data_seed=data_seed,
        ensemble_seed=ensemble_seed,
        num_possible_ensemble_members=4,
        training_instances_per_class=portion_training,
        validation_size=portion_validation,
        num_samples_allowed_for_ground_truth_approximation=0,
        n_checkpoints=None,
        t_checkpoints=None
    )

def check_loadability_of_base_properties(pi):

    # test that data is there
    assert pi.X is not None, "no data available"
    assert pi.X.shape[0] > 0, "no instances available"
    assert pi.X.shape[1] > 0, "no features available"
    assert pi.y is not None, "no labels available"

    # test that we can get prediction and deviation matrices
    deviations = pi.deviations
    assert isinstance(deviations, np.ndarray), "deviations are not an ndarray"
    assert len(deviations.shape) == 3, "deviations are not 3D"
    expected_shape = (pi.num_possible_ensemble_members, pi.X.shape[0], pi.num_labels)
    assert expected_shape == deviations.shape, f"deviations have wrong shape. Expected: {expected_shape} but saw {deviations.shape}"


@pytest.mark.parametrize("openmlid", [61, 188])
def test_load_problem_instance_for_openmlid(openmlid):
    pi = ProblemInstance(
        openmlid,
        is_classification=True,
        data_seed=0,
        ensemble_seed=0,
        num_possible_ensemble_members=4,
        training_instances_per_class=0.75,
        validation_size=0.25,
        num_samples_allowed_for_ground_truth_approximation=0,
        n_checkpoints=None,
        t_checkpoints=None
    )
    check_loadability_of_base_properties(pi)

@pytest.mark.parametrize("openmlid", [61, 188])
def test_serializability_for_openmlid(openmlid):
    pi = ProblemInstance(
        openmlid,
        is_classification=True,
        data_seed=0,
        ensemble_seed=0,
        num_possible_ensemble_members=4,
        training_instances_per_class=0.75,
        validation_size=0.25,
        num_samples_allowed_for_ground_truth_approximation=10**4,
        n_checkpoints=np.array([2, 4]),
        t_checkpoints=np.array([2, 4])
    )

    # check that the ProblemInstance can be fully recovered even after string serialization to json
    pi_recovered = ProblemInstance.from_dict(pi.to_dict())
    pi_recovered_json = ProblemInstance.from_dict(json.loads(json.dumps(pi.to_dict())))
    for pi_p in [pi_recovered, pi_recovered_json]:
        assert np.array_equal(pi.X, pi_p.X)
        assert np.array_equal(pi.y, pi_p.y)
        assert np.array_equal(pi._indices_train, pi_p._indices_train)
        assert np.array_equal(pi._indices_val, pi_p._indices_val)
        assert np.array_equal(pi._indices_oos, pi_p._indices_oos)
        assert type(pi.predictions) == type(pi_p.predictions) and np.all(pi.predictions == pi_p.predictions)
        assert type(pi.deviations) == type(pi_p.deviations) and np.all(pi.deviations == pi_p.deviations)
        assert type(pi.n_checkpoints) == type(pi_p.n_checkpoints) and np.all(pi.n_checkpoints == pi_p.n_checkpoints)
        assert type(pi.t_checkpoints) == type(pi_p.t_checkpoints) and np.all(pi.t_checkpoints == pi_p.t_checkpoints)
        assert np.array_equal(pi.num_samples_allowed_for_ground_truth_approximation, pi_p.num_samples_allowed_for_ground_truth_approximation)
    
    # check that we can even recover ground truth values without need to recompute those
    true_means_iid, true_vars_iid, true_means_cond, true_vars_cond = pi.means_iid, pi.vars_iid, pi.means_cond, pi.vars_cond
    assert pi._true_means_for_iid_case is not None
    assert pi._true_vars_for_iid_case is not None
    assert pi._true_means_for_cond_case is not None
    assert pi._true_vars_for_cond_case is not None
    pi_recovered = ProblemInstance.from_dict(pi.to_dict())
    pi_recovered_json = ProblemInstance.from_dict(json.loads(json.dumps(pi.to_dict())))
    for pi_p in [pi_recovered, pi_recovered_json]:
        assert type(pi._true_means_for_iid_case) == type(pi_p._true_means_for_iid_case) and np.all(pi._true_means_for_iid_case == pi_p._true_means_for_iid_case)
        assert type(pi._true_vars_for_iid_case) == type(pi_p._true_vars_for_iid_case) and np.all(pi._true_vars_for_iid_case == pi_p._true_vars_for_iid_case)
        assert type(pi._true_means_for_cond_case) == type(pi_p._true_means_for_cond_case) and np.all(pi._true_means_for_cond_case == pi_p._true_means_for_cond_case)
        assert type(pi._true_vars_for_cond_case) == type(pi_p._true_vars_for_cond_case) and np.all(pi._true_vars_for_cond_case == pi_p._true_vars_for_cond_case)

@pytest.mark.parametrize("n_classes, n_samples, n_features, seed", list(it.product([2, 3], [16, 32], [16, 32], [0, 1])))
def test_load_problem_instance_for_synthetic_data(n_classes, n_samples, n_features, seed):

    X, y = make_classification(n_classes=n_classes, n_samples=n_samples, n_features=n_features, n_informative=n_features // 2, random_state=seed)

    pi = ProblemInstance(
        data_description=(X, y),
        is_classification=True,
        data_seed=0,
        ensemble_seed=seed,
        num_possible_ensemble_members=4,
        training_instances_per_class=0.75,
        validation_size=0.25,
        num_samples_allowed_for_ground_truth_approximation=0,
        n_checkpoints=None,
        t_checkpoints=None
    )
    check_loadability_of_base_properties(pi)

def test_proper_data_role_distribution_for_relative_sizes():

    X, y = make_classification(n_classes=3, n_samples=10**4, n_features=20, n_informative=10, random_state=0)

    portion_validation = 0.1
    portion_training = 0.1

    pi = ProblemInstance(
        data_description=(X, y),
        is_classification=True,
        data_seed=0,
        ensemble_seed=0,
        num_possible_ensemble_members=4,
        training_instances_per_class=portion_training,
        validation_size=portion_validation,
        num_samples_allowed_for_ground_truth_approximation=0,
        n_checkpoints=None,
        t_checkpoints=None
    )

    # load data
    num_instances = pi.X.shape[0]

    # check proportions
    num_train_instances = len(pi._indices_train)
    num_valid_instances = len(pi._indices_val)
    num_oos_instances = len(pi._indices_oos)
    assert num_valid_instances / num_instances == portion_validation, f"Proportion of validation data is {num_valid_instances / num_instances} but should be {portion_validation}"
    assert num_train_instances / ((1 - portion_validation) * num_instances) == portion_training, f"Proportion of training data is {num_train_instances / ((1 - portion_validation) * num_instances)} but should be {portion_validation}"
    assert num_oos_instances / num_instances > 1 - portion_training - portion_validation, f"Proportion of out of sample data is {num_oos_instances / num_instances} but should be more than {1 - portion_training - portion_validation}"

def test_proper_data_role_distribution_for_absolute_sizes():

    X, y = make_classification(n_classes=3, n_samples=10**4, n_features=20, n_informative=10, random_state=0)

    portion_validation = 3
    portion_training = 0.1

    pi = ProblemInstance(
        data_description=(X, y),
        is_classification=True,
        data_seed=0,
        ensemble_seed=0,
        num_possible_ensemble_members=4,
        training_instances_per_class=portion_training,
        validation_size=portion_validation,
        num_samples_allowed_for_ground_truth_approximation=0,
        n_checkpoints=None,
        t_checkpoints=None
    )

    # load data
    num_instances = pi.X.shape[0]

    # check proportions
    num_valid_instances = len(pi._indices_val)
    assert num_valid_instances == portion_validation, f"There are {num_valid_instances} validation instances but should be {portion_validation}"
    
@pytest.mark.parametrize("seed", range(5))
def test_reproducibility_from_seed(seed):

    # create two instances with same parameters
    pis = [
        get_standard_problem_instance(data_seed=seed, ensemble_seed=seed)
        for _ in range(2)
    ]

    # check equality
    pi1, pi2 = pis
    assert np.array_equal(pi1.X, pi2.X)
    assert np.array_equal(pi1.y, pi2.y)
    assert np.array_equal(pi1.y_oh, pi2.y_oh)
    assert np.array_equal(pi1.predictions, pi2.predictions)
    assert np.array_equal(pi1.deviations, pi2.deviations)


def test_ground_truth_correctness_for_small_instances():

    seed = 0

    # get data
    n_classes = 2
    n_samples = 6
    n_features = 20
    X, y = make_classification(n_classes=n_classes, n_samples=n_samples, n_features=n_features, n_informative=n_features // 2, random_state=seed)

    n = 2
    t = 3

    num_samples_allowed_for_ground_truth_approximation = 10**6

    # create problem instance
    pi = ProblemInstance(
        data_description=(X, y),
        is_classification=True,
        data_seed=0,
        ensemble_seed=seed,
        num_possible_ensemble_members=4,
        training_instances_per_class=2,
        validation_size=2,
        num_samples_allowed_for_ground_truth_approximation=num_samples_allowed_for_ground_truth_approximation,
        n_checkpoints=n,
        t_checkpoints=t
    )
    assert pi.exact_ground_truth_feasible, "ProblemInstances affirms that no ground truth can be computed for this problem."
    
    # iid case
    gtc = GroundTruthComputer(pi.deviations)
    true_mean = gtc.get_true_parameter("E[Z_nt]", t=t)
    true_var = gtc.get_true_parameter("V[Z_nt]", t=t, n=n)
    assert np.isclose(true_mean, pi.means_iid[0])
    assert np.isclose(true_var, pi.vars_iid[0, 0])

    # conditional case
    gtc = GroundTruthComputer(pi.deviations_val)
    true_mean = gtc.get_true_parameter("E[Z_nt|D_val]", t=t)
    true_var = gtc.get_true_parameter("V[Z_nt|D_val]", t=t)
    assert np.isclose(true_mean, pi.means_cond[0])
    assert np.isclose(true_var, pi.vars_cond[0])

def test_ground_truth_approximation():
    
    seed = 0

    # get data
    n_classes = 3
    n_samples = 64
    n_features = 20
    X, y = make_classification(n_classes=n_classes, n_samples=n_samples, n_features=n_features, n_informative=n_features // 2, random_state=seed)

    n_checkpoints=np.array([2, 10, 20])
    t_checkpoints=np.array([2, 10, 20])

    num_samples_allowed_for_ground_truth_approximation = 10**6

    # create problem instance
    pis = ProblemInstance(
        data_description=(X, y),
        is_classification=True,
        data_seed=0,
        ensemble_seed=seed,
        num_possible_ensemble_members=4,
        training_instances_per_class=0.75,
        validation_size=0.25,
        num_samples_allowed_for_ground_truth_approximation=num_samples_allowed_for_ground_truth_approximation,
        n_checkpoints=n_checkpoints,
        t_checkpoints=t_checkpoints
    )
    assert not pis.exact_ground_truth_feasible, "Problem is too easy and can be solved directly."

    # approximate population
    pis._approximate_ground_truth_parameters(num_samples_per_job=num_samples_allowed_for_ground_truth_approximation // 10, n_jobs=2)

    # check that E[Z_nt] values are reasonable and consistent (decrease in t)
    assert (len(t_checkpoints), ) == pis.means_iid.shape
    prev_mean = np.inf
    for t, mean_for_t in zip(t_checkpoints, pis.means_iid):
        assert mean_for_t < prev_mean
        prev_mean = mean_for_t

    # check that values for V[Z_nt] are consistent (decrease with both n and t)
    assert (len(n_checkpoints), len(t_checkpoints)) == pis.vars_iid.shape
    for i in range(len(n_checkpoints[:-1])):
        for j in range(len(t_checkpoints[:-1])):
            assert pis.vars_iid[i, j] > pis.vars_iid[i, j + 1]
            assert pis.vars_iid[i, j] > pis.vars_iid[i + 1, j]

    # extract true values for E[Z_nt|D_val]
    assert (len(t_checkpoints), ) == pis.means_cond.shape
    for i in range(len(t_checkpoints) - 1):
        assert pis.means_cond[i] > pis.means_cond[i + 1]
    
    # extract true values for V[Z_nt|D_val]
    assert (len(t_checkpoints), ) == pis.vars_cond.shape
    for i in range(len(t_checkpoints) - 1):
        assert pis.vars_cond[i] > pis.vars_cond[i + 1]

def test_that_ground_truth_values_are_sensitive_to_change_in_data_seed():

    pis = [
        get_standard_problem_instance(data_seed=seed)
        for seed in range(2)
    ]

    # check equality
    pi1, pi2 = pis
    assert np.array_equal(pi1.X, pi2.X)
    assert np.array_equal(pi1.y, pi2.y)
    assert np.array_equal(pi1.y_oh, pi2.y_oh)
    assert pi1.validation_size == pi2.validation_size
    assert np.any(pi1._indices_train != pi2._indices_train)
    assert np.any(pi1._indices_val != pi2._indices_val)
    assert np.any(pi1._indices_oos != pi2._indices_oos)
    assert not np.array_equal(pi1.predictions, pi2.predictions) # data seed influences which data is used for training
    assert not np.array_equal(pi1.deviations, pi2.deviations) # naturally the deviations also change


def test_that_ground_truth_values_are_sensitive_to_change_in_ensemble_seed():

    pis = [
        get_standard_problem_instance(ensemble_seed=seed)
        for seed in range(2)
    ]

    # check equality
    pi1, pi2 = pis
    assert np.array_equal(pi1.X, pi2.X)
    assert np.array_equal(pi1.y, pi2.y)
    assert np.array_equal(pi1.y_oh, pi2.y_oh)
    assert pi1.validation_size == pi2.validation_size
    assert np.all(pi1._indices_train == pi2._indices_train)
    assert np.all(pi1._indices_val == pi2._indices_val)
    assert np.all(pi1._indices_oos == pi2._indices_oos)
    assert not np.array_equal(pi1.predictions, pi2.predictions) # data seed influences which data is used for training
    assert not np.array_equal(pi1.deviations, pi2.deviations) # naturally the deviations also change
