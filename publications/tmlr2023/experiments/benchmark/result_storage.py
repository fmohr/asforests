import numpy as np
import pandas as pd
import json
import collections.abc


class ResultStorage:

    def __init__(self, true_param_values, approach_names, t_checkpoints, estimates=None, runtimes=None):
        for p, v in true_param_values.items():
            if len(v) != len(t_checkpoints):
                raise ValueError(f"There must be a true value for each parameter value and each checkpoint")
        
        self._true_param_values = true_param_values
        self._approach_names = approach_names
        self._t_checkpoints = [int(t) for t in t_checkpoints]
        self._budgets = set()

        # self.estimates[p][a][t][b] will contain the estimate for parameter p obtained from approach a for ensemble size t when b ensembles were trained (budget)
        self._estimates = {}
        self._runtimes = {}
        for a in approach_names:
            self._estimates[a] = {}
            self._runtimes[a] = {}
            
            if estimates is not None:
                assert runtimes is not None, "if estimates are given, runtimes must not be None"
                for b in estimates[a]:
                    self.add_estimates(approach_name=a, budget=b, estimates_per_checkpoint=estimates[a][b], runtimes=runtimes[a][b])

        # add known budgets
        if estimates is not None:
            for a, estimates_for_a in estimates.items():
                for b in estimates_for_a.keys():
                    assert isinstance(b, int), f"Budget '{b}' is not an integer and hence not a valid budget."
                    self._budgets.add(b)

    @property
    def true_param_values(self):
        return self._true_param_values
    
    @property
    def approach_names(self):
        return self._approach_names
    
    @property
    def t_checkpoints(self):
        return self._t_checkpoints
    
    @property
    def budgets(self):
        return self._budgets
    
    def serialize(self, f=None):

        d = {
            "true_param_values": {p: [v for v in l] for p, l in self._true_param_values.items()},
            "approach_names": self._approach_names,
            "t_checkpoints": [int(t) for t in self._t_checkpoints],
            "estimates": self._estimates,
            "runtimes": self._runtimes
        }

        if f is None:

            """Convert the object to a JSON string."""
            return json.dumps(d)
        json.dump(d, f)

    @classmethod
    def unserialize(cls, src):

        def convert_keys_to_int(pairs):
            return {int(k) if k.isdigit() else k: v for k, v in pairs}
        
        if type(src) == str:
            """Convert a JSON string back to an object."""
            data = json.loads(src, object_pairs_hook=convert_keys_to_int)
        else:
            data = json.load(src, object_pairs_hook=convert_keys_to_int)
        
        return cls(**data)

    @classmethod
    def merge(cls, stores):

        if len(stores) < 2:
            raise ValueError(f"Need at least two result storages to merge.")

        # check that checkpoints coincide
        t_checkpoints = None
        true_param_values = None
        for s in stores:
            if t_checkpoints is None:
                t_checkpoints = s.t_checkpoints
                true_param_values = s.true_param_values
            else:
                if len(t_checkpoints) != len(s.t_checkpoints):
                    raise ValueError("Cannot merge results storages with different checkpoints")
                if np.any(t_checkpoints != s.t_checkpoints):
                    raise ValueError("Cannot merge results storages with different checkpoints")
                for param in true_param_values:
                    if np.any(true_param_values[param] != s.true_param_values[param]):
                        raise ValueError(f"Cannot merge results storages with different True parameter values for param {param}.")
        if true_param_values is None:
            raise ValueError(f"Cannot merge result stores with None for true_param_value")

        # collect approach names
        approach_names = set()
        for s in stores:
            approach_names |= set(s.approach_names)
        approach_names = sorted(approach_names)

        # merge estimates
        estimates = {}
        runtimes = {}
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping) and k in d:
                    deep_update(d[k], v)  # Recursive update for nested dictionaries
                else:
                    d[k] = v  # Overwrite for non-dictionaries
            return d
        for s in stores:
            deep_update(estimates, s._estimates)
            deep_update(runtimes, s._runtimes)
        return ResultStorage(
            approach_names=approach_names,
            true_param_values=true_param_values,
            t_checkpoints=t_checkpoints,
            estimates=estimates,
            runtimes=runtimes
            )
        
    
    def add_estimates(self, approach_name, budget, estimates_per_checkpoint, runtimes):
        
        if budget not in self._budgets:
            self._estimates[approach_name][budget] = {}
            self._runtimes[approach_name][budget] = {}
            self._budgets.add(budget)

        if budget not in self._estimates[approach_name]:
            self._estimates[approach_name][budget] = {}

        assert len(estimates_per_checkpoint) == len(self._t_checkpoints), f"Expected a dictionary with {len(self._t_checkpoints)} entries, one for each check point, but received {len(estimates_per_checkpoint)}"
        for t, estimates_for_t in estimates_per_checkpoint.items():                
            if t not in self._estimates[approach_name][budget]:
                self._estimates[approach_name][budget][t] = {}
            for p, e in estimates_for_t.items():
                self._estimates[approach_name][budget][t][p] = float(e)
        
        self._runtimes[approach_name][budget] = runtimes
    
    def rename_approach(self, n_from, n_to):
        i = self.approach_names.index(n_from)
        self.approach_names[i] = n_to
        self._estimates[n_to] = self._estimates[n_from]
        self._runtimes[n_to] = self._runtimes[n_from]
        del self._estimates[n_from]
        del self._runtimes[n_from]
    
    def get_estimates_from_approach_for_checkpoint(self, approach_name, t):
        results = []
        budgets = []
        for budget, estimates_for_budget in self._estimates[approach_name].items():
            budgets.append(budget)
            results.append(estimates_for_budget[t])
        return pd.DataFrame(results, index=budgets)
    
    def get_errors_from_approach_for_checkpoint(self, approach_name, t):
        estimates = self.get_estimates_from_approach_for_checkpoint(approach_name=approach_name, t=t)
        t_index = self._t_checkpoints.index(t)
        for key in estimates.columns:
            if key not in self._true_param_values:
                raise ValueError(f"No ground truth available for {key}")
        true_values_for_checkpoint = {p: v[t_index] for p, v in self._true_param_values.items()}
        errors = {
            col: estimates[col].apply(lambda e: e - true_values_for_checkpoint[col])
            for col in estimates.columns
        }
        return pd.DataFrame(errors, index=estimates.index)
