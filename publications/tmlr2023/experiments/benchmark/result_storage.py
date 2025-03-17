import numpy as np
import pandas as pd
import json


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
            for p, estimates_for_p in estimates.items():
                for a, estimates_for_a in estimates_for_p.items():
                    for t, estimates_for_t in estimates_for_a.items():
                        for b in estimates_for_t.keys():
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
            "true_param_values": {p: [int(v) for v in l] for p, l in self._true_param_values.items()},
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
        true_values_for_checkpoint = {p: v[t_index] for p, v in self._true_param_values.items()}
        errors = {
            col: estimates[col].apply(lambda e: e - true_values_for_checkpoint[col])
            for col in estimates.columns
        }
        return pd.DataFrame(errors, index=estimates.index)
