import numpy as np
import pandas as pd
import json


class ResultStorage:

    def __init__(self, true_param_values, approach_names, t_checkpoints, estimates=None):
        for p, v in true_param_values.items():
            if len(v) != len(t_checkpoints):
                raise ValueError(f"There must be a true value for each parameter value and each checkpoint")
        
        self._true_param_values = true_param_values
        self._approach_names = approach_names
        self._t_checkpoints = [int(t) for t in t_checkpoints]
        self._budgets = set()

        # self.estimates[p][a][t][b] will contain the estimate for parameter p obtained from approach a for ensemble size t when b ensembles were trained (budget)
        self.estimates = {
            p: {
                a: {
                    t_c: estimates[p][a][t_o] if estimates is not None and p in estimates and a in estimates[p] and t_o in estimates[p][a]
                    else {}
                    for t_c, t_o in zip(self._t_checkpoints, t_checkpoints)
                }
                for a in approach_names
            }
            for p in self._true_param_values.keys()
        }

        # add known budgets
        if estimates is not None:
            for p, estimates_for_p in estimates.items():
                for a, estimates_for_a in estimates_for_p.items():
                    for t, estimates_for_t in estimates_for_a.items():
                        for b in estimates_for_t.keys():
                            self._budgets.add(b)

    
    def serialize(self, f=None):

        d = {
            "true_param_values": {p: [int(v) for v in l] for p, l in self._true_param_values.items()},
            "approach_names": self._approach_names,
            "t_checkpoints": [int(t) for t in self._t_checkpoints],
            "estimates": self.estimates
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
    
    def add_estimates(self, approach_name, budget, estimates):
        self._budgets.add(budget)
        for col, val in estimates.items():
            assert isinstance(val, list) or isinstance(val, np.ndarray), f"Expected a list or numpy array for {col}, but {approach_name} returned {type(val)}"
            assert len(val) == len(self._t_checkpoints), f"Expected {len(self._t_checkpoints)} values for {col}, but {approach_name} returned {len(val)}"
            for t, v in zip(self._t_checkpoints, val):
                self.estimates[col][approach_name][t][budget] = float(v)
    
    def get_estimates_from_approach_for_checkpoint(self, approach_name, t):
        results = {}
        index = list(range(1, max(self._budgets) + 1))
        for param, estimates_for_param in self.estimates.items():
            results[param] = []
            for b in index:
                has_entry = b in estimates_for_param[approach_name][t]
                results[param].append(estimates_for_param[approach_name][t][b]) if has_entry else np.nan
        return pd.DataFrame(results, index=index)
    
    def get_errors_from_approach_for_checkpoint(self, approach_name, t):
        estimates = self.get_estimates_from_approach_for_checkpoint(approach_name=approach_name, t=t)
        t_index = self._t_checkpoints.index(t)
        true_values_for_checkpoint = {p: v[t_index] for p, v in self._true_param_values.items()}
        errors = {
            col: estimates[col].apply(lambda e: e - true_values_for_checkpoint[col])
            for col in estimates.columns
        }
        return pd.DataFrame(errors, index=estimates.index)
