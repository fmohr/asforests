import numpy as np
import pandas as pd
import json
from io import StringIO
from decimal import Decimal


class ResultStorage:

    def __init__(self, true_param_values, n_checkpoints, t_checkpoints, results=None):
        self._true_param_values = true_param_values

        for p, v in true_param_values.items():
            if p == "V[Z_nt]":
                if len(v.shape) != 2:
                    raise ValueError(f"Expected 2D ground truth for {p} but got something of shape {v.shape}: {v}")
            else:
                if len(v.shape) != 1:
                    raise ValueError(f"Expected 1D ground truth for {p} but got something of shape {v.shape}: {v}")
            
            expected_shape = (len(n_checkpoints), len(t_checkpoints)) if p == "V[Z_nt]" else (len(t_checkpoints), )
            if v.shape != expected_shape:
                raise ValueError(f"Shape of ground truth of {p} should be {expected_shape} but is {v.shape}: {v}")
        
        self._n_checkpoints = [int(n) for n in n_checkpoints]
        self._t_checkpoints = [int(t) for t in t_checkpoints]

        # self.estimates[p][a][t][b] will contain the estimate for parameter p obtained from approach a for ensemble size t when b ensembles were trained (budget)
        self._results = None
        if results is not None:
            self._results = results # do not conduct a sanity check

    @property
    def true_param_values(self):
        return self._true_param_values
    
    @property
    def approach_names(self):
        return sorted(pd.unique(self._results["approach"])) if self._results is not None else []
    
    @property
    def n_checkpoints(self):
        return self._n_checkpoints
    
    @property
    def t_checkpoints(self):
        return self._t_checkpoints

    @property
    def results(self):
        return self._results
    
    @property
    def budgets(self):
        return set(pd.unique(self._results["budget"])) if self._results is not None else set()

    def serialize(self, f=None):

        # serialize ground truth
        serializable_ground_truth = {}
        for p, l in self._true_param_values.items():
            if p == "V[Z_nt]":
                serializable_ground_truth[p] = [[Decimal(str(v)) for v in u] for u in l.tolist()]
            else:
                serializable_ground_truth[p] = [Decimal(str(v)) for v in l.tolist()]

        # serialize estimates
        if self._results is not None:
            serializable_results = self._results.copy()
            serializable_results["estimate"] = serializable_results["estimate"].apply(lambda x: str(Decimal(str(x))))
            serializable_results["runtime"] = serializable_results["runtime"].apply(lambda x: str(Decimal(str(x))))
            serializable_results = serializable_results.to_json(orient="records")
        else:
            serializable_results = None
        
        d = {
            "true_param_values": serializable_ground_truth,
            "n_checkpoints": [int(n) for n in self._n_checkpoints],
            "t_checkpoints": [int(t) for t in self._t_checkpoints],
            "results": serializable_results
        }

        if f is None:

            """Convert the object to a JSON string."""
            return json.dumps(d, default=str)
        json.dump(d, f, default=str)

    @classmethod
    def unserialize(cls, src):

        def convert_keys_to_int(pairs):
            return {int(k) if k.isdigit() else k: v for k, v in pairs}
        
        if type(src) == str:
            """Convert a JSON string back to an object."""
            data = json.loads(src, object_pairs_hook=convert_keys_to_int)
        else:
            data = json.load(src, object_pairs_hook=convert_keys_to_int)
        
        # true param values from Decimal str to float
        data["true_param_values"] = {
            k: np.array([float(u) if k != "V[Z_nt]" else [float(k) for k in u] for u in v])
            for k, v in data["true_param_values"].items()
        }
        
        data["results"] = pd.read_json(StringIO(data["results"]), dtype=object).astype({
            "budget": "int64",
            "t": "int64",
            "n": "Int64" # Int64 to handle nans
        })
        data["results"]["n"] = data["results"]["n"].replace({pd.NA: None, np.nan: None})
        data["results"]["n"] = data["results"]["n"].astype("object" if any(v is None for v in data["results"]["n"]) else "int64")
        print(data["results"]["estimate"])
        data["results"]["estimate"] = data["results"]["estimate"].apply(lambda x: float(Decimal(x)))
        data["results"]["runtime"] = data["results"]["runtime"].apply(lambda x: float(Decimal(x)))

        del data["precision"]
        return cls(**data)

    @classmethod
    def merge(cls, stores):

        if len(stores) < 2:
            raise ValueError(f"Need at least two result storages to merge.")

        # check that checkpoints coincide
        n_checkpoints = None
        t_checkpoints = None
        true_param_values = None
        for s in stores:
            if t_checkpoints is None:
                n_checkpoints = s.n_checkpoints
                t_checkpoints = s.t_checkpoints
                true_param_values = s.true_param_values
            else:
                if len(t_checkpoints) != len(s.t_checkpoints):
                    raise ValueError("Cannot merge results storages with different checkpoints")
                if np.any(t_checkpoints != s.t_checkpoints):
                    raise ValueError("Cannot merge results storages with different checkpoints")
                if len(n_checkpoints) != len(s.n_checkpoints):
                    raise ValueError("Cannot merge results storages with different checkpoints")
                if np.any(n_checkpoints != s.n_checkpoints):
                    raise ValueError("Cannot merge results storages with different checkpoints")
                for param in true_param_values:
                    if np.any(true_param_values[param] != s.true_param_values[param]):
                        raise ValueError(f"Cannot merge results storages with different True parameter values for param {param}.")
        if true_param_values is None:
            raise ValueError(f"Cannot merge result stores with None for true_param_value")

        # collect approach names
        df_results = pd.concat([s._results for s in stores], axis=0)
        return ResultStorage(
            true_param_values=true_param_values,
            n_checkpoints=n_checkpoints,
            t_checkpoints=t_checkpoints,
            results=df_results
        )
        
    
    def add_results(self, df):
        for i, row in df.iterrows():
            self.add_result(**row)
    
    def add_result(self, approach, budget, param, n, t, estimate, runtime):

        if param =="V[Z_nt]":
            if n not in self._n_checkpoints:
                raise ValueError(f"Unsupported value for {n=}. Should be in {self._n_checkpoints}")
        else:
            if n is not None:
                raise ValueError(f"no value of n should be given for parameters other than V[Z_nt] but saw {n=}")
        if t not in self._t_checkpoints:
            raise ValueError(f"Unsupported value for {t=}. Should be in {self._t_checkpoints}")

        if not isinstance(estimate, np.number) and type(estimate) != float:
            raise ValueError(f"Estimate should be float but got {type(estimate)}: {estimate}")
        key_cols = ["approach", "budget", "param", "n", "t"]
        val_cols = ["estimate", "runtime"]
        all_cols = key_cols + val_cols
        new_record = [approach, budget, param, n, t, estimate, runtime]
        new_recored_dict = {k: v for k, v in zip(all_cols, new_record)}
        if self._results is not None and np.any(np.all(self._results[key_cols].values == np.array([new_recored_dict[k] for k in key_cols]), axis=1)):
            red_dict = {k: new_recored_dict[k] for k in key_cols}
            raise ValueError(f"Double entry for record {red_dict}.")
        
        new_df = pd.DataFrame([new_record], columns=all_cols)
        self._results = new_df if self._results is None else pd.concat([self._results, new_df], ignore_index=True)
    
    def rename_approach(self, n_from, n_to):
        self._results.loc[self._results["approach"] == n_from, "approach"] = n_to
    
    def get_results_from_approach_for_checkpoint(self, approach_name, n_for_var_in_iid_case=None, t=None):
        
        # get all relevant n and t checkpoints
        if n_for_var_in_iid_case is not None:
            if not isinstance(n_for_var_in_iid_case, (list, np.ndarray)):
                n_for_var_in_iid_case = [n_for_var_in_iid_case]
        else:
            n_for_var_in_iid_case = self._n_checkpoints
        
        if t is not None:
            if not isinstance(t, (list, np.ndarray)):
                t = [t]
        else:
            t = self.t_checkpoints
        
        return self._results[
            (self._results["approach"] == approach_name) &
            (self._results["n"].isna() | self._results["n"].isin(n_for_var_in_iid_case)) &
            (self._results["t"].isin(t))
        ]
    
    def get_ground_truth_param_for_checkpoint(self, n_for_var_in_iid_case=None, t=None):
        
        # get all relevant n and t checkpoints
        if n_for_var_in_iid_case is not None:
            if not isinstance(n_for_var_in_iid_case, (list, np.ndarray)):
                n_for_var_in_iid_case = [n_for_var_in_iid_case]
        else:
            n_for_var_in_iid_case = self._n_checkpoints
        
        if t is not None:
            if not isinstance(t, (list, np.ndarray)):
                t = [t]
        else:
            t = self.t_checkpoints
        
        out = {}
        for p, v in self._true_param_values.items():
            t_indices = [self._t_checkpoints.index(u) for u in t]
            if p == "V[Z_nt]":
                n_indices = [self._n_checkpoints.index(u) for u in n_for_var_in_iid_case]
                out[p] = v[n_indices][:, t_indices].reshape(len(n_indices), len(t_indices))
            else:
                out[p] = v[t_indices]
        return out

    def get_errors_from_approach_for_checkpoint(self, approach_name, n_for_var_in_iid_case=None, t=None):
        
        # get all relevant n and t checkpoints
        if n_for_var_in_iid_case is not None:
            if not isinstance(n_for_var_in_iid_case, (list, np.ndarray)):
                n_for_var_in_iid_case = [n_for_var_in_iid_case]
        else:
            n_for_var_in_iid_case = self._n_checkpoints
        
        if t is not None:
            if not isinstance(t, (list, np.ndarray)):
                t = [t]
        else:
            t = self.t_checkpoints

        estimates = self.get_results_from_approach_for_checkpoint(approach_name=approach_name, n_for_var_in_iid_case=n_for_var_in_iid_case, t=t).copy()
        estimates["error"] = self.compute_error(estimates)
        return estimates
    
    def compute_error(self, df):

        def _f(r):
            pred = r["estimate"]
            if r["param"] == "V[Z_nt]":
                act = self._true_param_values[r["param"]][self.n_checkpoints.index(r["n"]), self.t_checkpoints.index(r["t"])]
            else:
                act = self._true_param_values[r["param"]][self.t_checkpoints.index(r["t"])]
            return pred - act
        return df.apply(_f, axis=1)

    def get_errors_on_highest_budget(self, params=None):
        out = []
        for (approach, param, t), df_approach in self._results.groupby(["approach", "param", "t"]):
            if param == "V[Z_nt]":
                for n, df_approach in df_approach.groupby("n"):
                    b = df_approach["budget"].max()
                    df_approach = df_approach[df_approach["budget"] == b]
                    assert len(df_approach) == 1
                    row = df_approach.iloc[0]                    
                    out.append([approach, b, param, n, t, row["estimate"], row["runtime"]])
            else:
                b = df_approach["budget"].max()
                df_approach = df_approach[df_approach["budget"] == b]
                assert len(df_approach) == 1
                row = df_approach.iloc[0]
                out.append([approach, b, param, None, t, row["estimate"], row["runtime"]])
        df = pd.DataFrame(out, columns=["approach", "budget", "param", "n", "t", "estimate", "runtime"])
        if params is not None:
            df = df[df["param"].isin(params)]
        df["error"] = self.compute_error(df)
        return df



