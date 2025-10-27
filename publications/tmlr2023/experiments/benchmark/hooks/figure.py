
        history = {}
        def hook(approaches: Dict[str, Approach], result_store):
            df = result_store.get_errors_on_highest_budget()

            for param, df_param in df.groupby("param"):

                if param == "E[Z_nt]":
                    ground_truth = pi.means_iid
                elif param == "E[Z_nt|D_val]":
                    ground_truth = pi.means_cond
                elif param == "V[Z_nt]":
                    ground_truth = pi.vars_iid[0]
                elif param == "V[Z_nt|D_val]":
                    ground_truth = pi.vars_cond
                    print(df_param)
                else:
                    raise ValueError()
                
                if param not in history:
                    history[param] = {}
                
                param_history = history[param]

                margin = 10**-4 if param == "V[Z_nt|D_val]" else 0.1

                lower = ground_truth.min() - margin
                upper = ground_truth.max() + margin

                fig = plt.figure(figsize=(24, 12))
                gs = gridspec.GridSpec(5, 3, width_ratios=[1, 1, 1])

                if param == "V[Z_nt]":
                    df_param = df_param[df_param["n"] == 2]

                # actual approximations
                ax = fig.add_subplot(gs[0:2, 0])
                #domain = np.linspace(1, 1000, 20)
                domain = pi.t_checkpoints
                for a_name, df_approach in df_param.groupby("approach"):
                    estimates_of_approach_in_round = df_approach.sort_values("t")["estimate"]
                    ax.plot(domain, estimates_of_approach_in_round, marker="o")
                ax.scatter(pi.t_checkpoints, ground_truth, color="black", marker="*", s=100, label="Ground Truth", zorder=10**2)
                ax.grid()
                b = df_param['budget'].iloc[0]
                ax.axvline(b, color="black", linewidth=1, label="Budget $b$")
                ax.set_xscale("log")
                ax.set_xlabel("t")
                #ax.set_ylim([lower, upper])
                if "V[Z_nt" in param:
                    ax.set_ylim([10**-6, 10**-2])
                    ax.set_yscale("log")
                ax.set_title(f"Predictions after {b} built ensemble members")
                ax.legend()

                # errors per anchor on log-scale
                ax = fig.add_subplot(gs[2:4, 0])
                for a_name, df_approach in df_param.groupby("approach"):
                    if a_name not in param_history:
                        param_history[a_name] = []
                    error_of_approach_in_round = np.abs(df_approach.sort_values("t")["error"])
                    param_history[a_name].append(error_of_approach_in_round)
                    ax.scatter(pi.t_checkpoints, error_of_approach_in_round)
                ax.axvline(b, color="black", linewidth=1, label="Budget $b$")
                ax.legend()
                ax.set_xscale("log")
                ax.set_xlabel("t")
                ax.set_yscale("log")
                ax.set_title(f"MAE after {b} built ensemble members")
                ax.grid()
                ax.set_ylim([10**-5 if param != "V[Z_nt|D_val]" else 10**-7, 10**-1 if param != "V[Z_nt|D_val]" else 10**-2])

                # learning curves
                for i, t in enumerate(pi.t_checkpoints, start=1):
                    ax = fig.add_subplot(gs[i - 1, 1])
                    for a_name, history_of_approach in param_history.items():
                        history_of_approach_as_array = np.array(history_of_approach)[:, i-1]
                        ax.plot(range(1, history_of_approach_as_array.shape[0] + 1), history_of_approach_as_array, label=a_name)
                    ax.set_xlim([1, 10**3])
                    ax.set_xscale("log")
                    ax.set_xlabel("Budget $b$ (Number of built ensemble members)")
                    ax.set_yscale("log")
                    ax.set_title(f"MAE over time for t={int(t)}")
                    ax.grid()
                    ax.set_ylim([10**-5 if param != "V[Z_nt|D_val]" else 10**int(-5 - np.log10(t)), 10**-1 if param != "V[Z_nt|D_val]" else 10**-2])
                ax_with_labels_for_legend = ax

                # smoothened learning curves
                window_size = 10
                for i, t in enumerate(pi.t_checkpoints, start=1):
                    ax = fig.add_subplot(gs[i - 1, 2])
                    for a_name, history_of_approach in param_history.items():
                        history_of_approach_as_array = np.array(history_of_approach)[:, i - 1]
                        windows = sliding_window_view(history_of_approach_as_array, window_shape=min(len(history_of_approach_as_array), window_size))
                        if windows.shape[0] > 1:
                            ax.plot(range(window_size, window_size + windows.shape[0]), windows.mean(axis=1), label=a_name)
                    ax.set_xlim([1, 10**3])
                    ax.set_xscale("log")
                    ax.set_xlabel("Budget $b$ (Number of built ensemble members)")
                    ax.set_yscale("log")
                    ax.set_title(f"MAE over time for t={int(t)}")
                    ax.grid()
                    ax.set_ylim([10**-5 if param != "V[Z_nt|D_val]" else 10**int(-5 - np.log10(t)), 10**-1 if param != "V[Z_nt|D_val]" else 10**-2])
                
                # create global legend of shared items
                handles, labels = ax_with_labels_for_legend.get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))

                # save figure
                fig.tight_layout()
                path = pathlib.Path(f"plots/animation_{openmlid}/{param}/{ensemble_sequence_seed}_{b}.jpeg")
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, bbox_inches="tight")
                print(f"Stored to {path}")
                plt.close()