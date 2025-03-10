
    fig, axs = plt.subplots(1, 2, sharey=True)
    for a_name in approaches.keys():
        t = t_checkpoints[0]
        a_errors = b.result_storage.get_errors_from_approach_for_checkpoint(a_name, t=t)
        axs[0].plot(a_errors["E[Z_nt]"], label=a_name, alpha=0.5)
        #axs[1].plot(errors[1], label=t_checkpoints)
    for ax in axs.flatten():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
        ax.legend()
    plt.show()

    

    if False:

        # compute error curves
        error_history_by_approach = {
            a_name: [] for a_name in approaches.keys()
        }

        relevant_keys = ["performance_mean"]

        for i in tqdm(range(10**2)):
            for a_name, errors_for_approach in next(gen).items():
                error_history_by_approach[a_name].append([errors_for_approach[k] for k in relevant_keys])
        error_history_by_approach = {
            a_name: np.array(errors).transpose(1, 0, 2)
            for a_name, errors in error_history_by_approach.items()
        }

        linestyles = {
            "a": "dotted",
            "bootstrapping": "solid",
            "theorem with datasets": "dashed"
        }

        fig, axs = plt.subplots(1, 2, sharey=True)
        for a_name, a_errors in error_history_by_approach.items():
            axs[0].plot(a_errors[0], label=[f"{t} ({a_name})" for t in t_checkpoints], linestyle=linestyles[a_name], alpha=0.5)
            #axs[1].plot(errors[1], label=t_checkpoints)
        for ax in axs.flatten():
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid()
            ax.legend()
        plt.show()
