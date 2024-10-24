from utils_ import Analyzer
import matplotlib.pyplot as plt
import numpy as np
import pathlib

print("Creating analyzer.")

datasets = []
for x in pathlib.Path("data").iterdir():
    name = x.name
    if x.is_file() and ".json.gz" in name:
        if x.stat().st_size / 1024**2 < 20:
            datasets.append(int(name[:name.index("_")]))


for openmlid in sorted(datasets):
    print(openmlid)
    if pathlib.Path(f"plots/{openmlid}.pdf").exists():
        continue
    analyzer = Analyzer.for_dataset(openmlid)

    print("Analyzer created. Now preparing figure.")
    for strategy in ["sample"]:#ś, "bound"]:
        fig = analyzer.create_full_belief_plot(alpha=0.99, eps=0.01, var_zt_estimation_strategy=strategy)
        fig.savefig(f"plots/{openmlid}.pdf")
    #plt.show()
