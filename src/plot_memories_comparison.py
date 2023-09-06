import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('log_rf_name', type=str)
parser.add_argument('log_carve_name', type=str)
parser.add_argument("thresholds", type=str)
parser.add_argument('plot_name', type=str)
args = parser.parse_args()

data_rf = pd.read_csv(args.log_rf_name, index_col=0)
data_carve = pd.read_csv(args.log_carve_name, index_col=0)

#threshold = data.columns.tolist()[1:]
thresholds_int = list(map(lambda x: int(x), args.thresholds.split("-")))
n_instances = data_rf.shape[0]
data_rf.memory = data_rf.memory*(2**20)/(10**6) #GB

instances_x_threshold_rf = [(data_rf.memory <= thr).sum() for thr in thresholds_int]
instances_x_threshold_carve = [(data_carve.memory <= thr).sum() for thr in thresholds_int]

print(instances_x_threshold_rf)
print(instances_x_threshold_carve)

x_vals = np.arange(-5, thresholds_int[-1]+5, 1)
y_vals = np.array([n_instances]*len(x_vals))

fig1, ax1 = plt.subplots(figsize=(8,6))

ax1.plot(x_vals, y_vals, '--', alpha=0.7, color="orange", label="\# instances", lw=4)
ax1.plot(thresholds_int, instances_x_threshold_rf, label="SILVA-RF", marker='o', ms=10, mew=4, mfc="w")
ax1.plot(thresholds_int, instances_x_threshold_carve, label="CARVE-LSE", color="red", marker='o', ms=10, mew=4, mfc="w")
ax1.set_xticks(thresholds_int)
ax1.set_xlim(-1, 105)
ax1.set_xlabel("Maximum memory consumption limit (GB)", fontsize=18)
ax1.set_ylabel("\# verified instances", fontsize=18)
ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)
ax1.legend(fontsize=18)

plt.grid()
ax1.xaxis.grid(linestyle="dotted")
ax1.yaxis.grid(linestyle="dotted")

ax1.set_title("MNIST", loc='left', fontsize=20)
ax1.set_title("\#trees={}, \#depth={}".format(101, 6), loc='right', fontsize=18)
plt.savefig(args.plot_name + ".pdf", bbox_inches="tight")