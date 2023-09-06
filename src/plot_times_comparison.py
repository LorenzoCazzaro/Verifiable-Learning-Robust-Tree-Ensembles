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

data_rf = pd.read_csv(args.log_rf_name)
data_lse = pd.read_csv(args.log_carve_name)


thresholds_int = list(map(lambda x: int(x), args.thresholds.split("-")))
n_instances = data_rf.shape[0]

instances_x_threshold_rf = [0]*len(thresholds_int)
instances_x_threshold_carve = [0]*len(thresholds_int)

times_rf = data_rf.times
oom_rf = data_rf.out_of_memory

times_carve = data_lse.times
oom_carve = data_lse.out_of_memory

for i in range(len(thresholds_int)):
	for j in range(n_instances):
		if not oom_rf[j] and times_rf[j] <= thresholds_int[i]:
			instances_x_threshold_rf[i] += 1

		if not oom_carve[j] and times_carve[j] <= thresholds_int[i]:
			instances_x_threshold_carve[i] += 1

print(instances_x_threshold_rf)
print(instances_x_threshold_carve)

fig1, ax1 = plt.subplots(figsize=(8,6))

x_vals = np.arange(-100, thresholds_int[-1]+100, 1)
y_vals = np.array([n_instances]*len(x_vals))

ax1.plot(x_vals, y_vals, '--', alpha=0.7, color="orange", label="\# instances", lw=4)
ax1.plot(thresholds_int, instances_x_threshold_rf, label="SILVA-RF", marker='o', ms=10, mew=4, mfc="w")
ax1.plot(thresholds_int, instances_x_threshold_carve, label="CARVE-LSE", color="red", marker='o', ms=10, mew=4, mfc="w")
ax1.set_xlabel("Time (s)", fontsize=18)
ax1.set_xlim(-25, 625)
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