import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt

_PARTITION_POLICY = "rand"
_DATASET_LABELS = {"fashion_mnist0-3": "Fashion-MNIST", "rewema" : "REWEMA", "mnist26" : "MNIST", "webspam": "Webspam"}
_PERT_LABELS = {"0.015" : "0.0150", "0.01": "0.0100", "0.005": "0.0050", "0.0002": "0.0002", "0.0004": "0.0004", "0.0006": "0.0006"}

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('n_trees', type=str)
parser.add_argument('depths', type=str)
parser.add_argument('ks', type=str)
parser.add_argument('random_state', type=str)
parser.add_argument('round', type=str)
parser.add_argument('m', type=str)
parser.add_argument('s', type=str)
parser.add_argument('min_perts', type=str)
parser.add_argument('max_perts', type=str)
parser.add_argument('plot_name', type=str)
args = parser.parse_args()

_LARGE_SPREAD_FILENAME = "../datasets/{}/models/lse/validation/log_{}_lse_part_{}_{}_{}_{}_{}_{}_m{}_s{}_mnpert{}_mxpert{}.txt"

dataset = args.dataset
n_trees = args.n_trees
depths = args.depths.split("-")
k_list = args.ks.split("-")
round = args.round
random_state = args.random_state
m = args.m
s = args.s
min_perts = args.min_perts.split("-")
max_perts = args.max_perts.split("-")

times = []
for i, k in enumerate(k_list):
        time_x_k = []
        for depth in depths:
                print(depth)
                lse_filename = _LARGE_SPREAD_FILENAME.format(dataset, dataset, _PARTITION_POLICY, n_trees, depth, k,round, random_state, m, s, min_perts[i], max_perts[i])
                print(lse_filename)
                p = subprocess.run(["tail", "-n", "2", lse_filename], stdout = subprocess.PIPE)
                time_string = p.stdout.decode().split(" ")[2].split("e")[0]
                minutes_seconds, centiseconds = time_string.split(".")
                minutes, seconds = minutes_seconds.split(":")
                time_in_seconds = int(minutes)*60 + int(seconds) + int(centiseconds)*0.01
                time_x_k.append(time_in_seconds)
        times.append(time_x_k)

print(times)

fig1, ax1 = plt.subplots(figsize=(8,6))

for i, k in enumerate(k_list):
        ax1.plot(depths, times[i], marker='o', ms=10, mew=4, mfc="w", label="LSE $k = {}$".format(_PERT_LABELS[k]))

ax1.set_xlabel("depth", fontsize=18)
ax1.set_ylabel("time (s)", fontsize=18)
ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)
ax1.legend(fontsize=18)

if dataset == "rewema":
        ax1.set_ylim(bottom=-0.5)
        ax1.set_ylim(top=15)
if dataset == "fashion_mnist0-3":
        ax1.set_ylim(bottom=-4)
        ax1.set_ylim(top=130)
if dataset == "mnist26":
        ax1.set_ylim(bottom=-5)
        ax1.set_ylim(top=150)
if dataset == "webspam":
        ax1.set_ylim(bottom=-70)
        ax1.set_ylim(top=1920)

plt.grid()
ax1.xaxis.grid(linestyle="dotted")
ax1.yaxis.grid(linestyle="dotted")

ax1.set_title(_DATASET_LABELS[dataset], loc='left', fontsize=20)
ax1.set_title("#trees={}".format(n_trees), loc='right', fontsize=18)
plt.savefig(args.plot_name + ".pdf", bbox_inches="tight")

