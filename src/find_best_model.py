import subprocess
import argparse
import numpy as np

_PARTITION_POLICY = "rand"
_MODEL_FOLDER = "../datasets/{}/models/lse/validation/"
_DATASET_PATH = "../datasets/{}/dataset/validation_set_normalized_gridsearch.csv"
_LSE_LOG = "silva_{}_lse_part_{}_{}_{}_{}_{}_{}_m{}_s{}_mnpert{}_mxpert{}.txt"

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('n_trees', type=str)
parser.add_argument('depth', type=str)
parser.add_argument("k", type=float)
parser.add_argument("random_state", type=str)
parser.add_argument("rounds", type=str)
parser.add_argument("ms", type=str)
parser.add_argument("ss", type=str)
parser.add_argument("min_perts_factors", type=str)
parser.add_argument("max_perts_factors", type=str)
args = parser.parse_args()

dataset = args.dataset
n_trees = args.n_trees
depth = args.depth
k = args.k
random_state = args.random_state
ms = args.ms.split("-")
ss = args.ss.split("-")
rounds = args.rounds.split("-")
min_perts_factors = list(map(float, args.min_perts_factors.split("-")))
max_perts_factors = list(map(float, args.max_perts_factors.split("-")))

def get_stats_from_report(lse_log):
	p = subprocess.run(["tail", "-n", "1", lse_log], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
	output = p.stdout.decode()
	output_err = p.stderr.decode()
	if "[SUMMARY]" not in output:
		if "Cannot read file" in output:
			return -2, -2, -2
		elif "Model not trained" in output:
			return -1, -1, -1
		else:
			raise NotImplementedError()

	output = output.splitlines()[-1].split(" ")[1:]
	output = list(filter(lambda a: a != '', output))

	total_instances = int(output[0])
	acc_instances = int(output[2])
	robust_instances = int(output[7])
	no_info_instances = int(output[6])

	return acc_instances/total_instances, robust_instances/total_instances, no_info_instances/total_instances

best_acc = 0
best_rob = 0
best_no_info = 0
best_mult = None
best_l = None
best_rounds = None
best_min_pert = None
best_max_pert = None

for round in rounds:
	for m in ms:
		for s in ss:
			for min_pert_factor, max_pert_factor in zip(min_perts_factors, max_perts_factors):
				lse_log = _MODEL_FOLDER.format(dataset) + _LSE_LOG.format(dataset, _PARTITION_POLICY, n_trees, depth, k, round, random_state, m, s, str(k*min_pert_factor)[:6], str(k*max_pert_factor)[:6])
				accuracy, robustness, no_info = get_stats_from_report(lse_log)
				info_str = ""
				print("mult: ", m, "; l: ", s,\
         "; rounds: ", round, "; min_pert: ", min_pert_factor, "; max_pert: ", max_pert_factor, "; acc: ", accuracy, "; rob: ", robustness)
				if accuracy >= 0:
					if (accuracy+robustness)/2 > (best_acc+best_rob)/2:
						best_mult = m
						best_l = s
						best_rounds = round
						best_min_pert = min_pert_factor
						best_max_pert = max_pert_factor
						best_acc = accuracy
						best_rob = robustness
						best_no_info = no_info
				elif accuracy == -2:
					print("Error: ", lse_log, " cannot read model.")
				else:
					print("Error: ", lse_log, " not found.")

print("BEST MODEL FOR dataset: ", dataset, "; n_tree: ", n_trees, "; depth: ", depth, "; k: ", k, "; random_state: ", random_state, "; mult: ", best_mult, "; l: ", best_l,\
	 "; rounds: ", best_rounds, "; min_pert: ", best_min_pert, "; max_pert: ", best_max_pert, "; acc: ", best_acc, "; rob: ", best_rob + best_no_info/2, "+-", best_no_info/2)

print("Train best model")

p = subprocess.run(["python3", "train_lses.py", dataset, n_trees, depth, str(k), random_state, best_rounds, best_mult, best_l, str(k*best_min_pert)[:6], str(k*best_max_pert)[:6]])
