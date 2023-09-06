import subprocess
import argparse

_PARTITION_POLICY = "rand"
_DATASET_FOLDER = "../datasets/{}/dataset/"

parser = argparse.ArgumentParser()
parser.add_argument('datasets', type=str)
parser.add_argument('n_trees', type=str)
parser.add_argument('depths', type=str)
parser.add_argument('ks', type=str)
parser.add_argument('random_states', type=str)
parser.add_argument('rounds', type=str)
parser.add_argument('ms', type=str)
parser.add_argument('ss', type=str)
parser.add_argument('min_perts', type=str)
parser.add_argument('max_perts', type=str)
parser.add_argument("--validation", action="store_true")
args = parser.parse_args()

dataset_list = args.datasets.split("/")
n_tree_list = args.n_trees.split("-")
depth_list = args.depths.split("-")
k_list = args.ks.split("-")
round_list = args.rounds.split("-")
random_state_list = args.random_states.split("-")
m_list = args.ms.split("-")
s_list = args.ss.split("-")
min_pert_list = args.min_perts.split("-")
max_pert_list = args.max_perts.split("-")

for dataset in dataset_list:
	for n_tree in n_tree_list:
		for depth in depth_list:
			for k in k_list:
					for round in round_list:
						for random_state in random_state_list:
							for m in m_list:
								for s in s_list:
									for min_pert, max_pert in zip(min_pert_list, max_pert_list):
										print("training lse_part_{}_{}_{}_{}_{}_{}_m{}_s{}_mnpert{}_mxpert{}.silva".format(_PARTITION_POLICY, n_tree, depth, k, round, random_state, m, s, min_pert, max_pert))
										if args.validation:
											p = subprocess.run(["/usr/bin/time", "python3", "LSE.py", k, "--dataset", dataset, "--trees", n_tree, "--depth", depth, "--rounds", round, "--dump_large_spread", str(1), "--m", m, "--s", s, "--random_seed", random_state, "--min_pert", min_pert, "--max_pert", max_pert, "--validation"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
										else:
											p = subprocess.run(["/usr/bin/time", "python3", "LSE.py", k, "--dataset", dataset, "--trees", n_tree, "--depth", depth, "--rounds", round, "--dump_large_spread", str(1), "--m", m, "--s", s, "--random_seed", random_state, "--min_pert", min_pert, "--max_pert", max_pert], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
										print("Put the log in ", "log_{}_lse_part_{}_{}_{}_{}_{}_{}_m{}_s{}_mnpert{}_mxpert{}.txt".format(dataset, _PARTITION_POLICY, n_tree, depth, k, round, random_state, m, s, min_pert, max_pert))
										f = open(("../datasets/{}/models/lse/validation/" if args.validation else "../datasets/{}/models/lse/").format(dataset) + "log_{}_lse_part_{}_{}_{}_{}_{}_{}_m{}_s{}_mnpert{}_mxpert{}.txt".format(dataset,  _PARTITION_POLICY, n_tree, depth, k, round, random_state, m, s, min_pert, max_pert), "w")
										f.write(p.stdout.decode())
										f.write(p.stderr.decode())
										f.close()

