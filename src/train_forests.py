import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('datasets', type=str)
parser.add_argument('n_trees', type=str)
parser.add_argument('depths', type=str)
parser.add_argument('random_states', type=str)
parser.add_argument('--n_jobs', type=str, default="1")
args = parser.parse_args()

dataset_list = args.datasets.split("/")
n_tree_list = args.n_trees.split("-")
depth_list = args.depths.split("-")
random_state_list = args.random_states.split("-")

for dataset in dataset_list:
	print("DATASET: ", dataset)
	for n_tree in n_tree_list:
		for depth in depth_list:
			for random_state in random_state_list:
				print("training rf_{}_{}_{}.txt".format(n_tree, depth, random_state))
				p = subprocess.run(["python3", "train_forest.py", dataset, n_tree, depth, random_state, "--n_jobs", args.n_jobs], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
