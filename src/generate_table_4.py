import subprocess
import argparse
import numpy as np
import pandas as pd

models_x_datasets = {
	"mnist26" :	["lse_part_rand_25_4_0.005_100_0_m6_s1_mnpert0.0025_mxpert0.005.silva", "lse_part_rand_101_6_0.005_500_0_m6_s1_mnpert0.0025_mxpert0.005.silva"],#, "lse_part_rand_25_4_0.01_100_0_m2_s1_mnpert0.005_mxpert0.01.silva", "lse_part_rand_101_6_0.01_500_0_m6_s1_mnpert0.005_mxpert0.01.silva", "lse_part_rand_25_4_0.015_100_0_m2_s1_mnpert0.0075_mxpert0.015.silva", "lse_part_rand_101_6_0.015_500_0_m2_s4_mnpert0.0075_mxpert0.015.silva"],
	"fashion_mnist0-3": ["lse_part_rand_25_4_0.005_100_0_m4_s1_mnpert0.0025_mxpert0.005.silva", "lse_part_rand_101_6_0.005_500_0_m6_s1_mnpert0.0025_mxpert0.005.silva", "lse_part_rand_25_4_0.01_100_0_m4_s1_mnpert0.005_mxpert0.01.silva", "lse_part_rand_101_6_0.01_500_0_m6_s1_mnpert0.005_mxpert0.01.silva", "lse_part_rand_25_4_0.015_100_0_m6_s1_mnpert0.015_mxpert0.0225.silva", "lse_part_rand_101_6_0.015_100_0_m4_s5_mnpert0.0075_mxpert0.015.silva"],
	"rewema": ["lse_part_rand_25_4_0.005_100_0_m2_s1_mnpert0.0025_mxpert0.005.silva", "lse_part_rand_101_6_0.005_500_0_m2_s1_mnpert0.0025_mxpert0.005.silva", "lse_part_rand_25_4_0.01_100_0_m6_s1_mnpert0.01_mxpert0.015.silva", "lse_part_rand_101_6_0.01_500_0_m4_s1_mnpert0.01_mxpert0.015.silva", "lse_part_rand_25_4_0.015_100_0_m6_s1_mnpert0.015_mxpert0.0225.silva", "lse_part_rand_101_6_0.015_500_0_m4_s2_mnpert0.015_mxpert0.0225.silva"],
	"webspam": ["lse_part_rand_25_4_0.0002_100_0_m2_s1_mnpert0.0002_mxpert0.0003.silva", "lse_part_rand_101_6_0.0002_500_0_m4_s5_mnpert0.0001_mxpert0.0002.silva", "lse_part_rand_25_4_0.0004_100_0_m2_s1_mnpert0.0002_mxpert0.0004.silva", "lse_part_rand_101_6_0.0004_500_0_m4_s6_mnpert0.0002_mxpert0.0004.silva", "lse_part_rand_25_4_0.0006_100_0_m2_s1_mnpert0.0006_mxpert0.0009.silva", "lse_part_rand_101_6_0.0006_500_0_m4_s6_mnpert0.0003_mxpert0.0006.silva"]
}

sizes_x_datasets = {
	"mnist26": [(25, 4), (101, 6), (25, 4), (101, 6), (25, 4), (101, 6)],
	"fashion_mnist0-3": [(25, 4), (101, 6), (25, 4), (101, 6), (25, 4), (101, 6)],
	"rewema": [(25, 4), (101, 6), (25, 4), (101, 6), (25, 4), (101, 6)],
	"webspam": [(25, 4), (101, 6), (25, 4), (101, 6), (25, 4), (101, 6)]
}

ks = {
	"mnist26" : [0.005, 0.005, 0.01, 0.01,  0.015, 0.015],
	"fashion_mnist0-3": [0.005, 0.005, 0.01, 0.01,  0.015, 0.015],
	"rewema": [0.005, 0.005, 0.01, 0.01,  0.015, 0.015],
	"webspam": [0.0002, 0.0002, 0.0004, 0.0004, 0.0006, 0.0006]
}

_LSE_PATH = "../datasets/{}/models/lse/validation/"
_TEST_SET_PATH = "../datasets/{}/dataset/test_set_normalized.csv"

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('p_norms', type=str)
parser.add_argument("time_limit", type=str)
args = parser.parse_args()

dataset = args.dataset
p_norms_list = args.p_norms.split("-")
time_limit = args.time_limit

column_names = ["dataset", "perturbation", "n_trees", "depth", "rob_lse"]

table = [column_names]

def get_metrics_from_carve(model_filename, dataset_filename, k, p_norm, n_instances, time_limit):
	print(model_filename, " examined!")

	accurate_instances = 0
	robust_instances = 0
	no_info_instances = 0

	for i in range(n_instances):
		p = subprocess.run(["timeout", "--signal=9", time_limit, "./carve/build/verify", "-i", model_filename, "-t", dataset_filename, "-p", p_norm, "-k", k, "-ioi", str(i)], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
		output = p.stdout.decode()

		if len(output) == 0:
			no_info_instances += 1
		else:
			output = output.splitlines()
			accurate_instances += int(output[-7].split(": ")[1])
			robust_instances += int(output[-4].split(": ")[1])
	
	return accurate_instances/n_instances, robust_instances/n_instances, no_info_instances/n_instances

test_set_filename = _TEST_SET_PATH.format(dataset)
n_instances = pd.read_csv(test_set_filename).shape[0]
lse_models = models_x_datasets[dataset]
sizes = sizes_x_datasets[dataset]
ks_dataset = ks[dataset]

for p_norm in p_norms_list:
	column_names.append(p_norm)

for i in range(len(lse_models)):
	lse_model = _LSE_PATH.format(dataset) + lse_models[i]
	n_trees, depth = sizes[i]
	k = ks_dataset[i]
	row = [dataset, k, n_trees, depth]
	for p_norm in p_norms_list:
		#examine lse
		lse_acc_temp, lse_rob_temp, lse_no_info_temp = get_metrics_from_carve(lse_model, test_set_filename, str(k), p_norm, n_instances, time_limit)
		row += [str(lse_rob_temp + lse_no_info_temp/2) + " +- " + str(lse_no_info_temp/2)]
	table.append(row)

print(np.matrix(table))