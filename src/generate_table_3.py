import subprocess
import argparse
import numpy as np
import pandas as pd

models_x_datasets = {
	"mnist26" : [
		["lse_part_rand_25_4_0.005_100_0_m6_s1_mnpert0.0025_mxpert0.005.silva", "lse_part_rand_101_6_0.005_500_0_m6_s1_mnpert0.0025_mxpert0.005.silva", "lse_part_rand_25_4_0.01_100_0_m2_s1_mnpert0.005_mxpert0.01.silva", "lse_part_rand_101_6_0.01_500_0_m6_s1_mnpert0.005_mxpert0.01.silva", "lse_part_rand_25_4_0.015_100_0_m2_s1_mnpert0.0075_mxpert0.015.silva", "lse_part_rand_101_6_0.015_500_0_m2_s4_mnpert0.0075_mxpert0.015.silva"], 
		["rf_25_4_0_valFalse.silva", "rf_101_6_0_valFalse.silva", "rf_25_4_0_valFalse.silva", "rf_101_6_0_valFalse.silva", "rf_25_4_0_valFalse.silva", "rf_101_6_0_valFalse.silva"]
	],
	"fashion_mnist0-3": [
		["lse_part_rand_25_4_0.005_100_0_m4_s1_mnpert0.0025_mxpert0.005.silva", "lse_part_rand_101_6_0.005_500_0_m6_s1_mnpert0.0025_mxpert0.005.silva", "lse_part_rand_25_4_0.01_100_0_m4_s1_mnpert0.005_mxpert0.01.silva", "lse_part_rand_101_6_0.01_500_0_m6_s1_mnpert0.005_mxpert0.01.silva", "lse_part_rand_25_4_0.015_100_0_m6_s1_mnpert0.015_mxpert0.0225.silva", "lse_part_rand_101_6_0.015_100_0_m4_s5_mnpert0.0075_mxpert0.015.silva"], 
		["rf_25_4_0_valFalse.silva", "rf_101_6_0_valFalse.silva", "rf_25_4_0_valFalse.silva", "rf_101_6_0_valFalse.silva", "rf_25_4_0_valFalse.silva", "rf_101_6_0_valFalse.silva"]
	],
	"rewema": [
		["lse_part_rand_25_4_0.005_100_0_m2_s1_mnpert0.0025_mxpert0.005.silva", "lse_part_rand_101_6_0.005_500_0_m2_s1_mnpert0.0025_mxpert0.005.silva", "lse_part_rand_25_4_0.01_100_0_m6_s1_mnpert0.01_mxpert0.015.silva", "lse_part_rand_101_6_0.01_500_0_m4_s1_mnpert0.01_mxpert0.015.silva", "lse_part_rand_25_4_0.015_100_0_m6_s1_mnpert0.015_mxpert0.0225.silva", "lse_part_rand_101_6_0.015_500_0_m4_s2_mnpert0.015_mxpert0.0225.silva"], 
		["rf_25_4_0_valFalse.silva", "rf_101_6_0_valFalse.silva", "rf_25_4_0_valFalse.silva", "rf_101_6_0_valFalse.silva", "rf_25_4_0_valFalse.silva", "rf_101_6_0_valFalse.silva"]
	],
	"webspam": [
		["lse_part_rand_25_4_0.0002_100_0_m2_s1_mnpert0.0002_mxpert0.0003.silva", "lse_part_rand_101_6_0.0002_500_0_m4_s5_mnpert0.0001_mxpert0.0002.silva", "lse_part_rand_25_4_0.0004_100_0_m2_s1_mnpert0.0002_mxpert0.0004.silva", "lse_part_rand_101_6_0.0004_500_0_m4_s6_mnpert0.0002_mxpert0.0004.silva", "lse_part_rand_25_4_0.0006_100_0_m2_s1_mnpert0.0006_mxpert0.0009.silva", "lse_part_rand_101_6_0.0006_500_0_m4_s6_mnpert0.0003_mxpert0.0006.silva"], 
		["rf_25_4_0_valFalse.silva", "rf_101_6_0_valFalse.silva", "rf_25_4_0_valFalse.silva", "rf_101_6_0_valFalse.silva", "rf_25_4_0_valFalse.silva", "rf_101_6_0_valFalse.silva"]
	]
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

_RF_PATH = "../datasets/{}/models/rf/"
_LSE_PATH = "../datasets/{}/models/lse/validation/"
_TEST_SET_PATH = "../datasets/{}/dataset/test_set_normalized.csv"

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument("time_limit", type=str)
args = parser.parse_args()

dataset = args.dataset
time_limit = args.time_limit

column_names = ["dataset", "n_trees", "depth", "acc standard", "acc_lse", "rob_standard", "rob_lse"]

table = [column_names]

def get_metrics_from_silva(model_filename, dataset_filename, k, n_instances):
	print(model_filename, " examined!")
	accuracy = 0
	p = subprocess.run(["./silva/src/silva", model_filename, dataset_filename, "--perturbation", "l_inf", k], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
	output_err = p.stderr.decode()
	if "Cannot read file" not in output_err:
		output = p.stdout.splitlines()[-1].decode().split(" ")
		output = list(filter(lambda a: a != '', output))
		correct = int(output[3])
		accuracy = correct/n_instances
	else:
		return 0, 0, 0

	robust_instances = 0
	no_info_instances = 0

	for i in range(n_instances):
		p = subprocess.run(["timeout", "--signal=9", args.time_limit, "./silva/src/silva", model_filename, dataset_filename, "--perturbation", "l_inf", k, "--sample-timeout", str(int(args.time_limit)+30), "--index-of-instance", str(i)], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
		output = p.stdout.decode()

		if len(output) == 0:
			no_info_instances += 1
		else:
			output = output.splitlines()[-1].split(" ")
			output = list(filter(lambda a: a != '', output))
			#print(output)
			robust_instances += int(output[8])
	
	return accuracy, robust_instances/n_instances, no_info_instances/n_instances


test_set_filename = _TEST_SET_PATH.format(dataset)
n_instances = pd.read_csv(test_set_filename).shape[0]
lse_models = models_x_datasets[dataset][0]
rf_models = models_x_datasets[dataset][1]
sizes = sizes_x_datasets[dataset]
ks_dataset = ks[dataset]

for i in range(len(lse_models)):
	lse_model = _LSE_PATH.format(dataset) + lse_models[i]
	rf_model = _RF_PATH.format(dataset) + rf_models[i]
	n_trees, depth = sizes[i]
	k = ks_dataset[i]
	row = [dataset, n_trees, depth]
	accs = []
	robs = []
	no_info = []
	#examine lse
	lse_acc_temp, lse_rob_temp, lse_no_info_temp = get_metrics_from_silva(lse_model, test_set_filename, str(k), n_instances)
	accs.append(lse_acc_temp)
	robs.append(lse_rob_temp)
	no_info.append(lse_no_info_temp)
	#examine rf
	rf_acc, rf_rob, rf_no_info = get_metrics_from_silva(rf_model, test_set_filename, str(k), n_instances)
	
	row += [rf_acc, lse_acc_temp, str(rf_rob + rf_no_info/2) + " +- " + str(rf_no_info/2), str(lse_rob_temp + lse_no_info_temp/2) + " +- " + str(lse_no_info_temp/2)]

	table.append(row)

print(np.matrix(table))
