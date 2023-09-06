import subprocess
import argparse

_PARTITION_POLICY = "rand"

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
parser.add_argument('--validation', action="store_true")
args = parser.parse_args()

_LARGE_SPREAD_FILENAME = "lse_part_{}_{}_{}_{}_{}_{}_m{}_s{}_mnpert{}_mxpert{}.silva"

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

_DATASET_FOLDER = "../datasets/{}/dataset/"
_MODEL_FOLDER = "../datasets/{}/models/lse/"
_LSE_LOG_FILENAME = "../datasets/{}/models/lse/validation/log_{}_lse_part_rand_{}_{}_{}_{}_{}_m{}_s{}_mnpert{}_mxpert{}.txt"

for dataset in dataset_list:
        test_set_filename = _DATASET_FOLDER.format(dataset) + ("validation_set_normalized_gridsearch.csv" if args.validation else "test_set_normalized.csv")
        for n_tree in n_tree_list:
                for depth in depth_list:
                        for k in k_list:
                                for round in round_list:
                                        for random_state in random_state_list:
                                                for m in m_list:
                                                        for s in s_list:
                                                                for min_pert, max_pert in zip(min_pert_list, max_pert_list):
                                                                    model_found = True
                                                                    try:
                                                                                f = open(_LSE_LOG_FILENAME.format(dataset, dataset, n_tree, depth, k,round, random_state, m, s, float(min_pert), float(max_pert)), "r")
                                                                                f.close()
                                                                    except:
                                                                                model_found = False

                                                                    lse_filename = _MODEL_FOLDER.format(dataset) + ("validation/" if args.validation else "") + _LARGE_SPREAD_FILENAME.format(_PARTITION_POLICY, n_tree, depth, k,round, random_state, m, s, float(min_pert), float(max_pert))
                                                                    if model_found:
                                                                                print(lse_filename)
                                                                                p = subprocess.run(["./silva/src/silva", lse_filename, test_set_filename, "--perturbation", "l_inf", k], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
                                                                                print(_MODEL_FOLDER.format(dataset) + ("validation/" if args.validation else "") + "silva_{}_lse_part_{}_{}_{}_{}_{}_{}_m{}_s{}_mnpert{}_mxpert{}.txt".format(dataset, _PARTITION_POLICY, n_tree, depth, k, round, random_state, m, s, float(min_pert), float(max_pert)), "w")
                                                                    else:
                                                                                print(lse_filename, " not found!")
                                                                    f = open(_MODEL_FOLDER.format(dataset) + ("validation/" if args.validation else "") + "silva_{}_lse_part_{}_{}_{}_{}_{}_{}_m{}_s{}_mnpert{}_mxpert{}.txt".format(dataset, _PARTITION_POLICY, n_tree, depth, k, round, random_state, m, s, float(min_pert), float(max_pert)), "w")
                                                                    if model_found:
                                                                                f.write(p.stdout.decode())
                                                                                f.write(p.stderr.decode())
                                                                    else:
                                                                                f.write("Model not trained")
                                                                    f.close()
