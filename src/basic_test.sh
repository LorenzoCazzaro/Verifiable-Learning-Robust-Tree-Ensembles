#!/bin/sh

#Train a small large-spread ensemble of 25 trees, maximum depth 4 and perturbation 0.015-
python3 LSE.py 0.015 --dataset mnist26 --trees 25 --depth 4 --rounds 100 --random_seed 0 --m 2 --s 4 --jobs 6 --dump_large_spread 1 --min_pert 0.0075 --max_pert 0.015 --validation

#Train a small RandomForest of 25 trees, maximum depth 4 and perturbation 0.015 by executing
python3 train_forest.py mnist26 25 4 0

#At this point, the file of the large-spread ensemble with .silva extension should be available at the ../datasets/mnist26/models/lse/validation folder.

#Observe the robustness computed by SILVA using the following command. The output will be redirected in a log file.
./silva/src/silva ../datasets/mnist26/models/lse/validation/lse_part_rand_25_4_0.015_100_0_m2_s4_mnpert0.0075_mxpert0.015.silva ../datasets/mnist26/dataset/test_set_normalized.csv --perturbation l_inf 0.015 --index-of-instance -1 > log_basic_test_silva.txt

#Observe the robustness computed by CARVE using the following command. The output will be redirected in a log file. The result should match the result obtained by using SILVA.
./carve/build/verify  -i ../datasets/mnist26/models/lse/validation/lse_part_rand_25_4_0.015_100_0_m2_s4_mnpert0.0075_mxpert0.015.silva -t ../datasets/mnist26/dataset/test_set_normalized.csv -p inf -k 0.015 -ioi -1 > log_basic_test_carve.txt

