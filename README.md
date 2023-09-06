# Verifiable Learning for Robust Tree Ensembles

This repository contains the implementation of LSE e CARVE proposed by Calzavara et. al. in their research paper titled [Verifiable Learning for Robust Tree Ensembles](https://arxiv.org/abs/2305.03626) accepted at the ACM Conference on Computer and Communications Security 2023 (CCS 2023). This repository also contains the code and the scripts to reproduce the experiments described in the paper.

## System configuration

In the paper you may find some details of the system in which we run the experiments. Here we report some details about the software. Our system used:
<ul>
	<li> python 3.8.15 </li>
	<li> scikit-learn 1.0.2 </li>
	<li> g++ 9.4.0 </li>
	<li> make 4.2.1 </li>
</ul>

## Obtain the datasets

You can produce the training sets and test sets used in our experimental evaluation by executing the following scripts in the <em>src/utils</em> folder:

`python3 split_mnist.py`

`python3 split_fmnist.py 0 3`

`python3 split_rewema.py`

`python3 split_webspam.py`

The command for splitting the REWEMA dataset requires uncompressing the zip folder in the rewema folder in the datasets folder.

For the webspam dataset, download the normalized unigram version of the dataset, put the raw dataset in the webspam/dataset folder and execute `python3 prepare_webspam.py` inside the utils folder before executing the command for splitting the dataset.

## Compile the tools

### SILVA

See the README.md in the <em>src/silva</em> folder.

### CARVE

See the README.md in the <em>src/carve</em> folder.

## Use the tools

### SILVA

See the README.md in the <em>src/silva</em> folder.

### CARVE

See the README.md in the <em>src/carve</em> folder.

### LSE

Run our LSE tool in the <em>src</em> folder to train large-spread ensembles. It requires:

<ul>
	<li> the perturbation k </li>
	<li> the name of the dataset</li>
	<li> the number of trees </li>
	<li> the maximum depth </li>
	<li> the factor multiplied to the initial number of trees (e.g., 2) </li>
	<li> the maximum number of rounds </li>
	<li> the number of subsets of features for the hierarchical training </li>
	<li> the n_jobs for training the initial random forest </li>
	<li> 1 or 0 for saving the resulting large spread or not</li>
	<li> the minimum perturbation to apply to the thresholds to enforce the large-spread condition</li>
	<li> the maximum perturbation to apply to the thresholds to enforce the large-spread condition</li>
	<li> if training the model on the 80% of the training set (20% is the validation set) or the entire training set</li>
</ul>

Example:

`python3 LSE.py 0.015 --dataset mnist26 --trees 101 --depth 6 --rounds 1000 --random_seed 0 --m 2 --s 4 --jobs 6 --dump_large_spread 1 --min_pert 0.0075 --max_pert 0.015 --validation`

## Train the models

All the models are in the <em>.silva</em> format. See the repository of SILVA for more information.

### Train forests

Execute the bash script <em>train_forests.sh</em> in the <em>src</em> folder after having generated the datasets. See the <em>train_forests.sh</em> script for examples about how to train random forests.

### Train large-spread ensembles
Use the python script <em>train_lses.py</em> in the <em>src</em> folder, that requires:
<ul>
	<li> list of dataset separated by "/". </li>
	<li> Number of trees of the large spread ensemble</li>
	<li> Depth of the large spread ensemble </li>
	<li> list of k separated by "-" </li>
	<li> list of number of rounds separated by "-" </li>
	<li> random state </li>
	<li> factor that multiplies the initial number of trees </li>
	<li> list of number of subsets considered in hierarchical training, separated by "-" </li>
</ul>

To reproduce our experiment, run in the <em>src</em> folder:

`python3 train_lses.py mnist26/rewema/fashion_mnist0-3 25 4 0.005-0.010-0.015 0 100-500 2-4-6 1-2-3-4-5-6 0.005-0.010-0.015 0.075-0.015-0.0225 --validation`

`python3 train_lses.py mnist26/rewema/fashion_mnist0-3 25 4 0.005-0.010-0.015 0 100-500 2-4-6 1-2-3-4-5-6 0.0025-0.005-0.0075 0.005-0.010-0.015 --validation`

`python3 train_lses.py mnist26/rewema/fashion_mnist0-3 101 6 0.005-0.010-0.015 0 100-500 2-4-6 1-2-3-4-5-6 0.005-0.010-0.015 0.075-0.015-0.0225 --validation`

`python3 train_lses.py mnist26/rewema/fashion_mnist0-3 101 6 0.005-0.010-0.015 0 100-500 2-4-6 1-2-3-4-5-6 0.0025-0.005-0.0075 0.005-0.010-0.015 --validation`

`python3 train_lses.py mnist26/rewema/fashion_mnist0-3 25 4 0.0002-0.0004-0.0006 0 100-500 2-4-6 1-2-3-4-5-6 0.0002-0.0004-0.0006 0.0003-0.0006-0.0009 --validation`

`python3 train_lses.py mnist26/rewema/fashion_mnist0-3 25 4 0.0002-0.0004-0.0006 0 100-500 2-4-6 1-2-3-4-5-6 0.0001-0.0002-0.0003 0.0002-0.0004-0.0006 --validation`

`python3 train_lses.py mnist26/rewema/fashion_mnist0-3 101 6 0.0002-0.0004-0.0006 0 100-500 2-4-6 1-2-3-4-5-6 0.0002-0.0004-0.0006 0.0003-0.0006-0.0009 --validation`

`python3 train_lses.py mnist26/rewema/fashion_mnist0-3 101 6 0.0002-0.0004-0.0006 0 100-500 2-4-6 1-2-3-4-5-6 0.0001-0.0002-0.0003 0.0002-0.0004-0.0006 --validation`

**Warning**: the training of the large-spread ensembles with the highest perturbation and no hierarchical training (1 subset of features) may require some time.

## Generate experimental results

Reproduce the results of tables and figures only after training all the models (see the sections above).

### Table 3

To generate a row of table 3, e.g., models with 101 tree, maximum depth 6 and perturbation 0.015, we run the following command to retrieve the accuracy provided by SILVA:

`./silva/src/silva ../datasets/mnist26/models/rf/rf_101_6_0_valFalse.silva ../datasets/mnist26/dataset/test_set_normalized.csv --perturbation l_inf 0.015`

Then, we run the verifier for each istance of the test set with a timeout of 1 second in order to see if it provides a result within the time limit and, if it gives a result, if the model is robust or not on the instance. For example, the following command is used for the instance of index 0:

`timeout --signal=9 1 ./silva/src/silva ../datasets/mnist26/models/rf/rf_101_6_0_valFalse.silva ../datasets/mnist26/dataset/test_set_normalized.csv --perturbation l_inf 0.015 --index-of-instance 0`

Do the same for the corresponding large-spread ensemble by changing the model path.

### Figure 3

Run <em>test_analyzer_scalability.py</em> in the <em>src</em> folder. It requires:
<ul>
	<li> the analyzer identifier (SILVA or CARVE) </li>
	<li> the path of the dataset </li>
	<li> the path of the large-spread ensemble <li>
	<li> the perturbation k </li>
	<li> the number of times to repeat the test </li>
</ul>

Run the following commands to run our experiment:

`python3 test_analyzer_scalability.py SILVA ../datasets/mnist26/dataset/test_set_normalized.csv ../datasets/mnist26/models/rf/rf_101_6_0_valFalse.silva 0.015 10`

`python3 test_analyzer_scalability.py CARVE ../datasets/mnist26/dataset/test_set_normalized.csv ../datasets/mnist26/models/lse/validation/lse_part_rand_101_6_0.015_500_0_m2_s4_mnpert0.0075_mxpert0.015.silva 0.015 10`

The two commands will produce two logs <em>log_scalability_SILVA_rf_101_6_0_0.015.csv</em> and <em>log_scalability_CARVE_lse_part_rand_101_6_0_0.015.csv</em>. These logs contains the verification time for each instance of the test set, the maximum consumption of memory for each instance and if the verifier went out of resources during the verification on a instance. Use the following commands to generate the two plots:

`python3 plot_times_comparison.py ./log_scalability_SILVA_rf_101_6_0_0.015.csv ./log_scalability_CARVE_lse_part_rand_101_6_0_0.015.csv 1-30-60-120-180-240-300-360-420-480-540-600 plot_times_lse_part_rand_101_6_0_0.015_1-30-60-120-180-240-300-360-420-480-540-600`

`python3 plot_memories_comparison.py ./log_scalability_SILVA_rf_101_6_0_0.015.csv ./log_scalability_CARVE_lse_part_rand_101_6_0_0.015.csv 4-8-16-32-64-100 plot_memories_lse_part_rand_101_6_0_0.015_4-8-16-32-64-100`

### Table 4

Use the following commands to obtain a row of the table:

`./verify -i ../../../datasets/<dataset_name>/models/validation/<lse_name.silva> -t ../../../datasets/<dataset_name>/dataset/<test_set_csv_name.csv> -p inf -k <k> -ioi -1`

`./verify -i ../../../datasets/<dataset_name>/models/validation/<lse_name.silva> -t ../../../datasets/<dataset_name>/dataset/<test_set_csv_name.csv> -p 1 -k <k> -ioi -1`

`./verify -i ../../../datasets/<dataset_name>/models/validation/<lse_name.silva> -t ../../../datasets/<dataset_name>/dataset/<test_set_csv_name.csv> -p 2 -k <k> -ioi -1`

### Table 5

Use the script <em>test_total_efficiency.py</em> in the <em>src</em> folder. It requires:
<ul>
	<li> The name of the analyser (SILVA or CARVE). </li>
	<li> The path of the test set w.r.t. the <em>src</em> folder </li>
	<li> The path of the test model w.r.t. the <em>src</em> folder </li>
	<li> k </li>
	<li> the time limit </li>
	<li> the memory limit </li>
</ul>

To reproduce our experiment, run:

`python3 test_total_efficiency.py SILVA ../datasets/mnist26/dataset/test_set_normalized.csv ../datasets/mnist26/models/rf/rf_101_6_0.silva 0.015 1 60 64`

`python3 test_total_efficiency.py CARVE ../datasets/mnist26/dataset/test_set_normalized.csv ../datasets/mnist26/models/lse/validation/lse_part_rand_101_6_0.015_500_0_m2_s4_mnpert0.0075_mxpert0.015.silva 0.015 1 60 64`

**Warning**: this script uses the cgroup linux feature. Make sure to have it installed in your linux system and run it using the super user account.

The resulting csv file "./log_total_scalability_..." in the <em>src</em> folder contains the results of SILVA and CARVE in the table.

### Figure 4-5

To plot the sub-figures, use the following commands (change the dataset names and the perturbations accordingly):

`python3 plot_performance_by_ntrees.py mnist26 25-51-75-101 6 0.015-0.01-0.005 0 500 6 6 0.015-0.01-0.005 0.0225-0.015-0.0075 performance_trees_rewema`

`python3 plot_performance_by_depth.py rewema 101 3-4-5-6 0.015-0.01-0.005 0 500 6 6 0.015-0.01-0.005 0.0225-0.015-0.0075 performance_depth_rewema`

### Table 6

Use the following command to verify each LSE previously trained following the section "Train large-spread ensembles" and generate a report for each model containing also the accuracy and robustness (change the dataset and the perturbation accordingly):

`python3 verify_only_gridsearch.py mnist26 101 6 0.01 0.5-1 1-1.5`

Then, use the following command to display the best model for each dataset and model size, along with the best hyper-parameters:

`python3 find_best_model.py mnist26 101 6 0.015 0 100-500 2-4-6 1-2-3-4-5-6 0.5-1 1-1.5`
