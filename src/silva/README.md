# SILVA

See the instructions at https://github.com/abstract-machine-learning/silva about how to use SILVA.

We added the command line option "--index-of-instance" to allow the user to run the verification on a single instance of the test set. For example, for running SILVA on the instance of the test set of index 2, you can use a command like this:

`./silva/src/silva ../datasets/mnist26/models/rf/rf_99_6_0.silva ../datasets/mnist26/dataset/test_set_normalized.csv --perturbation l_inf 0.015 --sample-timeout 600 --index-of-instance 2`