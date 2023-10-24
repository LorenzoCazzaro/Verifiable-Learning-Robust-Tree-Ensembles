CARVE
------------

### Compiling the code

If you have cloned the repo without `--recursive`, you will need to do

	git submodule update --init --recursive
	
before compiling the code.	 The, compile with

    mkdir build; cd build
    cmake ..
    make

Run

	./verify --help

for command line options.

## Use CARVE

Run CARVE in the <em>src</em> folder to train large-spread ensembles. It requires:

-   *-i* option: the path and filename of the tree-based classifier to verify
-   *-t* option: the path and filename of the test set on which verify the robustness of the modelthe name of the dataset
-   *-p* option: the norm ("inf" or a number)
-   *-k* option: the perturbation norm.
-   *-ioi* option: the index of the instance in the test set on which verify the robustness of the model. -1 for indicating every instance of the test set.

`./verify -i <path_to_model> -t <path_to_test_set> -p <p_value> -k <k_value> -ioi <ioi_value>`

### Example

`./verify -i ../../../datasets/mnist26/models/rf/??? -t ../../../datasets/mnist26/dataset/test_set_normalized.csv -p inf -k 0.01 -ioi -1`

`./verify -i ../../../datasets/mnist26/models/lse/validation/??? -t ../../../datasets/mnist26/dataset/test_set_normalized.csv -p 2 -k 0.01 -ioi -1`

For running the verifier on the instance 2 of the test-set, use the "ioi" option:

`./verify -i ../../../datasets/mnist26/models/rf/??? -t ../../../datasets/mnist26/dataset/test_set_normalized.csv -p inf -k 0.01 -ioi 2` 
    
    