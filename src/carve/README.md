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

### Example

`./verify -i ../../../datasets/<dataset_name>/models/<rf_name.silva> -t ../../../datasets/<dataset_name>/dataset/<dataset_csv_name.csv> -p inf -k 0.01 -ioi -1`

`./verify -i ../../../datasets/<dataset_name>/models/<rf_name.silva> -t ../../../datasets/<dataset_name>/dataset/<dataset_csv_name.csv> -p 2 -k 0.01 -ioi -1`

For running the verifier on the instance 2 of the test-set, use the "ioi" option:

`./verify -i ../../../datasets/<dataset_name>/models/<rf_name.silva> -t ../../../datasets/<dataset_name>/dataset/<dataset_csv_name.csv> -p inf -k 0.01 -ioi 2` 
    
    