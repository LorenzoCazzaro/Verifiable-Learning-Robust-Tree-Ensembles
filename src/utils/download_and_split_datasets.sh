#!/bin/bash

python3 split_mnist.py

python3 split_fmnist.py 0 3

python3 split_rewema.py

python3 split_webspam.py