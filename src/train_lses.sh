#train fashion_mnist

python3 train_forest.py fashion_mnist0-3 25 4 0

python3 train_forest.py fashion_mnist0-3 101 6 0

/usr/bin/time python3 LSE.py 0.005 --dataset fashion_mnist0-3 --trees 25 --depth 4 --rounds 100 --random_seed 0 --m 4 --s 1 --dump_large_spread 1 --min_pert 0.0025 --max_pert 0.005 --validation > ../datasets/fashion_mnist0-3/models/lse/validation/log_lse_fm_25_4_0.005.txt

/usr/bin/time python3 LSE.py 0.01 --dataset fashion_mnist0-3 --trees 25 --depth 4 --rounds 100 --random_seed 0 --m 4 --s 1 --dump_large_spread 1 --min_pert 0.005 --max_pert 0.01 --validation > ../datasets/fashion_mnist0-3/models/lse/validation/log_lse_fm_25_4_0.01.txt

/usr/bin/time python3 LSE.py 0.015 --dataset fashion_mnist0-3 --trees 25 --depth 4 --rounds 100 --random_seed 0 --m 6 --s 1 --dump_large_spread 1 --min_pert 0.015 --max_pert 0.0225 --validation > ../datasets/fashion_mnist0-3/models/lse/validation/log_lse_fm_25_4_0.015.txt

/usr/bin/time python3 LSE.py 0.005 --dataset fashion_mnist0-3 --trees 101 --depth 6 --rounds 500 --random_seed 0 --m 2 --s 1 --dump_large_spread 1 --min_pert 0.0025 --max_pert 0.005 --validation > ../datasets/fashion_mnist0-3/models/lse/validation/log_lse_fm_101_6_0.005.txt

/usr/bin/time python3 LSE.py 0.01 --dataset fashion_mnist0-3 --trees 101 --depth 6 --rounds 500 --random_seed 0 --m 6 --s 1 --dump_large_spread 1 --min_pert 0.005 --max_pert 0.01 --validation > ../datasets/fashion_mnist0-3/models/lse/validation/log_lse_fm_101_6_0.01.txt

/usr/bin/time python3 LSE.py 0.015 --dataset fashion_mnist0-3 --trees 101 --depth 6 --rounds 100 --random_seed 0 --m 4 --s 5 --dump_large_spread 1 --min_pert 0.0075 --max_pert 0.015 --validation > ../datasets/fashion_mnist0-3/models/lse/validation/log_lse_fm_101_6_0.015.txt



#train MNIST

python3 train_forest.py mnist26 25 4 0

python3 train_forest.py mnist26 101 6 0

/usr/bin/time python3 LSE.py 0.005 --dataset mnist26 --trees 25 --depth 4 --rounds 100 --random_seed 0 --m 6 --s 1 --dump_large_spread 1 --min_pert 0.0025 --max_pert 0.005 --validation > ../datasets/mnist26/models/lse/validation/log_lse_mn_25_4_0.005.txt

/usr/bin/time python3 LSE.py 0.01 --dataset mnist26 --trees 25 --depth 4 --rounds 100 --random_seed 0 --m 2 --s 1 --dump_large_spread 1 --min_pert 0.005 --max_pert 0.01 --validation > ../datasets/mnist26/models/lse/validation/log_lse_mn_25_4_0.01.txt

/usr/bin/time python3 LSE.py 0.015 --dataset mnist26 --trees 25 --depth 4 --rounds 100 --random_seed 0 --m 2 --s 1 --dump_large_spread 1 --min_pert 0.0075 --max_pert 0.015 --validation > ../datasets/mnist26/models/lse/validation/log_lse_mn_25_4_0.015.txt

/usr/bin/time python3 LSE.py 0.005 --dataset mnist26 --trees 101 --depth 6 --rounds 100 --random_seed 0 --m 6 --s 1 --dump_large_spread 1 --min_pert 0.0025 --max_pert 0.005 --validation > ../datasets/mnist26/models/lse/validation/log_lse_mn_101_6_0.005.txt

/usr/bin/time python3 LSE.py 0.01 --dataset mnist26 --trees 101 --depth 6 --rounds 500 --random_seed 0 --m 6 --s 1 --dump_large_spread 1 --min_pert 0.005 --max_pert 0.01 --validation > ../datasets/mnist26/models/lse/validation/log_lse_mn_101_6_0.01.txt

/usr/bin/time python3 LSE.py 0.015 --dataset mnist26 --trees 101 --depth 6 --rounds 500 --random_seed 0 --m 2 --s 4 --dump_large_spread 1 --min_pert 0.0075 --max_pert 0.015 --validation > ../datasets/mnist26/models/lse/validation/log_lse_mn_101_6_0.015.txt


#train REWEMA

python3 train_forest.py rewema 25 4 0

python3 train_forest.py rewema 101 6 0

/usr/bin/time python3 LSE.py 0.005 --dataset rewema --trees 25 --depth 4 --rounds 100 --random_seed 0 --m 2 --s 1 --dump_large_spread 1 --min_pert 0.0025 --max_pert 0.005 --validation > ../datasets/rewema/models/lse/validation/log_lse_rw_25_4_0.005.txt

/usr/bin/time python3 LSE.py 0.01 --dataset rewema --trees 25 --depth 4 --rounds 100 --random_seed 0 --m 6 --s 1 --dump_large_spread 1 --min_pert 0.01 --max_pert 0.015 --validation > ../datasets/rewema/models/lse/validation/log_lse_rw_25_4_0.01.txt

/usr/bin/time python3 LSE.py 0.015 --dataset rewema --trees 25 --depth 4 --rounds 100 --random_seed 0 --m 6 --s 1 --dump_large_spread 1 --min_pert 0.015 --max_pert 0.0225 --validation > ../datasets/rewema/models/lse/validation/log_lse_rw_25_4_0.015.txt

/usr/bin/time python3 LSE.py 0.005 --dataset rewema --trees 101 --depth 6 --rounds 500 --random_seed 0 --m 2 --s 1 --dump_large_spread 1 --min_pert 0.0025 --max_pert 0.005 --validation > ../datasets/rewema/models/lse/validation/log_lse_rw_101_6_0.005.txt

/usr/bin/time python3 LSE.py 0.01 --dataset rewema --trees 101 --depth 6 --rounds 500 --random_seed 0 --m 4 --s 1 --dump_large_spread 1 --min_pert 0.005 --max_pert 0.01 --validation > ../datasets/rewema/models/lse/validation/log_lse_rw_101_6_0.01.txt

/usr/bin/time python3 LSE.py 0.015 --dataset rewema --trees 101 --depth 6 --rounds 500 --random_seed 0 --m 4 --s 2 --dump_large_spread 1 --min_pert 0.015 --max_pert 0.0225 --validation > ../datasets/rewema/models/lse/validation/log_lse_rw_101_6_0.015.txt


#train Webspam

python3 train_forest.py webspam 25 4 0

python3 train_forest.py webspam 101 6 0

/usr/bin/time python3 LSE.py 0.005 --dataset webspam --trees 25 --depth 4 --rounds 100 --random_seed 0 --m 2 --s 1 --dump_large_spread 1 --min_pert 0.005 --max_pert 0.0075 --validation > ../datasets/webspam/models/lse/validation/log_lse_ws_25_4_0.005.txt

/usr/bin/time python3 LSE.py 0.01 --dataset webspam --trees 25 --depth 4 --rounds 100 --random_seed 0 --m 2 --s 1 --dump_large_spread 1 --min_pert 0.005 --max_pert 0.01 --validation > ../datasets/webspam/models/lse/validation/log_lse_ws_25_4_0.01.txt

/usr/bin/time python3 LSE.py 0.015 --dataset webspam --trees 25 --depth 4 --rounds 100 --random_seed 0 --m 2 --s 1 --dump_large_spread 1 --min_pert 0.015 --max_pert 0.0225 --validation > ../datasets/webspam/models/lse/validation/log_lse_ws_25_4_0.015.txt

/usr/bin/time python3 LSE.py 0.005 --dataset webspam --trees 101 --depth 6 --rounds 500 --random_seed 0 --m 4 --s 5 --dump_large_spread 1 --min_pert 0.0025 --max_pert 0.005 --validation > ../datasets/webspam/models/lse/validation/log_lse_ws_101_6_0.005.txt

/usr/bin/time python3 LSE.py 0.01 --dataset webspam --trees 101 --depth 6 --rounds 500 --random_seed 0 --m 4 --s 6 --dump_large_spread 1 --min_pert 0.005 --max_pert 0.01 --validation > ../datasets/webspam/models/lse/validation/log_lse_ws_101_6_0.01.txt

/usr/bin/time python3 LSE.py 0.015 --dataset webspam --trees 101 --depth 6 --rounds 500 --random_seed 0 --m 4 --s 6 --dump_large_spread 1 --min_pert 0.0075 --max_pert 0.015 --validation > ../datasets/webspam/models/lse/validation/log_lse_ws_101_6_0.015.txt