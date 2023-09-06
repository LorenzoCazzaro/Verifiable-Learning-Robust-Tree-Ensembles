import subprocess
import argparse
import pandas as pd
import numpy as np

_ANALIZERS = ["SILVA", "CARVE"]

parser = argparse.ArgumentParser()
parser.add_argument('analyzer', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('model_path', type=str)
parser.add_argument('k', type=str)
parser.add_argument('n_times_repeat', type=int)
parser.add_argument('--time_limit', type=int, default=600)
args = parser.parse_args()

analyzer = args.analyzer
dataset_path = args.dataset_path
model_path = args.model_path
k = args.k
n_times_repeat = args.n_times_repeat

test_set = pd.read_csv(dataset_path, skiprows=[0], header=None).drop([0], axis=1)
n_instances = test_set.shape[0]
verification_times = np.array(([[0.0]*n_times_repeat])*n_instances)
memory_consumptions = np.array(([[0.0]*n_times_repeat])*n_instances)
out_of_memory = np.array(([[False]*n_times_repeat])*n_instances)

assert analyzer in _ANALIZERS

print("MODEL: ", model_path)

for i in range(n_instances):
	for repeat_i in range(n_times_repeat):
		print("Instance ", str(i), " in verification.")
		if analyzer == _ANALIZERS[0]: #SILVA
			p = subprocess.run(["/usr/bin/time", "-f", "%{} %{}".format("M", "E"), "./silva/src/silva", model_path, dataset_path, "--perturbation", "l_inf", k, "--sample-timeout", str(args.time_limit), "--index-of-instance", str(i)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		else: #CARVE
			p = subprocess.run(["/usr/bin/time", "-f", "%{} %{}".format("M", "E"), "./carve/build/verify", "-i", model_path, "-t", dataset_path, "-p", "inf", "-k", k, "-ioi", str(i)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		
		output = p.stderr.decode().split("\n")[:-1]
		print(output)

		if len(output) == 1:
			time = output[0].split(" ")[1]
			memory = output[0].split(" ")[0]
			minutes_seconds, centiseconds = time.split(".")
			minutes, seconds = minutes_seconds.split(":")
			verification_times[i][repeat_i] = int(minutes)*60 + int(seconds) + int(centiseconds)*0.01
			print("Ver time: ", verification_times[i][repeat_i])
			memory_consumptions[i][repeat_i] = int(memory)/(10**6)

		elif len(output) == 2:
			time = output[1].split(" ")[1]
			memory = output[1].split(" ")[0]
			minutes_seconds, centiseconds = time.split(".")
			minutes, seconds = minutes_seconds.split(":")
			verification_times[i][repeat_i] = int(minutes)*60 + int(seconds) + int(centiseconds)*0.01
			memory_consumptions[i][repeat_i] = int(memory)/(10**6)
			out_of_memory[i][repeat_i] = True

verification_times = np.mean(verification_times, axis=1)
memory_consumptions = np.mean(memory_consumptions, axis=1)
out_of_memory = np.apply_along_axis(lambda x: sum(x) == n_times_repeat, 1, out_of_memory)

result = pd.DataFrame(data={"times": verification_times, "memory": memory_consumptions, "out_of_memory": out_of_memory})
result.to_csv("./log_scalability_{}_{}_{}.csv".format(analyzer, model_path.split("/")[-1].split(".")[0], k), index=False)

