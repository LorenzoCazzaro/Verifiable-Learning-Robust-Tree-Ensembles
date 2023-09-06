import subprocess
import argparse
import pandas as pd
import numpy as np
import os

_ANALIZERS = ["SILVA", "CARVE"]

parser = argparse.ArgumentParser()
parser.add_argument('analyzer', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('model_path', type=str)
parser.add_argument('k', type=str)
parser.add_argument('n_times_repeat', type=int)
parser.add_argument('time_limit', type=str, default="120")
parser.add_argument('memory_limit', type=str, default="64")
args = parser.parse_args()

analyzer = args.analyzer
dataset_path = args.dataset_path
model_path = args.model_path
k = args.k
n_times_repeat = args.n_times_repeat
time_limit = args.time_limit
memory_limit = args.memory_limit

test_set = pd.read_csv(dataset_path, skiprows=[0], header=None).drop([0], axis=1)
n_instances = test_set.shape[0]

assert analyzer in _ANALIZERS

os.system("cgcreate -g memory:myGroup")
os.system("echo {}G > /sys/fs/cgroup/memory/myGroup/memory.limit_in_bytes".format(memory_limit))
os.system("cat /sys/fs/cgroup/memory/myGroup/memory.limit_in_bytes")
os.environ["TIMEFORMAT"] = "%3R"
print(os.environ["TIMEFORMAT"])
os.system("echo $TIMEFORMAT")
print("MODEL: ", model_path)

verification_time_total = 0
maximum_memory_consumption = 0
out_of_resources_counter = 0

for i in range(n_instances):
	verification_time_x_instance = []
	maximum_memory_consumption_x_instance = 0
	out_of_resources_x_instance = []

	for repeat_i in range(n_times_repeat):
		verification_instance_out_of_resources = False
		print("Instance ", str(i), " in verification.")
		if analyzer == _ANALIZERS[0]:
			p = subprocess.run(["/usr/bin/time", "-f", "%{} %{}".format("M", "E"), "timeout", "--signal=9", time_limit, "cgexec", "-g", "memory:myGroup", "./silva/src/silva", model_path, dataset_path, "--perturbation", "l_inf", k, "--sample-timeout", str(int(time_limit)+30), "--index-of-instance", str(i)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		else: #CARVE
			p = subprocess.run(["/usr/bin/time", "-f", "%{} %{}".format("M", "E"), "timeout", "--signal=9", time_limit, "cgexec", "-g", "memory:myGroup", "./carve/build/verify", "-i", model_path, "-t", dataset_path, "-p", "inf", "-k", k, "-ioi", str(i)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		output = p.stderr.decode().split("\n")[:-1]
		print(output)

		if len(output) == 1:
			memory, etime  = output[0].split(" ")
			minutes_seconds, centiseconds = etime.split(".")
			minutes, seconds = minutes_seconds.split(":")

		elif len(output) == 2:
			etime = output[1].split(" ")[1]
			memory = output[1].split(" ")[0]
			minutes_seconds, centiseconds = etime.split(".")
			minutes, seconds = minutes_seconds.split(":")
			verification_instance_out_of_resources = True

		verification_time_x_instance.append(int(minutes)*60 + int(seconds) + int(centiseconds)*0.01)
		maximum_memory_consumption_x_instance = max(int(memory)/(10**6), maximum_memory_consumption_x_instance)
		out_of_resources_x_instance.append(verification_instance_out_of_resources)

	print("Ver times: ", verification_time_x_instance)
	print("Time to be summed: ", sum(verification_time_x_instance)/len(verification_time_x_instance))
	print("Out of resources: ", out_of_resources_x_instance)
	verification_time_total += sum(verification_time_x_instance)/len(verification_time_x_instance)
	print("New total ver time: ", verification_time_total)
	maximum_memory_consumption = max(maximum_memory_consumption_x_instance, maximum_memory_consumption)
	print("Maximum memory consumption: ", maximum_memory_consumption)
	out_of_resources_counter += 1 if sum(out_of_resources_x_instance) == len(out_of_resources_x_instance) else 0
	print("Out of resources counter: ", out_of_resources_counter)

print("Total ver time: ", verification_time_total)

result = pd.DataFrame(data={"time": [verification_time_total], "memory": [maximum_memory_consumption], "out_of_resources": [out_of_resources_counter]})
result.to_csv("./log_total_scalability_{}_{}_{}_TimeLimit{}_MemoryLimit{}.csv".format(analyzer, model_path.split("/")[-1].split(".")[0], k, time_limit, memory_limit), index=False)