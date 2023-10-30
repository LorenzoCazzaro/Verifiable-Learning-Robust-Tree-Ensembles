import numpy as np
import os

f = open("../../datasets/webspam/dataset/webspam_wc_normalized_unigram.svm", "r")
lines = f.readlines()
n_features = 254
data = np.zeros((len(lines), n_features+1))

for i, line in enumerate(lines):
	splitted_line = line.split(" ")
	data[i][0] = splitted_line[0]
	for values in splitted_line[1:]:
		v_splitted = values.split(":")
		if len(v_splitted) == 2:
			f, v = v_splitted
			data[i][int(f)] = v
		else:
			print("Values not parsed correctly! Line {}".format(i))
			print(v_splitted)

print("Saving data!")
np.savetxt("../../datasets/webspam/dataset/webspam.csv", data, delimiter=",")
os.system("rm webspam_wc_normalized_unigram.svm")
