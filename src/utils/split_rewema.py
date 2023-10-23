import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os

folder_path = "../../datasets/rewema"

#create folders
os.system("mkdir " + folder_path + "/models")
os.system("mkdir " + folder_path + "/models/rf")
os.system("mkdir " + folder_path + "/models/lse")
os.system("mkdir " + folder_path + "/models/lse/validation")

#load dataset
data = pd.read_csv("../../datasets/rewema/dataset/REWEMA.csv").to_numpy()
X = data[:, 2:].astype(float)
y = data[:, 1]
y[y=='B'] = '0'
y[y=='M'] = '1'
y = y.reshape((y.shape[0], 1))

#Scale in [0-1]
X = np.nan_to_num(X)
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

#Save splittings
dataset = np.concatenate((y, X), axis=1)
dataset_df = pd.DataFrame(dataset)
dataset_df.to_csv(folder_path + "/dataset/dataset_normalized.csv", index=False, header=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=7, stratify=y)
train = np.concatenate((y_train, X_train), axis=1)
test = np.concatenate((y_test, X_test), axis=1)
train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)
train_df.to_csv(folder_path + "/dataset/training_set_normalized.csv", index=False, header=False)
test_df.to_csv(folder_path + "/dataset/test_set_normalized.csv", index=False, header=False)
f = open(folder_path + "/dataset/test_set_normalized.csv", "r")
lines = f.read()
f.close()
f = open(folder_path + "/dataset/test_set_normalized.csv", "w")
f.write("# {} {}\n".format(test_df.shape[0], test_df.shape[1]-1))
f.write(lines)
f.close()

#data for gridsearch
X_train_grid, X_val_grid, y_train_grid, y_val_grid = train_test_split(X_train, y_train, test_size = 0.2, random_state=7, stratify=y_train)
train_grid = np.concatenate((y_train_grid, X_train_grid), axis=1)
val_grid = np.concatenate((y_val_grid, X_val_grid), axis=1)
train_grid_df = pd.DataFrame(train_grid)
val_grid_df = pd.DataFrame(val_grid)
train_grid_df.to_csv(folder_path + "/dataset/training_set_normalized_gridsearch.csv", index=False, header=False)
val_grid_df.to_csv(folder_path + "/dataset/validation_set_normalized_gridsearch.csv", index=False, header=False)
f = open(folder_path + "/dataset/validation_set_normalized_gridsearch.csv", "r")
lines = f.read()
f.close()
f = open(folder_path + "/dataset/validation_set_normalized_gridsearch.csv", "w")
f.write("# {} {}\n".format(val_grid_df.shape[0], val_grid_df.shape[1]-1))
f.write(lines)
f.close()