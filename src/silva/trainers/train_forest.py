import sys
import dataset_mapper
import classifier_mapper
import forest
import pandas as pd
import joblib

#Sanity check
if (len(sys.argv)) < 3:
    print("Usage: " + sys.argv[0] + " <dataset> <output> [<n_trees> [<max_depth> [n_jobs] [[<criterion>]]]]")
    sys.exit(-1)

# Reads parameters
dataset_path = sys.argv[1]
output_path = sys.argv[2]
n_trees = 5
if len(sys.argv) > 3:
    n_trees = int(sys.argv[3])

max_depth = 5
if len(sys.argv) > 4:
    max_depth = int(sys.argv[4])

criterion = 'gini'
n_jobs = 1
if len(sys.argv) > 5:
    n_jobs = int(sys.argv[5])

if len(sys.argv) > 6:
    criterion = sys.argv[6]

# Trains model
dataset_mapper = dataset_mapper.DatasetMapper()
x, y = dataset_mapper.read(dataset_path)
#print(x)
#x = pd.DataFrame(x, dtype=str)
#print(x)
#y = pd.DataFrame(y).to_numpy()
#y = y.reshape((y.shape[0],))
trainer = forest.Forest(n_trees, max_depth, criterion, n_jobs=n_jobs)
model = trainer.train(x, y)

classifier_mapper = classifier_mapper.ClassifierMapper()
classifier_mapper.create(model, output_path + ".silva")
joblib.dump(model, output_path + ".joblib")