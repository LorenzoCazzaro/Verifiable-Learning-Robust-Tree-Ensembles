import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from silva.src.trainers.classifier_mapper import ClassifierMapper
import argparse

_SAVE_RF_PATH = "rf/"
_RF_FILENAME = "rf_{}_{}_{}_val{}.silva"
_DATASET_FOLDER = "../datasets/{}/dataset/"
_MODELS_FOLDER = "../datasets/{}/models/"

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('n_trees', type=int, default=101)
parser.add_argument('max_depth', type=int, default=6)
parser.add_argument('random_state', type=int, default=0)
parser.add_argument('--criterion', type=str, default="gini")
parser.add_argument('--n_jobs', type=int, default=1)
parser.add_argument('--validation', action="store_true")
args = parser.parse_args()

data = pd.read_csv(_DATASET_FOLDER.format(args.dataset) + ("training_set_normalized.csv" if not args.validation else "training_set_normalized_gridsearch.csv"), delimiter = ",", skiprows = [0], header=None).to_numpy()
Y = data[:, 0].astype(int).astype(str)
X = data[:, 1:].astype(float)

clf = RandomForestClassifier(n_estimators = args.n_trees, max_depth = args.max_depth, criterion = args.criterion, n_jobs = args.n_jobs, random_state=args.random_state)
clf.fit(X, Y)
classifier_mapper = ClassifierMapper()
print(_MODELS_FOLDER.format(args.dataset) + _SAVE_RF_PATH + _RF_FILENAME.format(args.n_trees, args.max_depth, args.random_state, args.validation))
classifier_mapper.create(clf, _MODELS_FOLDER.format(args.dataset) + _SAVE_RF_PATH + _RF_FILENAME.format(args.n_trees, args.max_depth, args.random_state, args.validation))
