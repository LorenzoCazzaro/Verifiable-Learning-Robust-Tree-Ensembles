import argparse
import itertools
import logging
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from silva.src.trainers.classifier_mapper import ClassifierMapper
import time
from partitioned_randomforest import PartitionRandomForest

_SAVE_LSE_PATH = "lse/"
_SAVE_LSE_VALIDATION_PATH = "lse/validation/"
_LARGE_SPREAD_FILENAME = "lse_part_{}_{}_{}_{}_{}_{}_m{}_s{}_mnpert{}_mxpert{}.silva"
_DATASET_FOLDER = "../datasets/{}/dataset/"
_MODELS_FOLDER = "../datasets/{}/models/"
_PARTITION_POLICY = "rand"

'''
    FUNCTION: extract the thresholds for each feature from a tree
    INPUT: a DecisionTreeClassifier of scikit-learn
    OUTPUT: a dictionary containing the feature numbers as keys and list of thresholds in the tree as values
'''
def extract_thresholds(t):
    children_left = t.tree_.children_left
    children_right = t.tree_.children_right
    feature = t.tree_.feature
    threshold = t.tree_.threshold
    stack = [0]
    res = {}
    while len(stack) > 0:
        node_id = stack.pop()
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            f = feature[node_id]
            v = threshold[node_id]
            if f not in res.keys():
                res[f] = [v]
            else:
                res[f].append(v)
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
    return res

'''
    FUNCTION: given the two dict of thresholds of two distinct trees, return which thresholds of the two trees for the same feature are distant less or equal than 2k.
    INPUT: the two DecisionTreeClassifier of scikit-learn and the distance k
    OUTPUT: a dictionary containing the feature numbers as keys and list of tuples containing the id of trees and thresholds as output
'''
def intersects(t1,t2,k):
    l1 = extract_thresholds(t1)
    l2 = extract_thresholds(t2)
    common_features = set(l1.keys()).intersection(set(l2.keys()))
    overlaps = {}
    for f in common_features:
        for (v1,v2) in itertools.product(l1[f],l2[f]):
            if abs(v1-v2) <= 2 * k: #keep track only if the distance is less or equal than 2k
                if f not in overlaps.keys():
                    overlaps[f] = []
                overlaps[f].append((t1.tree_id,v1,t2.tree_id,v2))
    return overlaps

#NOTE: we will use the term "overlapping thresholds" for reporting thresholds whose distance is less than 2*k, with k real number

'''
    FUNCTION: given a list containing a forest of DecisionTreeClassifiers of scikit-learn, return which thresholds and trees of the forest are overlapping for some features.
    INPUT: a list of DecisionTreeEnsembles and a perturbation k
    OUTPUT: a dictionary containing the feature numbers as keys and list of tuples containing the id of trees and thresholds as output 
'''
def find_overlaps(dte,k):
    global_overlaps = {}
    for (i,t1) in enumerate(dte):
        for t2 in dte[i+1:]:
            overlaps = intersects(t1,t2,k)
            for f in overlaps.keys():
                if f not in global_overlaps.keys():
                    global_overlaps[f] = set([])
                for el in overlaps[f]:
                    global_overlaps[f].add(el)
    return global_overlaps

'''
    FUNCTION: perturb the thresholds of the nodes containing the feature f of the tree t by k
    INPUT: a DecisionTreeClassifier t, a feature (number) f, a threshold v and a perturbation k
    OUTPUT: None, the perturbation of the thresholds is a side-effect of the function on the tree
'''
def fix_tree(t,f,v,k):
    children_left = t.tree_.children_left
    children_right = t.tree_.children_right
    feature = t.tree_.feature
    threshold = t.tree_.threshold
    stack = [0]
    while len(stack) > 0: #visit all the nodes of the tree to find the nodes containing thresholds for the feature f
        node_id = stack.pop()
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node: #if it is a node containing a feature and a value (splitting node)
            feat = feature[node_id]
            val = threshold[node_id]
            if feat == f and val == v and 0 <= threshold[node_id] + k <= 1: #perturb only if you do not exceed the 0-1 bounds by applying the perturbation
                threshold[node_id] = threshold[node_id] + k
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
    return

'''
    FUNCTION: given a list of trees and information about the features for which there are trees with overlapping thresholds, perturb the thresholds to enforce the large-spread condition
    INPUT: a list of DecisionTreeClassifiers lse, a dict lse containing the features as key and a list of tuples containing ids of trees and overlapping thresholds as values, the minimum perturbation min_pert and the maximum perturbation max_pert to apply.
    OUTPUT: None, the perturbation of the thresholds is a side-effect of the function on the list of trees
'''
def fix_forest(lse, ovs, min_pert, max_pert):
    for (f,s) in ovs.items(): #feature f and list of tuples (id_tree_1, thresholds_for_feature_f_of_tree_1, id_tree_2, thresholds_for_feature_f_of_tree_2)
        for d in s:
            t1 = d[0] #id_tree_1
            v1 = d[1] #thresholds of id_tree_1 of feature f
            t2 = d[2] #id_tree_2
            v2 = d[3] ##thresholds of id_tree_2 of feature f
            minv = v1 if v1 < v2 else v2
            noise = noise = random.uniform(min_pert,max_pert) #perturb thresholds to spread them and enforce the large-spread condition
            if (minv == v1):
                for t in lse:
                    if t.tree_id == t1:
                        fix_tree(t,f,v1,-noise)
                    elif t.tree_id == t2:
                        fix_tree(t,f,v2,noise)
            else:
                for t in lse:
                    if t.tree_id == t1:
                        fix_tree(t,f,v1,noise)
                    elif t.tree_id == t2:
                        fix_tree(t,f,v2,-noise)
    return

'''
    FUNCTION: produce a large-spread ensemble from the list of trees dte. Have a look at the paper for the formalization of the algorithm.
    INPUT: the two DecisionTreeClassifiers in dte, the perturbation k, the number of trees nt to obtain from the pruning, the min perturbation and max perturbation to apply for perturbing the thresholds.
    OUTPUT: the pruned trees of the large-spread ensemble in a list
'''
def smart_pruning(dte, k, nt, rounds, min_pert, max_pert):
    trees = dte.estimators_
    lse = [trees[random.randint(0,len(trees)-1)]]   #SampleTree

    while (len(trees) > 0 and len(lse) < nt):
        overlaps = []

        #GetBestTree starts here
        for t in trees: 
            tmp = lse + [t]
            ov = find_overlaps(tmp,k)
            ov = {k: v for k, v in ov.items() if v != []}
            overlaps.append(ov)
        best_tree = overlaps.index(min(overlaps, key=lambda x: (len(x.keys()))))
        #GetBestTree concludes here -> select best tree for minimum number of features with overlaps

        #print("Found best tree: {}".format(best_tree))
        lse.append(trees[best_tree]) #T* U {t}
        print("Current forest size: {}".format(len(lse)))
        trees.pop(best_tree) # T / {t}
        print("Remaining trees: {}".format(len(trees)))

        #FixForest starts here
        counter = 0
        ov_x_iters = find_overlaps(lse,k)
        pert_lse = deepcopy(lse)
        #len(ov_x_iters.keys()) > 0 ----> NO large-spread
        while len(ov_x_iters.keys()) > 0 and counter < rounds:
            fix_forest(pert_lse, ov_x_iters, min_pert, max_pert)
            counter = counter + 1
            ov_x_iters = find_overlaps(pert_lse,k)
        if counter == rounds:
            print("Cannot fix overlaps! Select another tree.")
            lse.pop()
        else:
            lse = pert_lse
            print("Fixed overlaps!\n")
    if len(lse) == nt:
        print("Successfully trained large-spread ensemble of {} trees!".format(nt))
    else:
        print("Training failed: stopped at {} trees".format(len(lse)))
    return lse

'''
    FUNCTION: given the a feature list (set of feature), split it in a partition of s subsets.
    INPUT: the feature list and number of sets s.
    OUTPUT: a list of lists of features (feature sets).
'''
def partition_features_random(feature_list, s):
    n_feat_x_set = len(feature_list)//s
    remainder = len(feature_list)%s
    feats_sets = []

    for _ in range(0, s):
        feat_set = []
        n_feat_to_extract = n_feat_x_set
        if remainder > 0:
            remainder -= 1
            n_feat_to_extract += 1
        
        for _ in range(0, n_feat_to_extract):
            index = random.randint(0, len(feature_list)-1)
            feat_set.append(feature_list[index])
            feature_list.pop(index)
        feats_sets.append(feat_set)

    return feats_sets


#Script starts here
parser = argparse.ArgumentParser()
parser.add_argument('k', type=float)
parser.add_argument('--dataset', default='mnist_784')
parser.add_argument('--trees', type=int, default=10)
parser.add_argument('--depth', type=int, default=7) #N.B.: maximum depth
parser.add_argument('--m', type=int, default=2)
parser.add_argument('--rounds', type=int, default=101) #N.B.: maximum number of rounds
parser.add_argument('--s', type=int, default=1)
parser.add_argument('--jobs', type=int, default=4)
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--dump_large_spread', type=int, default=False)
parser.add_argument('--min_pert', type=float, default=0)
parser.add_argument('--max_pert', type=float, default=1)
parser.add_argument('--validation', action="store_true")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
random.seed(args.random_seed)

#load the dataset given its name (folder name)
folder_name = args.dataset
train = pd.read_csv(_DATASET_FOLDER.format(folder_name) + ("training_set_normalized_gridsearch.csv" if args.validation else "training_set_normalized.csv"), skiprows=[0], header=None)
X_train = train.iloc[:, 1:]
X_train.columns = range(X_train.columns.size)
y_train = train.iloc[:, 0].astype(int).astype(str)

print("Random seed: {}".format(args.random_seed))
print("Loaded dataset: {}".format(args.dataset))
assert (len(np.unique(y_train)) == 2), "only binary classification tasks are supported"

atk = args.k
print("k: ", atk, ", min_pert: ", args.min_pert, ", max_pert: ", args.max_pert)
feats_sets = partition_features_random(X_train.columns.tolist(), args.s) #partition the set of features for performing the hierarchical training

print("Training large spread ensemble: {} trees of maximum depth {}".format(args.trees, args.depth))

large_spread_estimators = []

n_subforests_to_train = args.s
n_remaining_trees_to_train = args.trees

for feats_set in feats_sets: #for each feature subset train a large-spread ensemble
    #keep track of the number of trees of the current sub-forest to train to reach the desired total size of the forest.
    n_trees_subforest = n_remaining_trees_to_train//n_subforests_to_train
    if n_remaining_trees_to_train%n_subforests_to_train > 0:
        n_trees_subforest += 1

    print("Training traditional subforest: {} trees of maximum depth {}".format(args.m * n_trees_subforest, args.depth))
    clf = RandomForestClassifier(random_state=args.random_seed, n_estimators=args.m * n_trees_subforest, max_depth=args.depth, n_jobs=args.jobs)
    clf.fit(X_train.iloc[:, feats_set], y_train)

    for (i,t) in enumerate(clf.estimators_):
        t.tree_id = i

    print("Forest size before pruning: {}".format(len(clf.estimators_)))

    st_pruning = time.time()
    pruned = smart_pruning(clf,atk,n_trees_subforest,args.rounds, args.min_pert, args.max_pert)
    et_pruning = time.time()
    print("Time required for pruning: ", et_pruning - st_pruning, ' seconds')
    print("Forest size after pruning: {}".format(len(pruned)))
    n_remaining_trees_to_train = n_remaining_trees_to_train - len(pruned)
    n_subforests_to_train -= 1
    assert (len(find_overlaps(pruned, atk).keys()) == 0), "the large spread ensemble is not large spread!"

    large_spread_estimators += [(pruned, feats_set)]
    
large_spread = PartitionRandomForest(large_spread_estimators, X_train.shape[1], 2, ['0', '1'])

if args.dump_large_spread:
    classifier_mapper = ClassifierMapper()
    classifier_mapper.create(large_spread, (_MODELS_FOLDER.format(folder_name) + (_SAVE_LSE_VALIDATION_PATH if args.validation else _SAVE_LSE_PATH)) + _LARGE_SPREAD_FILENAME.format(_PARTITION_POLICY, large_spread.n_trees_, args.depth, atk, args.rounds, args.random_seed, args.m, args.s, args.min_pert, args.max_pert))