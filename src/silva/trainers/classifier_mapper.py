from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from partitioned_randomforest import PartitionRandomForest
#from parse import parse
#import catboost
import csv

class ClassifierMapper:
    def create(self, classifier, file_path):
        if isinstance(classifier, svm.SVC):
            self.createSvm(classifier, file_path)
        elif isinstance(classifier, tree.DecisionTreeClassifier):
            self.createDecisionTree(classifier, file_path)
        elif isinstance(classifier, ensemble.RandomForestClassifier):
            self.createForest(classifier, file_path)
        elif isinstance(classifier, PartitionRandomForest):
            self.createPartitionedForest(classifier, file_path)
        #elif isinstance(classifier, catboost.CatBoostClassifier):
        #    self.createGradientBoostingForest(classifier, file_path)


    def createSvm(self, classifier, file_path):
        with open(file_path, mode = 'w') as destination_file:
            csv_writer = csv.writer(destination_file, delimiter = ' ')
            csv_writer.writerow([
                'ovo',
                classifier.support_vectors_.shape[1],
                classifier.classes_.shape[0]]
            )
            
            if classifier.kernel == 'linear':
                csv_writer.writerow(['linear'])
            elif classifier.kernel == 'poly':
                csv_writer.writerow(['polynomial', classifier.degree, classifier.coef0])
            elif classifier.kernel == 'rbf':
                csv_writer.writerow(['rbf', classifier.gamma])
            
            n_classes = classifier.classes_.shape[0]
            if n_classes > 2:
                for i in range(0, classifier.classes_.shape[0]):
                    csv_writer.writerow([classifier.classes_[i].replace(" ", "-"), classifier.n_support_[i]])
            else:
                csv_writer.writerow([classifier.classes_[1].replace(" ", "-"), classifier.n_support_[1]])
                csv_writer.writerow([classifier.classes_[0].replace(" ", "-"), classifier.n_support_[0]])
            csv_writer.writerow(list(classifier.dual_coef_.flatten()))
            
            for support_vector in classifier.support_vectors_:
                csv_writer.writerow(support_vector)
            csv_writer.writerow(classifier.intercept_)


    def createDecisionTree(self, classifier, file_path):
        file = open(file_path, 'w')
        file.write('classifier-decision-tree {:d} {:d}\n'.format(classifier.n_features_, classifier.n_classes_))
        file.write('{:s}\n'.format(' '.join(classifier.classes_)))
        self.exportTree(classifier, file)
        file.close()


    def createForest(self, classifier, file_path):
        file = open(file_path, 'w')
        file.write('classifier-forest {:d}\n'.format(len(classifier.estimators_)))
        for tree in classifier.estimators_:
            file.write('classifier-decision-tree {:d} {:d}\n'.format(classifier.n_features_, classifier.n_classes_))
            file.write('{:s}\n'.format(' '.join(classifier.classes_)))
            self.exportTree(tree, file)
        file.close()

    #NEW
    def createPartitionedForest(self, classifier, file_path):
        file = open(file_path, 'w')
        file.write('classifier-forest {:d}\n'.format(classifier.n_trees_))
        for trees, features in zip(classifier.tree_list_, classifier.features_x_tree_):
            for tree in trees:
                file.write('classifier-decision-tree {:d} {:d}\n'.format(classifier.n_features_, classifier.n_classes_))
                file.write('{:s}\n'.format(' '.join(classifier.classes_)))
                self.exportPartitionedTree(tree, features, file)
        file.close()


    def createGradientBoostingForest(self, classifier, file_path):
        # classifier.get_all_params()
        # model.get_leaf_values()
        # classifier.get_borders()
        # classifier.get_params()
        # model._object._get_tree_splits(0, None)
        file = open(file_path, 'w')
        file.write('classifier-forest {:d}\n'.format(classifier.tree_count_))
        
        n_classes = len(classifier.classes_)
        n_features = len(classifier.feature_importances_)
        n_trees = classifier.get_params()['num_trees']
        max_depth = classifier.get_params()['max_depth']
        leaves = classifier.get_leaf_values()
        for i in range(0, n_trees):
            file.write('classifier-decision-tree {:d} {:d}\n'.format(n_features, n_classes))
            file.write('{:s}\n'.format(' '.join(classifier.classes_)))
            splits = list(map(lambda x: "{}, bin={}".format(x, x), reversed(classifier._object._get_tree_splits(i, None))))
            leaves_offset = sum(map(lambda x: x * n_classes, classifier.get_tree_leaf_counts()[:i]));
            n_leaves = classifier.get_tree_leaf_counts()[i]
            max_depth = len(splits)

            stack = [(0, 0)]
            while len(stack) > 0:
                depth, bitmask = stack.pop()

                if depth == max_depth:
                    file.write("LEAF_LOGARITHMIC ")
                    for j in range(n_classes):
                        file.write("{:f}".format(leaves[leaves_offset + bitmask * n_classes + j]))
                        if j + 1 < n_classes:
                            file.write(" ")
                    file.write("\n")
                else:
                    file.write("SPLIT {:s} {:s}\n".format(splits[depth][0], splits[depth][1]))
                    stack.append((depth + 1, bitmask | (1 << (max_depth - depth - 1))))
                    stack.append((depth + 1, bitmask))
        file.close()


    def exportTree(self, decision_tree, file):
        tree = decision_tree.tree_
        space_size = tree.n_features
        n_classes = tree.n_classes[0]
        n_nodes = tree.node_count
        
        stack = [0]
        while len(stack) > 0:
            node_id = stack.pop()

            # Leaf
            if tree.children_left[node_id] == tree.children_right[node_id]:
                samples_per_classes = list(map(lambda x: str(int(x)), tree.value[node_id][0]))
                file.write('LEAF {:s}\n'.format(' '.join(samples_per_classes)))

            # Split node
            else:
                feature = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                file.write('SPLIT {:d} {:g}\n'.format(feature, threshold))
                stack.append(tree.children_right[node_id])
                stack.append(tree.children_left[node_id])

    def exportPartitionedTree(self, decision_tree, features, file):
        tree = decision_tree.tree_
        space_size = tree.n_features
        n_classes = tree.n_classes[0]
        n_nodes = tree.node_count
        
        stack = [0]
        while len(stack) > 0:
            node_id = stack.pop()

            # Leaf
            if tree.children_left[node_id] == tree.children_right[node_id]:
                #TODO: changed for excluding ties
                #samples_per_classes = list(map(lambda x: str(int(x)), tree.value[node_id][0]))
                samples_per_classes = list(map(lambda x: int(x), tree.value[node_id][0]))
                if len(samples_per_classes) > 0 and samples_per_classes.count(max(samples_per_classes)) > 1:
                    samples_per_classes[samples_per_classes.index(max(samples_per_classes))] += 1
                samples_per_classes = list(map(lambda x: str(x), samples_per_classes))
                file.write('LEAF {:s}\n'.format(' '.join(samples_per_classes)))

            # Split node
            else:
                feature = features[tree.feature[node_id]]
                threshold = tree.threshold[node_id]
                file.write('SPLIT {:d} {:g}\n'.format(feature, threshold))
                stack.append(tree.children_right[node_id])
                stack.append(tree.children_left[node_id])
