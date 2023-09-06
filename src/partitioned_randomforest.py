import numpy as np

class PartitionRandomForest():

	def __init__(self, tree_list, n_features, n_classes, classes): #tree_list = list(([trees], [feats index]))
		self.tree_list_ = []
		self.features_x_tree_ = []
		self.n_trees_ = 0

		for trees in tree_list:
			self.tree_list_.append(trees[0])
			self.features_x_tree_.append(trees[1])
			self.n_trees_ += len(trees[0])

		self.n_features_ = n_features
		self.n_classes_ = n_classes
		self.classes_ = classes

	#NOT CURRENTLY USED 
	def predict(self, data):
		predictions = np.empty((data.shape[0], self.n_trees_))
		i = 0
		for trees, features in zip(self.tree_list_, self.features_x_tree_):
			for tree in trees:
				res = tree.predict(data.iloc[:, features])
				predictions[:, i] = res
				i += 1
		#assume labels [0, 1]
		return np.apply_along_axis(lambda x: '1' if np.sum(x) > len(x)//2 else '0', 1, predictions)