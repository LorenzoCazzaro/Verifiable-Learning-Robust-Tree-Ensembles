from sklearn import ensemble

class Forest:
    def __init__(self, n_trees, max_depth, criterion = 'gini', random_state = 0, n_jobs=1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.n_jobs = n_jobs

    def train(self, x, y):
        clf = ensemble.RandomForestClassifier(
            n_estimators = self.n_trees,
            max_depth = self.max_depth,
            criterion = self.criterion,
            random_state = self.random_state,
            n_jobs = self.n_jobs
        )
        clf.fit(x, y)
        return clf
