import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score


class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] < self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        return self


def gini(x):
    """
    Return the gini impurity score for values in y
    Assume y can be any number of classes >= 2
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    unique_values, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)
    gini_impurity = 1.0 - np.sum(probabilities**2)
    return gini_impurity


def find_best_split(X, y, loss, min_samples_leaf, max_features):
    best = (-1, -1, loss(y))
    num_ = np.random.choice(
        range(X.shape[1]), int(X.shape[1] * max_features), replace=False
    )
    for i in num_:
        random_indices = np.random.randint(0, X.shape[0], size=11)
        candidates = X[random_indices, i]
        for split in candidates:
            yl = y[X[:, i] < split]
            yr = y[X[:, i] >= split]
            if len(yl) < min_samples_leaf or len(yr) < min_samples_leaf:
                continue
            l = (len(yl) * loss(yl) + len(yr) * loss(yr)) / (len(yl) + len(yr))
            if l == 0:
                return i, split
            if l < best[2]:
                best = (i, split, l)
    return best[0], best[1]


class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None, max_features=0.3):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss  # loss function; either np.var for regression or gini for classification
        self.max_features = max_features

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)

    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) <= self.min_samples_leaf or len(set(y)) == 1:
            return self.create_leaf(y)
        else:
            col, split = find_best_split(X, y, self.loss, self.min_samples_leaf, self.max_features)
        if col == -1:
            return self.create_leaf(y)
        left_mask = X[:, col] < split
        right_mask = ~left_mask

        # Recursive call to construct left and right subtrees
        lchild = self.fit_(X[left_mask], y[left_mask])
        rchild = self.fit_(X[right_mask], y[right_mask])

        return DecisionNode(col, split, lchild, rchild)

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        preds = []
        for i in range(len(X_test)):
            preds.append(self.root.predict(X_test[i, :]))
        return np.array(preds)


class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features=0.3):
        super().__init__(min_samples_leaf, loss=np.var, max_features=0.3)
        self.max_features = max_features

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        return r2_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features=0.3):
        super().__init__(min_samples_leaf, loss=gini, max_features=0.3)
        self.max_features = max_features

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        return accuracy_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        return LeafNode(y, np.bincount(y,minlength=self.unq_classes))
