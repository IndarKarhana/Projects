import numpy as np
from sklearn.utils import resample

from dtree import *

class RandomForest621:
    def __init__(self, n_estimators=10,oob_score=False, reg = True):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan


    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        self.trees = []
        self.oob_indexes_ = []  # List to store OOB indexes for each tree

        for _ in range(self.n_estimators):


            bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
            oob_indices = np.ones(len(X), dtype=bool)
            oob_indices[bootstrap_indices] = False

            # Save the OOB indices
            self.oob_indexes_.append(oob_indices)

            # Create and fit decision tree
            if self.reg == True:
                tree = RegressionTree621(min_samples_leaf = self.min_samples_leaf, max_features=self.max_features)
            else:
                tree = ClassifierTree621(min_samples_leaf = self.min_samples_leaf, max_features=self.max_features) 
                tree.unq_classes = len(np.unique(y))

            #tree fit 
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])

            # Append the trained tree to the list of estimators
            self.trees.append(tree)
        
        if self.oob_score:
            self.oob_score_ = self.compute_oob_score(X, y)       
            

class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.trees = []
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.reg = True

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        predi = [] 
        toti = []
        for i in self.trees:
            pred_f = i.predict(X_test)
            preds = np.array([j.prediction * j.n for j in pred_f])
            tot = np.array([j.n for j in pred_f])
            predi.append(preds)
            toti.append(tot)
        predi = np.sum(np.array(predi), axis = 0)
        toti = np.sum(np.array(toti), axis = 0)
        return predi/toti
        

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        return r2_score(y_test,self.predict(X_test))
    
    def compute_oob_score(self, X, y):
        """
        Compute the out-of-bag (OOB) validation score estimate.
        """
        oob_predictions = np.zeros_like(y, dtype=float)
        oob_toti = np.zeros_like(y, dtype=float)
        for tree, oob_indices in zip(self.trees, self.oob_indexes_):
            pred_f = tree.predict(X[oob_indices])
            preds = [j.prediction * j.n for j in pred_f]
            tot = [j.n for j in pred_f]
            oob_predictions[oob_indices] += preds
            oob_toti[oob_indices] += tot
        
        unused_obb = oob_toti != 0
        
        predi_ = oob_predictions[unused_obb]/oob_toti[unused_obb]
        return r2_score(y[unused_obb], predi_)
        
class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.max_features = max_features
        self.reg = False

    def predict(self, X_test) -> np.ndarray:
        predi = [] 
        for i in self.trees:
            pred_f = i.predict(X_test)
            preds = [j.prediction for j in pred_f]
            predi.append(preds)
        predi = np.sum(np.array(predi), axis = 0)
        predi += 1
        return np.argmax(predi, axis = 1)
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        return accuracy_score(self.predict(X_test), y_test)
    
    def compute_oob_score(self, X, y):
        """
        Compute the out-of-bag (OOB) validation score estimate.
        """
        oob_predictions = np.zeros((len(y),len(np.unique(y))), dtype=float)
        for tree, oob_indices in zip(self.trees, self.oob_indexes_):
            pred_f = tree.predict(X[oob_indices])
            preds = [j.prediction for j in pred_f]
            oob_predictions[oob_indices] += np.array(preds)

        oob_predictions += 1
        return accuracy_score(y,np.argmax(oob_predictions, axis = 1))