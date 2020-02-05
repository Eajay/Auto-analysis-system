import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

class MachineLearning:
    def __init__(self):
        self.X = None
        self.y = None
        self.cv = 6
        self.scores = ["accuracy", "precision", "recall", "f1"]

    def set_data(self, feature_data, label):
        self.X = feature_data
        self.y = label

    def logistic_regression(self, C=1.0, penalty='l2', solver='lbfgs'):
        assert solver in ["newton-cg", "lbfgs", "liblinear", "sag", "saga"], "Solver not found"
        l_reg = LogisticRegression(penalty=penalty, solver=solver, C=1.0)
        res = {}
        for score in self.scores:
            tmp = cross_val_score(l_reg, self.X, self.y, scoring=score, cv=self.cv)
            res[score] = "%s: %0.3f (+/- %0.3f)" % (score, tmp.mean(), tmp.std())
        return res

    def SVM(self, kernel='rbf', C=1.0):
        assert kernel in ["linear", "poly", "rbf", "sigmoid"], "Kernel not found"
        svm = SVC(kernel=kernel, C=C, gamma='scale')
        res = {}
        for score in self.scores:
            tmp = cross_val_score(svm, self.X, self.y, scoring=score, cv=self.cv)
            res[score] = "%s: %0.3f (+/- %0.3f)" % (score, tmp.mean(), tmp.std())

        return res

    def KNN(self):
        n_neighbors = int(math.sqrt(len(self.y)))
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        res = {}
        for score in self.scores:
            tmp = cross_val_score(knn, self.X, self.y, scoring=score, cv=self.cv)
            res[score] = "%s: %0.3f (+/- %0.3f)" % (score, tmp.mean(), tmp.std())
        return res

    def RandomForest(self, n_estimators=100):
        random_forest = RandomForestClassifier(max_depth=3, n_estimators=n_estimators)
        res = {}
        for score in self.scores:
            tmp = cross_val_score(random_forest, self.X, self.y, scoring=score, cv=self.cv)
            res[score] = "%s: %0.3f (+/- %0.3f)" % (score, tmp.mean(), tmp.std())
        return res

    def K_Means(self):
        n_cluster = len(set(self.y))
        k_means = KMeans(n_clusters=n_cluster)
        res = {}
        for score in self.scores:
            tmp = cross_val_score(k_means, self.X, self.y, scoring=score, cv=self.cv)
            res[score] = "%s: %0.3f (+/- %0.3f)" % (score, tmp.mean(), tmp.std())
        return res

    def Naive_Bayes(self, smoothing=1e-9):
        gnb = GaussianNB(var_smoothing=smoothing)
        res = {}
        for score in self.scores:
            tmp = cross_val_score(gnb, self.X, self.y, scoring=score, cv=self.cv)
            res[score] = "%s: %0.3f (+/- %0.3f)" % (score, tmp.mean(), tmp.std())
        return res






