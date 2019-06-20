from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class ClassificationUtil:

    def get_svm_model(self, x, y):
        clf = svm.SVC(gamma='auto')
        clf.fit(x, y)
        return clf

    def get_dt_model(self, x, y):
        clf = tree.DecisionTreeClassifier()
        clf.fit(x, y)
        return clf

    def get_knn_model(self, x, y, k):

        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(x, y)
        return neigh

    def classify(self, model, x):
        return model.predict(x)

    def get_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def get_precision(self, y_true, y_pred):
        return precision_score(y_true, y_pred)

    def get_recall(self, y_true, y_pred):
        return recall_score(y_true, y_pred)



