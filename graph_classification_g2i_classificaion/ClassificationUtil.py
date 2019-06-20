from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



class ClassificationUtil:

    def get_svm_model(X,Y):
        clf = svm.SVC(gamma='auto')
        clf.fit(X, Y)
        return clf

    def get_dt_model(X,Y):
        clf = tree.DecisionTreeClassifier()
        clf.fit(X, Y)
        return clf

    def get_knn_model(X,Y,k):

        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X, Y)
        return neigh

    def classify(model,X):
        return model.predict(X)

    def get_accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def get_precision(y_true, y_pred):
        return precision_score(y_true, y_pred)

    def get_recall(y_true, y_pred):
        return recall_score(y_true, y_pred)



