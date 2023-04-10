# Imports
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

def support_vector_classifier(X_train, X_test, y_train, y_test):
    """Build a machine learning model using support vector machine algorithm
    """  
    # Build model  
    svm_clf = SVC(kernel="rbf", probability=True)
    svm_clf.fit(X_train.values, y_train)

    # Accuracy
    svm_acc = svm_clf.score(X_test.values, y_test)

    return svm_clf, svm_acc
    