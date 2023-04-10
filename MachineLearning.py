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

def random_forest_classifier(X_train, X_test, y_train, y_test):
    """Build a machine learning model using random forest algorithm
    """  
    # Build model  
    rnd_clf = RandomForestClassifier(n_estimators=20, n_jobs=-1)
    rnd_clf.fit(X_train.values, y_train)

    # Accuracy
    rnd_acc = rnd_clf.score(X_test.values, y_test)

    return rnd_clf, rnd_acc
    
def k_neighbors_classifier(X_train, X_test, y_train, y_test):
    """Build a machine learning model using k neighbors algorithm
    """  
    # Build model  
    k_clf = KNeighborsClassifier(n_neighbors=4)
    k_clf.fit(X_train.values, y_train)

    # Accuracy
    k_acc = k_clf.score(X_test.values, y_test)

    return k_clf, k_acc

def linear_discriminant_analyzer(X_train, X_test, y_train, y_test):
    """Build a machine learning model using linear discriminant analysis algorithm
    """  
    # Build model  
    lda_clf = LinearDiscriminantAnalysis()
    lda_clf.fit(X_train.values, y_train)

    # Accuracy
    lda_acc = lda_clf.score(X_test.values, y_test)

    return lda_clf, lda_acc