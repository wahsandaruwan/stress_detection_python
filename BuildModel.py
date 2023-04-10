# -----Imports-----
import pandas as pd
import pickle as pk

from Utility import *
from MachineLearning import *

# -----Read the dataset-----
data = pd.read_csv("./Data/final_dataset.csv")

# -----Min-max feature scaling-----
data_min_max_scaled = min_max_feature_scaling(data)

# -----Split dataset-----
X_train, X_test, y_train, y_test = split_datasets(data_min_max_scaled, data)

# -----Create multiple machine learning models-----
# Support vector classifier
svm_clf, svm_acc = support_vector_classifier(X_train, X_test, y_train, y_test)

# Random forest classifier
rnd_clf, rnd_acc = random_forest_classifier(X_train, X_test, y_train, y_test)

# K neighbors classifier
k_clf, k_acc = k_neighbors_classifier(X_train, X_test, y_train, y_test)

# Linear discriminant analysis
lda_clf, lda_acc = linear_discriminant_analyzer(X_train, X_test, y_train, y_test)

# Decision tree classifier
dec_clf, dec_acc = decision_tree_classifier(X_train, X_test, y_train, y_test)

# Ada boost classifier
ada_clf, ada_acc = ada_boost_classifier(X_train, X_test, y_train, y_test)

# Gradient boost classifier
gd_clf, gd_acc = gradient_boost_classifier(X_train, X_test, y_train, y_test)

# Print accuracies of all 7 models
print(svm_acc)
print(rnd_acc)
print(k_acc)
print(lda_acc)
print(dec_acc)
print(ada_acc)
print(gd_acc)

# -----Create a voting classification model-----
voting_clf, voting_acc = voting_classifier(X_train, X_test, y_train, y_test, svm_clf, rnd_clf, k_clf, lda_clf, dec_clf, ada_clf, gd_clf)

# Print accuracy
print(voting_acc)

# -----Save trained model-----
voting_pickle = open('./Model/voting_pickle_file', 'wb')
pk.dump(voting_clf, voting_pickle)
voting_pickle.close()