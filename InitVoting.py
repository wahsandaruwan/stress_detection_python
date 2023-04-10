# Imports
import pandas as pd
import pickle as pk

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report

from Utility import *

# Read the dataset
data = pd.read_csv("./Data/final_dataset.csv")

# Min-max feature scaling
data_min_max_scaled = min_max_feature_scaling(data)

# Split dataset
X_train, X_test, y_train, y_test = split_datasets(data_min_max_scaled)