#! /bin/bash/env python3

# data pre-processing
import os
import re
import numpy as np
import pandas as pd
import skbio.stats.composition as comp
import sklearn.model_selection as mod
import sklearn.feature_selection as fs
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
import matplotlib.pyplot as plt

# modeling
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Set wd
os.chdir("/Users/clairessabrown/Desktop/data_sci_fun/microbiome")

# Read in raw ASV table: 104 samples, ~30k ASVs
unscaled_tab = pd.read_csv("micro_features.tab", sep="\t")

# Read in labels
with open("micro_labels.tab") as f:
    labs = f.read().split("\t")
# Remove new-line characters that have numbers after them
regex = re.compile(r'\n.*')
labels = [re.sub(regex, "",e) for e in labs]
# Remove first element "x"
labels.pop(0)

# Ensure that this is not the rarefied ASV table
sample_counts = unscaled_tab.sum(axis=1) # T

# Perform total sum scaling normalization (TSS)
scaled = unscaled_tab.div(unscaled_tab.sum(axis=1), axis=0)
# scaled.sum(axis=1) # check

# Substitute zeros with small pseudocounts since...
zeros_scaled = comp.multiplicative_replacement(scaled) # numpy.ndarray

# Isoform log transform since...
ilr_transformed = comp.ilr(zeros_scaled)

# Convert ndarray back to dataframe because...
df_ilr_transformed = pd.DataFrame(ilr_transformed, index=scaled.index, columns=scaled.columns)

########################################################################################################
# Decision tree methods tended to perform well
# HFE OTU feature reduction method brought a substantial performance improvement for nearly all methods
# After feature reduction most methods performed similarly so need to do that
########################################################################################################

# Split data into test and training sets
# Do before feature selection so features selected from training set, not whole dataset
train, test, y_train, y_test = mod.train_test_split(df_ilr_transformed, labels, test_size=0.25)


#######################################################################################################################
############################################## Feature selection/reduction ############################################
#######################################################################################################################
# f test for classification; selecting best 15
f_15selector = fs.SelectKBest(k=5) # Select best 15 features (ASVs) using the f test for classification
# Build model
f_15selector.fit(train, y_train)
f_feats = f_15selector.get_support()
# len(f_feats[f_feats == True]) # 15 as expected
l_f_feats = list(f_feats)
# Select top 15 features (ASVs)
important_f_feats = train.iloc[:,l_f_feats]


# Mutual information (MI): relies on nonparametric methods based on entropy estimation from k-nearest neighbors distances
mi_selector = SelectKBest(mutual_info_classif, k=5)
# Build model algorithm
mi_selector.fit(train, y_train)
# Get vector of 15 important features
mi_feats = list(mi_selector.get_support())
important_mi_feats = train.iloc[:,mi_feats]

# Find ASVs that are matching between the two feature reduction methods if any
s_important_mi_feats = set(important_mi_feats)
s_important_f_feats = set(important_f_feats)
len(s_important_mi_feats.difference(s_important_f_feats)) # 15: features selected between methods don't match


#######################################################################################################################
############################################## Model Building/Testing #################################################
#######################################################################################################################

# XGBoost with features selected from mutual information model
xgboost_mi = XGBClassifier()
# Train XGBoost model
xgboost_mi.fit(important_mi_feats, y_train)
# Make predictions with xgboost model on training dataset
xgboost_mi_train_preds = xgboost_mi.predict(important_mi_feats)
# Get accuracy of model by comparing true labels with predicted ones
xgboost_mi_train_accuracy = accuracy_score(y_train, xgboost_mi_train_preds) # 1.0
# See confusion matrix
confusion_matrix(y_train, xgboost_mi_train_preds, list(set(labels)))
# >>> list(set(labels))
# ['Group 1', 'Group 2', 'Group 3']
# array([[24,  0,  0],
#        [ 0, 17,  0],
#        [ 0,  0, 37]])

# Random forest

