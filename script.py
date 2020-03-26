"""
    Python script to submit as a part of the project of ELTP 2020 course.
    
    This script serves as a template. Please use proper comments and meaningful variable names.
"""

"""
    Group Members:
        (1) Ruixue PENG
        (2) Chaoran ZHANG
        (3) Shuang YANG
    Github Link: https://github.com/shyang97/Ensemble-Project
"""

"""
    Import necessary packages
"""
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

"""
    Text preprocessing
"""
# Read data, assign index, drop useless columns
data = pd.read_csv('X_train_update.csv',index_col=0)
data = data.drop(labels=['description','productid','imageid'],axis=1)
train_y = pd.read_csv('Y_train.csv',index_col=0)
data_test = pd.read_csv('X_test_update.csv')
data_test = data_test.drop(labels=['description','productid','imageid'],axis=1)

# Load spacy pretrained model
spacy_nlp = spacy.load("fr_core_news_sm")

# Create a function for handling accented characters
def normalize_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')

    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')

    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')
    
    return string

# Create a function for text preprocessing
def raw_to_tokens(raw_string, spacy_nlp):
    # Lower-casing
    string = raw_string.lower()
    
    # Normalize the accents
    string = normalize_accent(string)
        
    # Tokenize
    spacy_tokens = spacy_nlp(string)
        
    # Remove punctuation tokens and create string tokens
    string_tokens = [token.orth_ for token in spacy_tokens if not token.is_punct if not token.is_stop]
   
    # Join the tokens back into a single string
    clean_string =  " ".join(string_tokens)
    
    return clean_string

# Preprocess training and test text
# Training
docs_raw=data['designation']
docs_raw.reset_index(inplace=True, drop=True)
for i in range(len(docs_raw)):
    docs_raw[i]=raw_to_tokens(docs_raw[i],spacy_nlp)

# Test
docs_test_raw=data_test['designation']
docs_test_raw.reset_index(inplace=True, drop=True)
for i in range(len(docs_test_raw)):
    docs_test_raw[i]=raw_to_tokens(docs_test_raw[i],spacy_nlp)

# Vectorization - transform text into feature sets
tfidf=TfidfVectorizer()
train_X_tfidf =tfidf.fit_transform(list(docs_raw))
train_X=train_X_tfidf.toarray()
test_X_tfidf =tfidf.transform(list(docs_test_raw))
test_X=test_X_tfidf.toarray()

"""
    Methods implementing the models
"""
# RandomForestClassifier
def run_random_forest(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
#     create the proper instance of the model with the best hyperparameters you found
    clf = RandomForestClassifier(n_estimators=150, max_depth=200, oob_score=True)
#     fit the model with a given training data
    clf.fit(X_train, y_train)

#     run the prediction on a given test data
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

#     return accuracy and F1 score
    return accuracy, f1

# DecisionTreeClassifier
def run_decision_tree(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return accuracy, f1

# BaggingClassifier
def run_bagging(X_train, y_train, X_test, y_test):
    clf = BaggingClassifier(max_samples=0.5, max_features=0.5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return accuracy, f1

# ExtraTreesClassifier
def run_extratree(X_train, y_train, X_test, y_test):
    clf = ExtraTreesClassifier(min_samples_split=2,min_samples_leaf=1,bootstrap=False)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return accuracy, f1

#GBDTClassifier
def run_gbdt(X_train, y_train, X_test, y_test):
    clf =  GradientBoostingClassifier(max_samples=0.5, max_features=0.5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return accuracy, f1
    

# AdaBoostClassifier
def run_adaboost(X_train, y_train, X_test, y_test):
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(splitter='best', min_samples_split=12, 
                                                                   min_samples_leaf=18,criterion='gini',
                                                                   max_features=9,max_depth=30),
                                                                   learning_rate=0.1,
                                                                   n_estimators=60)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return accuracy, f1

# XGBClassifier
def run_xgboost(X_train, y_train, X_test, y_test):
    clf = XGBClassifier(n_estimators=150, max_depth=7, min_child_weight=1, subsample=0.7, colsample_bytree=0.4, learning_rate=0.1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return accuracy, f1

# LGBMClassifier
def run_lightgbm(X_train, y_train, X_test, y_test):
    clf = LGBMClassifier(max_depth=15, num_leaves=40, min_child_samples=15, learning_rate=0.1, feature_fraction=0.6,n_estimators=150)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return accuracy, f1

# CatBoostClassifier
def run_catboost(X_train, y_train, X_test, y_test):
    clf = CatBoostClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return accuracy, f1



"""
   Main function
"""
if __name__ == "__main__":
    # Split the training set into train/test subsets in order to see out-sample performance
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3)

    model_1_acc, model_1_f1 = run_decision_tree(X_train, y_train, X_test, y_test)
    model_2_acc, model_2_f1 = run_bagging(X_train, y_train, X_test, y_test)
    model_3_acc, model_3_f1 = run_random_forest(X_train, y_train, X_test, y_test)
    model_4_acc, model_4_f1 = run_extratree(X_train, y_train, X_test, y_test)
    model_5_acc, model_5_f1 = run_gbdt(X_train, y_train, X_test, y_test)
    model_6_acc, model_6_f1 = run_adaboost(X_train, y_train, X_test, y_test)
    model_7_acc, model_7_f1 = run_xgboost(X_train, y_train, X_test, y_test)
    model_8_acc, model_8_f1 = run_lightgbm(X_train, y_train, X_test, y_test)
    model_9_acc, model_9_f1 = run_catboost(X_train, y_train, X_test, y_test)


    # print the results
    print("DecisionTreeClassifier", model_1_acc, model_1_f1)
    print("BaggingClassifier", model_2_acc, model_2_f1)
    print("RandomForestClassifier", model_3_acc, model_3_f1)
    print("ExtraTreesClassifier", model_4_acc, model_4_f1)
    print("GBDTClassifier", model_5_acc, model_5_f1)
    print("AdaBoostClassifier", model_6_acc, model_6_f1)
    print("XGBClassifier", model_7_acc, model_7_f1)
    print("LGBMClassifier", model_8_acc, model_8_f1)
    print("CatBoostClassifier", model_9_acc, model_9_f1)
