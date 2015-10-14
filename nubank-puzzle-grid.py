# -*- coding: utf-8 -*-
"""
 * Created by PyCharm Community Edition.
 * Project: nubank-puzzle
 * Author name: Iraquitan Cordeiro Filho
 * Author login: pma007
 * File: nubank-puzzle-grid
 * Date: 10/14/15
 * Time: 16:39
 * To change this template use File | Settings | File Templates.
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# Pre-process step
num_train = train.ix[:, train.applymap(np.isreal).all(axis=0)]  # Get numerical features
cat_train = train.ix[:, np.invert(train.applymap(np.isreal).all(axis=0))]  # Get categorical features

categorical_type = 'dictvectorizer'
cat_dict = (dict(cat_train.ix[x]) for x in range(cat_train.shape[0]))  # Categorical data generator faster then pandas
# cat_dict = cat_train.to_dict(orient='records')  # Categorical data dict
if categorical_type == 'dictvectorizer':
    vec = DictVectorizer()
    cat_data = vec.fit_transform(cat_dict).toarray()
elif categorical_type == 'featurehasher':
    feat_hash = FeatureHasher()
    hasher = FeatureHasher(input_type='string', n_features=2**8)
    cat_data = hasher.transform(cat_dict)
elif categorical_type == 'onehotencoder':
    le_data = np.empty(cat_train.shape)
    for col in range(cat_train.shape[1]):
        le = LabelEncoder()
        le_data[:, col] = le.fit_transform(cat_train.ix[:, col])
    enc = OneHotEncoder()
    cat_data = enc.fit_transform(le_data).toarray()


np_data = np.array(num_train)
ids = np.array(np_data[:, -2], dtype=np.int)
x_tr = np_data[:, :-2]
x_tr_cat = cat_data
y_tr = np_data[:, -1]

x_tr = np.hstack((np.matrix(ids).transpose(), x_tr))
x_tr_cat = np.hstack((np.matrix(ids).transpose(), x_tr_cat))
y_tr = np.vstack((ids, y_tr)).transpose()

# Divide into a train and test set for better evaluation
x_train, x_test, y_train, y_test = train_test_split(x_tr, y_tr, test_size=0.33, random_state=42)

# Sampling
subsample_rate = 8
sample_ids = np.random.choice(x_train.shape[0], x_train.shape[0]/subsample_rate, replace=False)
x_train = x_train[sample_ids, 1::]  # remove ids column
y_train = y_train[sample_ids, 1::]  # remove ids column


# Set the parameters by cross-validation
tuned_parameters = [{'regression__n_estimators': [10, 30, 50, 70, 90],
                     'regression__max_features': ['auto', 'sqrt', 'log2'],
                     'regression__warm_start': [False, True],
                     'feat_sel__k':[10, 20, 30, 50, 70, 80]},  # RandomForestRegressor
                    {'regression__n_estimators': [10, 30, 50, 70, 90],
                     'regression__max_features': ['auto', 'sqrt', 'log2'],
                     'regression__warm_start': [False, True],
                     'feat_sel__k':[10, 20, 30, 50, 70, 80]},  # ExtraTreesRegressor
                    {'regression__n_estimators': [100, 120, 150],
                     'regression__max_features': ['auto', 'sqrt', 'log2'],
                     'regression__loss': ['ls', 'lad', 'huber', 'quantile'],
                     'regression__warm_start': [False, True],
                     'feat_sel__k':[10, 20, 30, 50, 70, 80]},  # GradientBoostingRegressor
                    {'regression__n_estimators': [50, 80, 100],
                     'regression__base_estimator': [DecisionTreeRegressor(), SVR()],
                     'regression__loss': ['linear', 'square', 'exponential'],
                     'regression__warm_start': [False, True],
                     'feat_sel__k':[10, 20, 30, 50, 70, 80]},  # AdaBoostRegressor
                    {'regression__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                     'regression__C': [1, 10, 100, 1000],
                     'feat_sel__k':[10, 20, 30, 50, 70, 80]},  # SVR
                    ]

models = [RandomForestRegressor(), ExtraTreesRegressor(), GradientBoostingRegressor(), AdaBoostRegressor(), SVR()]

for model, tun_params in zip(models, tuned_parameters):
    pipeline = Pipeline([
        ('standardize', StandardScaler()),
        ('feat_sel', SelectKBest(f_regression)),
        ('regression', model)
        ])
    clf = GridSearchCV(estimator=pipeline, param_grid=tun_params, cv=5, scoring='r2', n_jobs=-1)
    clf.fit(x_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print('Test set score: {}'.format(r2_score(y_true, y_pred)))
    print()
