# -*- coding: utf-8 -*-
"""
 * Created by PyCharm Community Edition.
 * Project: nubank-puzzle
 * Author name: Iraquitan Cordeiro Filho
 * Author login: pma007
 * File: nubank-puzzle
 * Date: 10/9/15
 * Time: 17:07
 * To change this template use File | Settings | File Templates.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cross_validation import KFold, train_test_split
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso, RandomizedLasso
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

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
# x_tr = x_tr[::subsample_rate, :]
# y_tr = y_tr[::subsample_rate]
tt = GridSearchCV()

# Training step
r2scores = []
dummy_r2scores = []
n_etim = 50
n_fs = 10
kf = KFold(x_tr.shape[0], n_folds=n_fs)
train_type = 'regressor'
if train_type == 'regressor':
    mdl = RandomForestRegressor(n_estimators=n_etim, n_jobs=-1)
    dummy_dml = DummyRegressor(strategy='mean')
else:
    raise Exception('Train type not defined!')

start_time = datetime.now()
fold = 1
for train, test in kf:
    fold_start_time = datetime.now()
    mdl.fit(x_tr[train, :], y_tr[train])
    dummy_dml.fit(x_tr[train, :], y_tr[train])
    if train_type == 'regressor':
        y_pred = mdl.predict(x_tr[test, :])
        dummy_y_pred = dummy_dml.predict(x_tr[test, :])
    r2scores.append(r2_score(y_tr[test], y_pred))
    dummy_r2scores.append(r2_score(y_tr[test], dummy_y_pred))
    fold_elapsed_time = datetime.now() - fold_start_time
    print('Fold {} finished in {}'.format(fold, fold_elapsed_time))
    fold += 1
elapsed_time = datetime.now() - start_time
print('{}-fold process finished in {}'.format(fold, elapsed_time))
print('R2 scores: {}\nR2 mean score: {}'.format(r2scores, np.mean(r2scores)))
print('Dummy R2 scores: {}\nDummy R2 mean score: {}'.format(dummy_r2scores, np.mean(dummy_r2scores)))

# Testing step


# print(train.shape)
# print(test.shape)

# Using Pipeline
# fsel_estimators = [('rand_lasso', RandomizedLasso(n_jobs=-1)), ('k_best', SelectKBest(f_regression)),
#                    ('extra_r_trees', ExtraTreesRegressor(n_jobs=-1))]
# combined_fsel = FeatureUnion(fsel_estimators, n_jobs=-1,
#                              transformer_weights={'rand_lasso'})
# pipeline = Pipeline([
#     ('standardize', StandardScaler()),
#     ('feature_selection', combined_fsel),
#     ('regression', RandomForestRegressor())
#     ])
#
# param_grid = dict(feature_selection__rand_lasso__alpha=['aic', 'bic'],
#                   feature_selection__k_best__k=[10, 20, 30, 50, 70, 80],
#                   feature_selection__extra_r_trees__n_estimators=[10, 20, 30, 50, 70, 80]
#                   )
#
# grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10, n_jobs=-1, cv=n_fs, scoring=r2_score)
