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
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# Pre-process step
num_train = train.ix[:, train.applymap(np.isreal).all(axis=0)]
np_data = np.array(num_train)
ids = np_data[:, -2]
x_tr = np_data[:, :-2]
y_tr = np_data[:, -1]

subsample_rate = 2
x_tr = x_tr[::subsample_rate, :]
y_tr = y_tr[::subsample_rate]

# Training step
r2scores = []
kf = KFold(x_tr.shape[0], n_folds=10)
train_type = 'regressor'
if train_type == 'regressor':
    mdl = RandomForestRegressor(n_estimators=10, n_jobs=-1)
elif train_type == 'classifier':
    mdl = RandomForestClassifier(n_estimators=10, n_jobs=-1)
else:
    raise Exception('Train type not defined!')

start_time = datetime.now()
fold = 1
for train, test in kf:
    fold_start_time = datetime.now()
    mdl.fit(x_tr[train, :], y_tr[train])
    if train_type == 'regressor':
        y_pred = mdl.predict(x_tr[test, :])
    elif train_type == 'classifier':
        y_pred = mdl.predict_proba(x_tr[test, :])
    r2scores.append(r2_score(y_tr[test], y_pred))
    fold_elapsed_time = datetime.now() - fold_start_time
    print('Fold {} finished in {}'.format(fold, fold_elapsed_time))
    fold += 1
elapsed_time = datetime.now() - start_time
print('{}-fold process finished in {}'.format(fold, elapsed_time))
print('R2 scores: {}\nR2 mean score: {}'.format(r2scores, np.mean(r2scores)))

# Testing step


# print(train.shape)
# print(test.shape)
