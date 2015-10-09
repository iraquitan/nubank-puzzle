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
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# Pre-process step
np_data = np.array(train)
ids = np_data[:, -2]
x_tr = np_data[:, :-2]
y_tr = np_data[:, -1]

# Training step
r2scores = []
kf = KFold(x_tr.shape[0], n_folds=10)
for train, test in kf:
    mdl = RandomForestRegressor(n_jobs=-1).fit(x_tr[train, :], y_tr[train])
    y_pred = mdl.predict(x_tr[test, :])
    r2scores.append(r2_score(y_tr[train], y_pred))
print(r2scores)

# Testing step


print(train.shape)
print(test.shape)
