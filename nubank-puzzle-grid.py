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
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_rand
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_selection import f_regression, SelectKBest, RFE
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import RandomizedLasso
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# SETUP
reproducible_results = True
if reproducible_results:
    rand_state = 42
    np.random.seed(42)
else:
    rand_state = None
    np.random.seed(None)

search_method = 'randomized'  # 'Randomized'
# search_method = 'grid'  # 'Grid'
n_iter_search = 10

data_type = 'categorical'

# LOAD STEP
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# PRE-PROCESS STEP
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
else:
    raise Exception('categorical_type not supported!')

np_data = np.array(num_train)
ids = np.array(np_data[:, -2], dtype=np.int)
x_tr = np_data[:, :-2]
x_tr_cat = cat_data
y_tr = np_data[:, -1]

x_tr = np.hstack((np.matrix(ids).transpose(), x_tr))
x_tr_cat = np.hstack((np.matrix(ids).transpose(), x_tr_cat))
y_tr = np.vstack((ids, y_tr)).transpose()

# TRAIN | TEST SPLIT
if data_type == 'numerical':
    x_train, x_test, y_train, y_test = train_test_split(x_tr, y_tr, test_size=0.33, random_state=rand_state)
elif data_type == 'categorical':
    x_train, x_test, y_train, y_test = train_test_split(x_tr_cat, y_tr, test_size=0.33, random_state=rand_state)
elif data_type == 'both':
    raise Exception('Both data types not implemented yet!')
else:
    raise Exception('data_type analysis not supported!')

# SAMPLING
subsample_rate = 16
sample_ids = np.random.choice(x_train.shape[0], x_train.shape[0]/subsample_rate, replace=False)
x_train = x_train[sample_ids, 1::]  # remove ids column
y_train = y_train[sample_ids, 1::].flatten()  # remove ids column
x_test = x_test[:, 1::]  # remove ids column
y_test = y_test[:, 1::].flatten()  # remove ids column

# SEARCH VALIDATION
if search_method == 'grid':
    tuned_parameters = [{'regression__n_estimators': [10, 30, 50, 70, 90],
                         'regression__max_features': ['auto', 'sqrt', 'log2'],
                         'regression__warm_start': [False, True]},  # RandomForestRegressor
                        {'regression__n_estimators': [10, 30, 50, 70, 90],
                         'regression__max_features': ['auto', 'sqrt', 'log2'],
                         'regression__warm_start': [False, True]},  # ExtraTreesRegressor
                        {'regression__n_estimators': [100, 120, 150],
                         'regression__max_features': ['auto', 'sqrt', 'log2'],
                         'regression__loss': ['ls', 'lad', 'huber', 'quantile'],
                         'regression__warm_start': [False, True]},  # GradientBoostingRegressor
                        {'regression__n_estimators': [50, 80, 100],
                         'regression__base_estimator': [DecisionTreeRegressor(), GradientBoostingRegressor(), SVR()],
                         'regression__loss': ['linear', 'square', 'exponential']},  # AdaBoostRegressor
                        {'regression__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                         'regression__C': [1, 10, 100]}  # SVR
                        ]
    feature_sel_parameters = [dict(feat_sel__alpha=['aic', 'bic'],
                                   feat_sel__selection_threshold=[.15, .25, .35],
                                   feat_sel__scaling=[.35, .5, .75],
                                   feat_sel__sample_fraction=[.6, .75, .9]),
                              dict(feat_sel__k=[10, 20, 30, 50, 70, 80]),
                              dict(feat_sel__n_estimators=[10, 20, 30, 50, 70, 80]),
                              dict(feat_sel__estimator=[SVR(), ],
                                   feat_sel__n_features_to_select=[10, 30, 50, 70, 90, None],
                                   feat_sel__step=[1, 5, 10])]
elif search_method == 'randomized':
    tuned_parameters = [{'regression__n_estimators': sp_randint(50, 100),
                         'regression__max_features': ['auto', 'sqrt', 'log2'],
                         'regression__warm_start': [False, True]},  # RandomForestRegressor
                        {'regression__n_estimators': sp_randint(50, 100),
                         'regression__max_features': ['auto', 'sqrt', 'log2'],
                         'regression__warm_start': [False, True]},  # ExtraTreesRegressor
                        {'regression__n_estimators': sp_randint(100, 200),
                         'regression__max_features': ['auto', 'sqrt', 'log2'],
                         'regression__loss': ['ls', 'lad', 'huber', 'quantile'],
                         'regression__warm_start': [False, True]},  # GradientBoostingRegressor
                        {'regression__n_estimators': sp_randint(50, 150),
                         'regression__base_estimator': [DecisionTreeRegressor(), GradientBoostingRegressor(), SVR()],
                         'regression__loss': ['linear', 'square', 'exponential']},  # AdaBoostRegressor
                        {'regression__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                         'regression__C': sp_randint(1, 100)}  # SVR
                        ]
    feature_sel_parameters = [dict(feat_sel__alpha=['aic', 'bic'],
                                   feat_sel__selection_threshold=sp_rand(.15, .20),
                                   feat_sel__scaling=sp_rand(.35, .4),
                                   feat_sel__sample_fraction=sp_rand(.6, .3)),
                              dict(feat_sel__k=sp_randint(10, 90)),
                              dict(feat_sel__n_estimators=sp_randint(10, 90)),
                              dict(feat_sel__n_features_to_select=sp_randint(10, 90),
                                   feat_sel__step=sp_randint(1, 10))]
else:
    raise Exception('seach_method must be grid or randomized!')
models = [RandomForestRegressor(random_state=rand_state), ExtraTreesRegressor(random_state=rand_state),
          GradientBoostingRegressor(random_state=rand_state), AdaBoostRegressor(random_state=rand_state), SVR()]

feat_sel_models = [RandomizedLasso(random_state=rand_state), SelectKBest(f_regression),
                   ExtraTreesRegressor(random_state=rand_state), RFE(estimator=SVR(kernel='linear'))]
result = []
for model, tun_params in zip(models, tuned_parameters):
    for feature_sel_model, fs_params in zip(feat_sel_models, feature_sel_parameters):
        print('Setup: Regressor: {} -- Feature Selection: {}'.format(model.__class__.__name__,
                                                                     feature_sel_model.__class__.__name__))
        pipeline = Pipeline([
            ('standardize', StandardScaler()),
            ('feat_sel', feature_sel_model),
            ('regression', model)
            ])
        new_tun_params = tun_params.copy()
        new_tun_params.update(fs_params)  # Update tuned parameters with feature selection parameters
        if search_method == 'grid':
            clf = GridSearchCV(estimator=pipeline, param_grid=new_tun_params, cv=5, scoring='r2', n_jobs=-1)
        elif search_method == 'randomized':
            clf = RandomizedSearchCV(estimator=pipeline, param_distributions=new_tun_params, cv=5, scoring='r2',
                                     n_iter=n_iter_search, n_jobs=-1, random_state=rand_state)
        clf.fit(x_train, y_train)
        print("\t-> Best parameters set found on development set:")
        print('')
        print('\t\t' + str(clf.best_params_))
        print('')
        # print("Grid scores on development set:")
        # print('')
        # for params, mean_score, scores in clf.grid_scores_:
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean_score, scores.std() * 2, params))
        # print('')
        print("\t-> The model is trained on the full development set.")
        print("\t-> The scores are computed on the full evaluation set.")
        print('')
        y_true, y_pred = y_test, clf.predict(x_test)
        test_score = r2_score(y_true, y_pred)
        print('\t-> Test set score: {}'.format(test_score))
        print('')
        result.append({'setup': '{}_{}'.format(model.__class__.__name__, feature_sel_model.__class__.__name__),
                       'best_params': clf.best_params_,
                       'best_estimator': clf.best_estimator_,
                       'best_score': clf.best_score_,
                       'test_score': test_score})

# Save results to file
save_time = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
filename = search_method+'_{}.p'
pickle.dump(result, open(filename.format(save_time), "wb"))
