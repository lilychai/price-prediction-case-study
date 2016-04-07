from price_predictor import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone

import pandas as pd
import numpy as np
import zipfile


start = timeit.default_timer()      # wall time, not CPU time

print 'Loading Files:'
print '  Inflation.csv...'
start_loadfile = timeit.default_timer()
df_inflation = pd.read_csv('../data/Inflation.csv', sep='\t')
df_inflation.index = df_inflation.Year
df_inflation = df_inflation.Ave.sort_index(ascending=False)     # average annual inflation

print '  Train.csv...'
df_train = pd.read_csv(zipfile.ZipFile('../data/Train.zip', mode='r').open('Train.csv'), low_memory=False)

print '      (time: %.5f s)' % float(timeit.default_timer() - start_loadfile)


## Create and Train
print 'Train 6 models:'
predictors = {}
for pg, grp in df_train.groupby('ProductGroup'):
    estimator = Pipeline([('scaler', MinMaxScaler()),
                          ('estimator', RandomForestRegressor(50, max_depth=13, max_features=0.4, n_jobs=-1)),
                         ])

    predictor = PricePredictor(estimator, df_inflation)
    predictor.fit(grp)
    predictors[pg] = predictor


## Predict
print 'Load final test files:'
print '  test.csv...'
df_test = pd.read_csv(zipfile.ZipFile('../data/Test.zip', mode='r').open('test.csv'), low_memory=False)
print '  test_soln.csv...'
test_solution = pd.read_csv('../data/do_not_open/test_soln.csv')

print 'Predicting 6 models:'
y_final = np.zeros(len(df_test))
for pg, grp in df_test.groupby('ProductGroup'):
    inds = np.arange(len(df_test))[(df_test.ProductGroup == pg).values]
    y_pred = predictors[pg].predict(grp)
    y_final[inds] += y_pred


error = rmsle(test_solution['SalePrice'].values, y_final)
print '>> RMS log error:', error

print '      (time: %.5f s)' % float(timeit.default_timer() - start)

## write results to file
df_results = pd.DataFrame(np.c_[df_test['SalesID'].values, y_final], columns = ['SalesID', 'SalePrice'])
df_results.to_csv('../result_oo_6models.csv', index=False)   # 0.267527092345 -- with ProductGroup dummies
                                                             # 0.268046944016 -- without ProductGroup dummies
