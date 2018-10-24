#Imports
import data_processing
import utils
#Inside imports
import sys
sys.path.insert(0,'/home/kamilsmolak/PycharmProjects/CHASE/Database')
from data_feeder import WaterDataFeeder, GeoDataFeeder

#Outer imports
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from psycopg2 import connect

from statsmodels.tsa import stattools
from statsmodels.stats import diagnostic
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
from statsmodels.iolib.smpickle import load_pickle

sns.set()
sns.set_style("whitegrid")
import itertools


from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from math import sqrt
import random
import scipy.stats

global day_cut
day_cut = 24

def continous_predictor(regressor, x_test, y_test):
    preds = []
    x_test = x_test[0,:].reshape(1,-1)
    for iters in range(len(y_test)):
        pred = regressor.predict(x_test).reshape(-1,1)
        preds.append(pred[0][0])
        x_test = np.append(x_test[:,1:],pred,axis=1)
    preds = np.array(preds).reshape(-1,1)
    return preds

def ML_internal(train_data, lag_size, folds, regressor):
    """
    Predicts time-series using internal data
    :param train_data: Train/test data
    :param lag_size: size of the lag
    :param folds: Number of folds in CV process
    :param regressor: defined regressor
    :return: mean error, trained model, residuals, predictions and (temporary) identifier
    """
    error = []
    for k in range(folds):
        train, test, ident = data_processing.leaveweek(train_data, 14)  # todo: CV
        x_train, y_train = data_processing.stride_data(train, lag_size)
        y_train = y_train.reshape(-1, 1)

        x_test, y_test = data_processing.stride_data(test, lag_size)
        y_test = y_test.reshape(-1, 1)

        regressor.fit(x_train, y_train)

        preds = continous_predictor(regressor, x_test, y_test)

        error.append(utils.RMSE(y_test, preds))
        preds = np.array(preds).reshape(-1, 1)
        res = y_test - preds

        stacked = np.hstack((y_test, preds))
        stacked_df = pd.DataFrame(data=stacked, columns=['org', 'pred'])

    preds = pd.DataFrame(preds, index=test.index[lag_size:])  # todo: find another way for datetime embedding
    return np.mean(error), regressor, res, preds, ident


def ML_external(train_data, lag_size, folds, regressor, exog_data, lagsx):
    pass


def ML_predict(train_data, lag_size, folds, method='RF', exog_data = None, lagsx = None): #todo: split to internal and external interface
    """
    Makse machine learning prediction based only on historical data
    :param train_data: DataFrame with training-test data
    :param lag_size: lenght of lag
    :param folds: number of folds to perform
    :param method: what method to use ('RF','SVR','ET') default == 'RF'
    :return: mean error, trained model, residuals, predictions and (temporary) identifier
    """

    if method == 'RF':
        rfr = RandomForestRegressor(n_estimators=590,criterion='mse', oob_score=True)
    if method == 'SVR':
        rfr = SVR(kernel='linear', epsilon=2.831, C=1.661)
    if method == 'ET':
        rfr = ExtraTreesRegressor(n_estimators=680, criterion='mse', oob_score=True, bootstrap=True)

    if exog_data is not None and lagsx is not None:
        print("Running exogenous mode")
        error, regressor, res, preds, ident = ML_external(train_data, lag_size, folds, rfr, exog_data, lagsx)
    else:
        print("Running internal mode")
        error, regressor, res, preds, ident = ML_internal(train_data, lag_size, folds, rfr)
        return error, regressor, res, preds, ident


def calculate_best(method,sector,train_begin,train_end,validation_begin,validation_end,doff=True):
    # preprocessing read/parametrize/embed/align TRAIN
    train_aligned = data_processing.prepare(sector,train_begin,train_end)
    validation_aligned = data_processing.prepare(sector,validation_begin,validation_end)

    train = train_aligned[1]
    xtrain = train_aligned[0].fillna(0)

    validation = validation_aligned[1]
    xvalidation = validation_aligned[0].fillna(0)

    lags = 168
    lagsx = 5

    if doff:
        e, model, res, final_preds, ident = ML_predict(train, lags, 1, method = method)
        print("MEAN: %f" % e)

        x_validation, y_validation = data_processing.stride_data(validation,lags)

        valid_preds = continous_predictor(model,x_validation, y_validation)

        print("VALIDATION: %f" % utils.RMSE(y_validation, valid_preds)) #todo: Investigate validation step, raises high RMSE
        valid_preds = np.array(valid_preds).reshape(-1, 1)
        #res = y - valid_preds

    xtrain = xtrain.fillna(0)

    e, model, res, final_xpreds, ident2 = ML_predict(train, lags, 1, method=method, xtrain, lagsx)
    print("MEAN: %f" % e)
    #x, y = xprepare(validation, xvalidation, lags, lagsx)

    valid_preds = []
    as_strided = np.lib.stride_tricks.as_strided
    x = as_strided(validation, (1, lags), (validation.values.strides * 2))
    y = np.array(validation[lags:].values).reshape(-1, 1)
    xx = as_strided(xvalidation, (len(xvalidation) - lagsx + 1, lagsx), (xvalidation.values.strides * 2))[
              lags - lagsx + 1:]

    for iters in range(len(y)):
        x_array = np.hstack((x, xx[iters].reshape(1, -1)))
        pred = model.predict(x_array).reshape(-1, 1)
        valid_preds.append(float(pred))
        x = np.append(x[:, 1:], pred, axis=1)

    #pred = model.predict(x).reshape(-1, 1)
    print("VALIDATION: %f" % RMSE(y, valid_preds))

    valid_preds = np.array(valid_preds).reshape(-1, 1)
    #res = y - pred

    ident2 += 'g'
    return final_preds,final_xpreds, ident, ident2