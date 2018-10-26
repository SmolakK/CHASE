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
    """
    Step-by-step prediction. At each step a forecasted value is used.
    :param regressor: Trained regression model
    :param x_test: Test lags
    :param y_test: Ground truth for predictions
    :return:
    """
    preds = []
    x_test = x_test[0,:].reshape(1,-1)
    for iters in range(len(y_test)):
        pred = regressor.predict(x_test).reshape(-1,1)
        preds.append(pred[0][0])
        x_test = np.append(x_test[:,1:],pred,axis=1)
    preds = np.array(preds).reshape(-1,1)
    return preds


def exog_continous_predictor(regressor, x_test, y_test, lag):
    """
    Step-by-step prediction. At each step a forecasted value is used. Appends exogenous data.
    :param regressor: Trained regression model
    :param x_test: Test lags
    :param y_test: Ground truth for predictions
    :param lag: size of a lag of internal data
    :return:
    """
    preds = []
    x_test = x_test[0,:].reshape(1,-1)
    for iters in range(len(y_test)):
        pred = regressor.predict(x_test).reshape(-1,1)
        preds.append(pred[0][0])
        x_appended = np.append(x_test[:,1:lag],pred, axis=1)
        x_test = np.append(x_appended,x_test[:,lag:], axis=1)
    preds = np.array(preds).reshape(-1,1)
    return preds


def ML_internal(train_data, lag_size, folds, regressor):
    """
    Predicts time-series using internal data
    :param train_data: Train/test data
    :param lag_size: size of the lag
    :param folds: Number of folds in CV process
    :param regressor: Defined regressor
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

        single_error = utils.RMSE(y_test,preds)
        error.append(single_error)
        x = np.mean(test).values[0]
        print("Relative error: %s:" % (str((single_error/x)*100))+"%")
        preds = np.array(preds).reshape(-1, 1)
        res = y_test - preds

        stacked = np.hstack((y_test, preds))
        stacked_df = pd.DataFrame(data=stacked, columns=['org', 'pred'])

    preds = pd.DataFrame(preds, index=test.index[lag_size:])  # todo: find another way for datetime embedding
    return np.mean(error), regressor, res, preds, ident


def ML_external(train_data, lag_size, folds, regressor, exog_data, lagsx):
    """
    Predicts time-series using internal and external geolocated data
    :param train_data: Train/test internal data
    :param lag_size: size of the lag for internal data
    :param folds: Number of folds in CV process
    :param regressor: Defined regressor
    :param exog_data: Train/Test external data
    :param lagsx: size of the lag for external data
    :return: mean error, trained model, resiudals, prediction and (temporary) internal identifier
    """
    error = []
    for k in range(folds):
        train, test, ident = data_processing.leaveweek(train_data,14) #todo: CV
        exog_train, exog_test, exog_ident = data_processing.leaveweek(exog_data,14)

        x_train, y_train = data_processing.exog_stride_data(train, lag_size, exog_train, lagsx)
        y_train = y_train.reshape(-1, 1)

        x_test, y_test = data_processing.exog_stride_data(test, lag_size, exog_test, lagsx)
        y_test = y_test.reshape(-1, 1)

        regressor.fit(x_train,y_train)

        preds = exog_continous_predictor(regressor, x_test, y_test, lag_size)

        single_error = utils.RMSE(y_test,preds)
        error.append(single_error)
        x = np.mean(test).values[0]
        print("Relative error: %s:" % (str((single_error/x)*100))+"%")
        preds = np.array(preds).reshape(-1, 1)
        res = y_test - preds

        stacked = np.hstack((y_test, preds))
        stacked_df = pd.DataFrame(data=stacked, columns=['org', 'pred'])

    preds = pd.DataFrame(preds, index=test.index[lag_size:])  # todo: find another way for datetime embedding
    return np.mean(error), regressor, res, preds, ident


def ML_predict(train_data, lag_size, folds, method='RF', exog_data = None, lagsx = None):
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
    #todo: importance chcek
    if exog_data is not None and lagsx is not None:
        print("Running exogenous mode")
        error, regressor, res, preds, ident = ML_external(train_data, lag_size, folds, rfr, exog_data, lagsx)
        return error, regressor, res, preds, ident
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
        e, model, res, preds, ident = ML_predict(train, lags, 1, method = method)
        print("MEAN: %f" % e)

        x_validation, y_validation = data_processing.stride_data(validation,lags)

        valid_preds = continous_predictor(model, x_validation, y_validation)

        print("VALIDATION: %f" % utils.RMSE(y_validation, valid_preds)) #todo: Investigate validation step, raises high RMSE
        valid_preds = np.array(valid_preds).reshape(-1, 1)
        #res = y - valid_preds

    e, model, res, exog_preds, ident2 = ML_predict(train, lags, 1, method, xtrain, lagsx)
    print("MEAN: %f" % e)

    x_validation, y_validation = data_processing.exog_stride_data(validation,lags,xvalidation,lagsx)

    exog_valid_preds = exog_continous_predictor(model, x_validation, y_validation,lags)

    print("VALIDATION: %f:" % utils.RMSE(y_validation,exog_valid_preds))

    ident2 += 'g'
    return preds, exog_preds, ident, ident2