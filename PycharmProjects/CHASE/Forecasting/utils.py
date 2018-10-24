import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
import scipy

#utils


def plot_series(series):
    plt.plot(series)
    plt.show()


def RMSE(original_df,forecasted_df):
    mse = mean_squared_error(original_df, forecasted_df)
    rmse = sqrt(mse)
    print('RMSE: %.3f' % rmse)
    return rmse


def TwoFold(data,size):
    train,test = train_test_split(data,train_size=size,test_size=1-size,shuffle=False)
    return train, test


def signal_corr(sig1, sig2):
    return scipy.stats.pearsonr(sig1, sig2)[0]
