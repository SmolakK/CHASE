import preprocessing
import data_reader
import pandas as pd
import datetime
import numpy as np


def stride_data(data,lag):
    """
    Uses numpy tool stride_tricks to prepare arrays containing number of lags offset by one step at each row
    :param data: DataFrame with time-series
    :param lag: Data lag (number of columns of array)
    :return: Stride for X and Y
    """
    as_strided = np.lib.stride_tricks.as_strided

    x_array = as_strided(data, (len(data) - lag, lag), (8,8))
    y_array = np.array(data[lag:].values).reshape(-1,1)
    return x_array,y_array


def leaveweek(data,days):
    """
    Splits dataset into train and test with test of given length
    :param data: DataFrame with time-series
    :param days: Length of test set
    :return: train and test and (temporary) identifier of forecast time
    """
    begin = data.index[-1] - datetime.timedelta(days=days)
    mindate = []

    for element in data.index:
        if element.date() == begin.date():
            mindate.append(element)
    begin = min(mindate)

    begin_ind = np.where(data.index==begin)[0][0]

    test = data.iloc[begin_ind::]
    train = data.iloc[:begin_ind]
    return train,test, str(begin.date())


def parametrize(water,geo,begin):
    """
    Parametrizes geo time-series dataframe
    :param water: water time-series dataframe
    :param geo: geo time-series dataframe
    :return: parametrized geo dataset
    """
    decay_size, offset = preprocessing.best_decay_offset(water, geo, 5, begin, False)
    geo_decayed = preprocessing.decay(geo, decay_size[0])  # put decay
    geo_norm = preprocessing.standardise_days(geo_decayed)
    geo_regular = preprocessing.regular_week(geo_norm, offset[0])  # put offset
    return geo_regular


def prepare(sector,begin,end,doff=True):
    """
    :paam sector: DMA id
    :param begin: start date
    :param end: end date
    :param doff: check if decay and offset have to be applied (default=True)
    :return: aligned dataframe (geodata/waterdata)
    """
    if isinstance(begin, str):
        begin = datetime.datetime.strptime(begin,'%Y-%m-%d')
    if isinstance(end, str):
        end = datetime.datetime.strptime(end,'%Y-%m-%d')

    geodf = data_reader.geo_to_df(sector, begin, end)['count']  # read geo data
    geodf = preprocessing.filter_geo(geodf)  # filter geo data

    waterdf = data_reader.water_to_df(60, sector, begin, end)['flow']  # read water data

    geodf = pd.DataFrame(geodf)
    waterdf = pd.DataFrame(waterdf)

    if doff:
        geo_regular = parametrize(waterdf, geodf, begin)

    water_norm = preprocessing.standardise_days(waterdf)
    geo_embedded = preprocessing.embed(geo_regular, begin, 9)
    stan_align = geo_embedded.align(water_norm, join='inner', axis=0)
    align = geo_embedded.align(waterdf, join='inner',axis=0)

    # todo: check if applying standardised forecast improves results
    return align