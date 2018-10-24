import datetime
import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt

year = 2017
holidays = [datetime.date(year,1,1), datetime.date(year,1,6),
            datetime.date(year,4,17), datetime.date(year,5,1),
            datetime.date(year, 5, 3), datetime.date(year,6,15),
            datetime.date(year, 8, 15), datetime.date(year, 11, 1),
            datetime.date(year,11,11), datetime.date(year, 12, 24),
            datetime.date(year,12,25), datetime.date(year, 12, 26)]


def filter_geo(df):
    """Filters geolocated data - deletes data for 0 and 2 am
    :param df: dataframe of time-series"""
    mask = df.index.time != datetime.time(0)
    mask2 = df.index.time != datetime.time(2)
    df = df.where(mask,other=0)
    df = df.where(mask2,other=0)
    return df


def decay(df,decay):
    """
    Adds decay to time-series
    :param df: dataframe of time-series
    :param decay: decay size
    :return: decayed TS
    """
    return df.rolling(min_periods=1, window=decay+1).sum()


def standardise_days(df):
    """Standardises data by day
    :param df: dataframe of time-series"""
    grouped = df.groupby(pd.Grouper(freq='D'))
    normalized_list = []

    for g in grouped:
        normalized = (g[1] - g[1].mean()) / (g[1].std())
        normalized_list.append(normalized)
    return pd.concat(normalized_list)


def dict_offset(dict,offset):
    """
    Adds offset to regularized week
    :param dict: dict with regularized apperances
    :param offset: offset parameter
    :return:
    """
    new_regular = {}
    if offset != 0:
        for key in dict.keys():
            new_regular[key[0] * 24 + key[1].hour + offset] = dict[key]
    else:
        for key in dict.keys():
            new_regular[key[0] * 24 + key[1].hour + key[1].minute/60.] = dict[key]
    # todo: add other resoultions (one hour is maximum now)
    return new_regular


def regular_week(df,offset=0):
    """
    Regularizes time-series weekly
    :param df: dataframe with TS to regularize
    :param offset: offset parameter
    :return: regularized TS in DataFrame
    """
    regular_week = {}
    for k, v in df.iterrows():
        isoday = k.weekday()
        time = k.time()
        if (isoday, time) in regular_week.keys():
            regular_week[(isoday, time)].append(v)
        else:
            regular_week[(isoday, time)] = [v]

    for count in regular_week:
        regular_week[count] = np.nanmean(regular_week[count])

    #Adding offset
    new_regular = dict_offset(regular_week,offset)

    df_regular_week = pd.DataFrame.from_dict(new_regular, orient='index', columns=['presence'])

    sorted_regular_week = df_regular_week.sort_index()
    return sorted_regular_week


def embed(sorted_regular_dataframe,begin,week_length):
    """
    Embeds regular weeks into datetime and checks for holidays, which are substituted by sunday
    :param sorted_regular_dataframe: regularized dataframe
    :param begin: when to begin embedding
    :param week_length: length of embedding
    :return: embedded time-series
    """
    if isinstance(begin, str):
        begin = datetime.datetime.strptime(begin,'%Y-%m-%d')

    conv_dict = {}
    for n in range(week_length):
        for ind,val in sorted_regular_dataframe.iterrows():
            added_ind = datetime.timedelta(hours=ind) + begin + datetime.timedelta(days=7 * n)
            if added_ind.date() in holidays:
                try:
                    conv_dict[added_ind] = sorted_regular_dataframe.loc[ind%24-1 + 6*24][0]
                except:
                    closestdiff = np.array(sorted_regular_dataframe.index)-(ind%24-1 + 6*24)
                    closest = closestdiff[np.argmin(abs(closestdiff))] + (ind%24-1 + 6*24)
                    conv_dict[added_ind] = sorted_regular_dataframe.loc[closest][0]
            else:
                conv_dict[added_ind] = val[0]

    regular_embedded = pd.DataFrame.from_dict(conv_dict, orient="index")
    regular_embedded = pd.Series() if regular_embedded.empty else regular_embedded.iloc[:, 0]
    return regular_embedded


def best_decay_offset(waterdf, geo_df, check_size,begin,plot=False):
    """
    Finds the best parameters maximizng the correlation between geo and water time-series
    :param waterdf: water time-series dataframe
    :param geo_df: geo time-series dataframe
    :param check_size: size of decay/offset check of size [nxn]
    :param begin: when to start searching (should be in data time range, otherwise you will get an error)
    :param plot: if you want a 2D plot showed
    :return: the best parameters array [decay, offset]
    """
    if isinstance(begin, str):
        begin = datetime.datetime.strptime(begin,'%Y-%m-%d')

    decays = []
    water_norm = standardise_days(waterdf) #Standardise water data

    for dec in range(check_size): #Iterate through check_size of decay for geo data
        geo_decayed = decay(geo_df, dec) #apply decay of check size
        geo_norm = standardise_days(geo_decayed) #Standardise over decayed data

        correlations = []

        for off in range(check_size): #iterate through check size of offset for geo data
            sorted_regular_week = regular_week(geo_norm, off)

            regular_embeded = embed(sorted_regular_week, begin, 4)

            water_norm = water_norm.dropna()
            regular_embeded = pd.DataFrame(regular_embeded.dropna())
            aligned = waterdf.align(regular_embeded, join='inner', axis=0)
            correlations.append(utils.signal_corr(aligned[0].values, aligned[1].values))

        decays.append(correlations)

    decays = np.array(decays).reshape(check_size, -1)
    np.nan_to_num(decays,0)

    if plot:
        plt.clf()
        plt.contour(decays, corner_mask=True, colors='k', linewidths=.5)
        plt.contourf(decays, cmap='RdYlBu_r')
        plt.colorbar()
        plt.xlabel('Offset')
        plt.ylabel('Decay size')
        plt.title("Pearson's correlation for various offsets and decay sizes", size=12)
        plt.show()
    print("Correlation: %f" % np.max(decays) + " For parameters %i/%i" % (np.where(decays == np.max(decays))))
    return np.where(decays == np.max(decays))
