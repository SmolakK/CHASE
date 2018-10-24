#Inside imports
import sys
sys.path.insert(0,'/home/kamilsmolak/PycharmProjects/CHASE/Database')
from data_feeder import WaterDataFeeder, GeoDataFeeder
import numpy as np
import pandas as pd


def water_to_df(bucket_size,sector,begin,end):
    """
    :param bucket_size: aggregation window
    :param sector: sector identifier
    :param begin: start date string
    :param end:  end date string
    :return: water data in DataFrame
    """
    wdf = WaterDataFeeder()
    wdf.bucket(bucket_size, sector, date_begin=begin, date_end=end) #reads data from database
    flows = np.array(wdf.flows, dtype=float).reshape(-1, 1)
    time = np.array(wdf.time).reshape(-1, 1)
    time = [t[0].replace(tzinfo=None) for t in time]

    df1 = pd.DataFrame(flows, index=time, columns=['flow'])
    return df1


def geo_to_df(sector,begin,end):
    """
    :param sector: sector identifier
    :param begin: start date string
    :param end: end date string
    :return: geo data in DataFrame
    """
    gdf = GeoDataFeeder()
    gdf.agg_data(sector, date_begin=begin, date_end=end)
    counts = np.array(gdf.counts, dtype=int).reshape(-1, 1)
    time = np.array(gdf.time).reshape(-1, 1)
    time = [t[0].replace(tzinfo=None) for t in time]

    df1 = pd.DataFrame(counts, index=time, columns=['count'])
    df1.index = pd.to_datetime(df1.index,utc=True)
    return df1
