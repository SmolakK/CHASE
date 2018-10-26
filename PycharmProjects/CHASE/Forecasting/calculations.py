from data_reader import geo_to_df, water_to_df
from preprocessing import regular_week, embed, best_decay_offset
from water import calculate_best
import data_processing
from water import ML_predict
import datetime

import sys
sys.path.insert(0,'/home/kamilsmolak/PycharmProjects/CHASE/Database')
from database_connector import DatabaseConnector


def get_sector_list():
    dbcon = DatabaseConnector('chase')
    cur = dbcon.connect()
    cur.execute("""SELECT gid FROM sectors""")
    row = cur.fetchall()
    sector_list = [r[0] for r in row]
    return sector_list


def dates_iterate(begin,length):
    if isinstance(begin, str):
        begin = datetime.datetime.strptime(begin,'%Y-%m-%d')

    for days in range(length):
        forecast_begin = begin + datetime.timedelta(days=days)
        print(forecast_begin)


dates_iterate('2017-10-01',61)


#calculate_best('ET','32','2017-10-01','2017-12-01','2017-12-01','2017-12-15')