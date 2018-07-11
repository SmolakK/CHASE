import numpy as np
from database_connector import DatabaseConnector

class DataFeeder():

    def __init__(self):
        dbcon = DatabaseConnector('chase')
        self.cur = dbcon.connect()
        return self.cur


class WaterDataFeeder(DataFeeder):

    def __init__(self):
        self._cur = DataFeeder.__init__(self)
        self._flow = []
        self._time = []
        self._sectors = []

    @property
    def flows(self):
        return self._flow

    @property
    def time(self):
        return self._time

    @property
    def sectors(self):
        return self._sectors

    def bucket(self, bucket_size, sector, date_begin = '2017-01-01', date_end = '2017-12-31'):
        """
        :param bucket_size: size of aggregation window in minutes (int)
        :param date_begin: date to start from (string YYYY-MM-DD)
        :param date_end: date to end with (string YYYY-MM-DD)
        :param sector: sector identifier
        """
        self._cur.execute("""SELECT time_bucket('%s minutes', time) as interval, avg(flow), sector
        FROM water 
        WHERE flow > 0 and flow < 10000 and time >= %s and time <= %s and sector = %s
        GROUP BY interval, sector
        ORDER by interval""",[bucket_size,date_begin,date_end, str(sector)])

        fetched = np.array(self._cur.fetchall()).reshape(-1,3)
        self._flow = fetched[:,1]
        self._time = fetched[:,0]
        self._sectors = fetched[:,2]


class WeatherDataFeeder(DataFeeder):

    def __init__(self):
        self._cur = DataFeeder.__init__(self)
        self._stationids = []
        self._time = []
        self._temperatures = []
        self._pressures = []
        self._rhs = []
        self._sectors = []

    @property
    def stationid(self):
        return self._stationids

    @property
    def time(self):
        return self._time

    @property
    def temperatures(self):
        return self._temperatures

    @property
    def pressures(self):
        return self._pressures

    @property
    def rhs(self):
        return self._rhs

    @property
    def sectors(self):
        return self._sectors

    def bucket(self, bucket_size, sector, date_begin = '2017-01-01', date_end = '2017-12-31'):
        """
        :param bucket_size: size of aggregation window in minutes (int)
        :param sector: sector identifier
        :param date_begin: date to start from (string YYYY-MM-DD)
        :param date_end: date to end with (string YYYY-MM-DD)
        """
        self._cur.execute("""SELECT time_bucket('%s minutes', time) as interval, avg(temp), avg(pressure),
        avg(rh), sector 
        FROM weather
        WHERE time >= %s and time <= %s and sector = %s
        GROUP BY interval, sector
        ORDER by interval""",[bucket_size, date_begin, date_end, str(sector)])

        fetched = np.array(self._cur.fetchall()).reshape(-1,5)
        self._time = fetched[:, 0]
        self._temperatures = fetched[:,1]
        self._pressures = fetched[:,2]
        self._rhs = fetched[:,3]
        self._sectors = fetched[:,4]


class GeoDataFeeder(DataFeeder):

    def __init__(self):
        self._cur = DataFeeder.__init__(self)
        self._time = []
        self._coordinates = []
        self._ids = []
        self._sectors = []

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def time(self):
        return self._time

    @property
    def ids(self):
        return self._ids

    @property
    def sectors(self):
        return self._sectors

    def raw_data(self, sector, date_begin='2017-01-01', date_end='2017-12-31'):
        """
        :param date_begin: date to start from (string YYYY-MM-DD)
        :param date_end: date to end with (string YYYY-MM-DD)
        :param sector: sector identifier
        """
        self._cur.execute("""SELECT time, lon, lat, id, sector 
        FROM mobile WHERE time >= %s and time <= %s and sector = %s""",
                               [date_begin,date_end, str(sector)])

        fetched = np.array(self._cur.fetchall()).reshape(-1,5)

        self._coordinates = fetched[:,1:3]
        self._time = fetched[:,0]
        self._ids = fetched[:,3]
        self._sectors = fetched[:,4]
