#import osgeo.ogr
import os
import re
import datetime as dtime
import pytz
from database_connector import DatabaseConnector
import datetime

if __name__ == '__main__':
    dbcon = DatabaseConnector('chase')
    cur = dbcon.connect()


def shape_load(shape_path):
    """
    Loads shapes from given path to sectors table in WKT format in WGS84
    :param shape_path: path to shapefile
    :return: None
    """
    cur.execute("ALTER SEQUENCE sectors_id_seq RESTART WITH 1")  # zeroing id in table sectors
    shapefile = osgeo.ogr.Open(shape_path)  # open and read shapefile
    layer = shapefile.GetLayer(0)
    for i in range(layer.GetFeatureCount()):  # get features
        feature = layer.GetFeature(i)
        name = feature.GetField("STREFA").decode("utf-8")  # get field value
        num = feature.GetField("NUMER").decode("utf-8")
        wkt = feature.GetGeometryRef().ExportToWkt()  # export geometry to WKT
        cur.execute("INSERT INTO sectors (gid,shape,name) VALUES ('%s', ST_GeometryFromText('%s', 4326), '%s')" %
                    (num.encode("utf-8"), wkt, name.encode("UTF8")))  # write all the data into table


def shape_clear():
    """Clears sectors table"""
    cur.execute("DELETE FROM sectors")


def water_load():
    """Dla Barbary"""
    pass


def water_clear():
    """Clears water table"""
    cur.execute("DELETE FROM water")


def mobile_load(mobile_path):
    """
    Based on sample data from Selectivv, loads mobile phone data in the database
    :param mobile_path: path to mobile phones data
    :return: None
    """
    temp_file_path = mobile_path.rstrip('.csv') + '_temp.csv'  # create temporary file
    temp_file = open(temp_file_path, 'w')

    unique_users = []
    tz = pytz.timezone('Europe/London')

    print("BUILDING TEMP FILE")

    with open(mobile_path, 'r', buffering=(2 << 16) + 4) as ffile:  # read with higher buffer
        for row in ffile:
            row = row.split(',')
            row[1] = datetime.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')  # convert string to timestamptz
            #row[1] = dtime.datetime.fromtimestamp(float(row[1]),tz)  # convert UNIX to timestamptz

            if row[0].upper() not in unique_users:  # search for unique ids
                uq_user = row[0].rstrip('\r\n').upper()
                unique_users.append(row[0].upper())
                cur.execute("INSERT INTO users(id) VALUES (%s)", [uq_user])  # insert detected users to users table

            row[0] = row[0].replace('\r','').upper()
            row = ','.join(map(str, row))  # join converted list back
            temp_file.write(row)

    print("FINISHED BUILDING TEMP FILE")

    temp_file.close()
    temp_file = open(temp_file_path, 'r', buffering=(2 << 16) + 4)

    print("COPYING TEMP FILE INTO DB")
    cur.copy_expert(
        """COPY mobile(id,time,lat,lon,app) FROM STDIN WITH CSV DELIMITER AS ',' ENCODING 'utf-8' QUOTE AS '"' """,
        temp_file)

    temp_file.close()
    os.remove(temp_file_path)

    print("DATA COPIED")
    print("ASSIGING POINTS TO SECTORS")

    cur.execute("""CREATE TABLE temp_mobile AS (
SELECT sectors.gid as sector,mobile.time as time, mobile.id as id, 
    mobile.lon as lon, mobile.lat as lat, mobile.app as app
    FROM mobile, sectors 
    WHERE ST_Within(ST_SetSRID(ST_MakePoint(mobile.lon,mobile.lat),4326),sectors.shape))""")

    cur.execute("""DELETE FROM mobile""")

    cur.execute("""INSERT INTO mobile(time,lon,lat,id,sector,app) SELECT time,lon,lat,id,sector,app FROM temp_mobile""")

    cur.execute("""DROP TABLE temp_mobile""")

    print("DATA LOADED")


def mobile_clear():
    """Clears mobile table"""
    cur.execute("DELETE FROM mobile")
    cur.execute("DELETE FROM users")

def weather_load(weather_path):
    """
    Loads weather data into the database
    :param weather_path: path to weather file (CSV)
    :return: None
    """
    temp_file_path = weather_path.rstrip('.csv') + '_temp.csv'  # create temporary file
    temp_file = open(temp_file_path, 'wb')

    unique_stations = []

    print("BUILDING TEMP FILE")

    with open(weather_path, 'rb', buffering=(2 << 16) + 4) as ffile:  # read with higher buffer
        for row in ffile:
            row = re.split(' +',row)
            timetz = dtime.datetime.strptime(' '.join([row[0],row[1]]), '%Y-%m-%d ''%H:%M:%S')

            if row[2] not in unique_stations:  # search for unique ids
                unique_stations.append(row[2])
                cur.execute("INSERT INTO weather_station(station_id,lat,lon,height) VALUES (%s,%s,%s,%s)",
                            [row[2],float(row[3]),float(row[4]),float(row[5])])  # insert detected stations

            weather_record = ','.join(map(str, [row[2],timetz,row[6],row[7],row[8]]))  # join converted list back
            temp_file.write(weather_record)

    print("FINISHED BUILDING TEMP FILE")

    temp_file.close()
    temp_file = open(temp_file_path, 'rb', buffering=(2 << 16) + 4)

    print("COPYING TEMP FILE INTO DB")
    cur.copy_expert(
        """COPY weather(station_id,time,temp,pressure,rh) FROM STDIN WITH CSV DELIMITER AS ',' 
        ENCODING 'utf-8' QUOTE AS '"' """,
        temp_file)

    temp_file.close()
    os.remove(temp_file_path)

    print("DATA COPIED")
    print("ADDING STATIONS' GEOMETRY")
    cur.execute("""UPDATE weather_station SET shape = ST_SetSRID(ST_MakePoint(lon,lat),4326)""")

    print("ASSIGING POINTS TO SECTORS")

    cur.execute("""UPDATE weather SET sector = gid
    FROM (SELECT sectors.gid as gid,weather.time as timez, weather.station_id as id  FROM weather, sectors, weather_station
    WHERE weather.station_id = weather_station.station_id AND ST_Within(weather_station.shape,sectors.shape)) T1
    WHERE T1.timez = weather.time and T1.id = weather.station_id""")

    print("DATA LOADED")


def weather_clear():
    """Clears weather table"""
    cur.execute("DELETE FROM weather")
    cur.execute("DELETE FROM weather_station")
