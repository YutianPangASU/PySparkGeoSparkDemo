#! /home/anaconda3 python
#-*- coding: utf-8 -*-

"""
@Author: Yutian Pang
@Date: 2020-10-25

This Python script is used to process the iff sector flight event data for flight event prediction task.

@Last Modified by: Yutian Pang
@Last Modified date: 2020-10-26
"""

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os


def event_density_feature(df_ori, df_new, lat_threshold, lon_threshold, timestamp_threshold):
    """
    This function is used to count the nearby event of the current event in df_new from the entire dataset df_ori within
    a given threshold.
    :param df_ori:
    :param df_new:
    :param lat_threshold:
    :param lon_threshold:
    :param timestamp_threshold:
    :return:
    """
    # Calculate Density Feature
    count = pd.DataFrame(index=df_new.index, columns=['ev_count_' + str(t) for t in timestamp_threshold])
    for index, row in df_new.iterrows():
        lat_idx = abs(df_ori['Lat'] - row['Lat']) < lat_threshold
        lon_idx = abs(df_ori['Lon'] - row['Lon']) < lon_threshold
        for time_thres in timestamp_threshold:
            time_idx = abs(df_ori['tEv'] - row['tEv']) < time_thres
            total_idx = time_idx & lat_idx & lon_idx
            # count unique callsigns corresponding to the current row
            count.loc[index, 'ev_count_{}'.format(time_thres)] = df_new.loc[total_idx, 'AcId'].unique().size

    # concatenate two dataframes
    df_new = pd.concat([df_new, count-1], axis=1)
    return df_new


def previous_event_sequence_feature(df_ori, df_new, timestamp_threshold):
    """
    This function calculate the previous happened event before the current event of the same callsign using the dummy
    variables.
    :param df_ori:
    :param df_new:
    :param timestamp_threshold:
    :return:
    """
    col_name = cols + ['ev_count_' + str(t) for t in timestamp_threshold] + \
               [ev.replace('EV', 'EvType_EV') for ev in df['EvType'].value_counts().index.to_list()]
    df_new = df_new.reindex(columns=col_name).replace(np.NaN, 0)

    # generate event sequence feature
    index_old = 0
    for index, row in df_new.iterrows():
        row_label = pd.get_dummies(df_ori.loc[index_old:index - 1]['EvType'], columns=['EvType'])
        row_label = row_label.rename(columns=lambda x: x.replace('EV', 'EvType_EV')).sum(axis=0)
        df_new.loc[index] = row.drop(row_label.index.to_list()).append(row_label)
        index_old = index

    return df_new


def aircraft_density_feature(df_new, iff_file_path, lat_threshold, lon_threshold, timestamp_threshold, geospark=False):
    """
    This function is to calculate the aircraft density feature of the current event using PySpark. This function has an
    option of geospark=True to speedup the query process. This will require a successful install of geospark.
    :param df_new:
    :param iff_file_path:
    :param lat_threshold:
    :param lon_threshold:
    :param timestamp_threshold:
    :param geospark:
    :return:
    """
    # spark configuration
    os.environ["SPARK_HOME"] = '/home/ypang6/anaconda3/lib/python3.7/site-packages/pyspark'
    os.environ["PYTHONPATH"] = '/home/ypang6/anaconda3/bin/python3.7'
    os.environ['PYSPARK_PYTHON'] = '/home/ypang6/anaconda3/bin/python3.7'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/ypang6/anaconda3/bin/python3.7'
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, ShortType, StringType, LongType, IntegerType, DoubleType
    from pyspark import SparkConf

    # define my schema of the database
    myschema = StructType([
        StructField("recType", ShortType(), True),  # 1  //track point record type number
        StructField("recTime", StringType(), True),  # 2  //seconds since midnigght 1/1/70 UTC
        StructField("fltKey", LongType(), True),  # 3  //flight key
        StructField("bcnCode", IntegerType(), True),  # 4  //digit range from 0 to 7
        StructField("cid", IntegerType(), True),  # 5  //computer flight id
        StructField("Source", StringType(), True),  # 6  //source of the record
        StructField("msgType", StringType(), True),  # 7
        StructField("acId", StringType(), True),  # 8  //call sign
        StructField("recTypeCat", IntegerType(), True),  # 9
        StructField("lat", DoubleType(), True),  # 10
        StructField("lon", DoubleType(), True),  # 11
        StructField("alt", DoubleType(), True),  # 12  //in 100s of feet
        StructField("significance", ShortType(), True),  # 13 //digit range from 1 to 10
        StructField("latAcc", DoubleType(), True),  # 14
        StructField("lonAcc", DoubleType(), True),  # 15
        StructField("altAcc", DoubleType(), True),  # 16
        StructField("groundSpeed", IntegerType(), True),  # 17 //in knots
        StructField("course", DoubleType(), True),  # 18  //in degrees from true north
        StructField("rateOfClimb", DoubleType(), True),  # 19  //in feet per minute
        StructField("altQualifier", StringType(), True),  # 20  //Altitude qualifier (the “B4 character”)
        StructField("altIndicator", StringType(), True),  # 21  //Altitude indicator (the “C4 character”)
        StructField("trackPtStatus", StringType(), True),  # 22  //Track point status (e.g., ‘C’ for coast)
        StructField("leaderDir", IntegerType(), True),  # 23  //int 0-8 representing the direction of the leader line
        StructField("scratchPad", StringType(), True),  # 24
        StructField("msawInhibitInd", ShortType(), True),  # 25 // MSAW Inhibit Indicator (0=not inhibited, 1=inhibited)
        StructField("assignedAltString", StringType(), True),  # 26
        StructField("controllingFac", StringType(), True),  # 27
        StructField("controllingSec", StringType(), True),  # 28
        StructField("receivingFac", StringType(), True),  # 29
        StructField("receivingSec", StringType(), True),  # 30
        StructField("activeContr", IntegerType(), True),  # 31  // the active control number
        StructField("primaryContr", IntegerType(), True),
        # 32  //The primary(previous, controlling, or possible next)controller number
        StructField("kybrdSubset", StringType(), True),  # 33  //identifies a subset of controller keyboards
        StructField("kybrdSymbol", StringType(), True),  # 34  //identifies a keyboard within the keyboard subsets
        StructField("adsCode", IntegerType(), True),  # 35  //arrival departure status code
        StructField("opsType", StringType(), True),  # 36  //Operations type (O/E/A/D/I/U)from ARTS and ARTS 3A data
        StructField("airportCode", StringType(), True),  # 37
        StructField("trackNumber", IntegerType(), True),  # 38
        StructField("tptReturnType", StringType(), True),  # 39
        StructField("modeSCode", StringType(), True)  # 40
    ])

    if geospark:
        from geospark.register import upload_jars
        from geospark.register import GeoSparkRegistrator
        upload_jars()
        from geospark.utils import GeoSparkKryoRegistrator, KryoSerializer
        SparkConf().set("spark.serializer", KryoSerializer.getName)
        SparkConf().set("spark.kryo.registrator", GeoSparkKryoRegistrator.getName)
        from geospark.utils.adapter import Adapter
        path_to_jars = "/home/ypang6/anaconda3/lib/python3.7/site-packages/pyspark/jars/"
        jars = ["geospark-sql_3.0-1.3.2-SNAPSHOT.jar", "geospark-1.3.2-SNAPSHOT.jar"]

        spark = SparkSession.builder.appName("Terminal_Area_Flight_Data_Query") \
            .config("spark.jars", ",".join([os.path.join(path_to_jars, el) for el in jars])) \
            .getOrCreate()

        GeoSparkRegistrator.registerAll(spark)

        # load iff sector data
        df = spark.read.csv(iff_file_path, header=False, sep=",", schema=myschema)

        # select columns
        cols = ['recType', 'recTime', 'acId', 'lat', 'lon', 'alt']
        df = df.select(*cols).filter(df['recType'] == 3).withColumn("recTime", df['recTime'].cast(IntegerType()))

        # register pyspark df in SQL
        df.registerTempTable("pointtable")

        # create shape column in geospark
        spatialdf = spark.sql(
            """
            SELECT ST_Point(CAST(lat AS Decimal(24, 20)), CAST(lon AS Decimal(24, 20))) AS geom, recTime
            FROM pointtable
            """)

        spatialdf.createOrReplaceTempView("spatialdf")

        # register pyspark spatialdf in SQL
        spatialdf.registerTempTable("spatialdf")

        count = pd.DataFrame(index=df_new.index, columns=['ac_count_' + str(t) for t in timestamp_threshold])
        for index, row in df_new.iterrows():
            for t_threshold in timestamp_threshold:
                ev_time = int(row['tMidnightSecs'] + row['tStart'])
                lat = row['Lat']
                lon = row['Lon']

                rectangular_query = "SELECT * FROM (SELECT * FROM spatialdf WHERE recTime>={} AND recTime<{}) WHERE ST_Contains(ST_PolygonFromEnvelope({}, {}, {}, {}), geom)" \
                    .format(ev_time-t_threshold, ev_time, lat-lat_threshold, lon-lon_threshold, lat+lat_threshold, lon+lon_threshold)

                count.loc[index, 'ac_count_{}'.format(t_threshold)] = spark.sql(rectangular_query).count()

        # concatenate dataframes
        df_new = pd.concat([df_new, count], axis=1)

        spark.stop()
        return df_new

    else:
        spark = SparkSession.builder.appName("Terminal_Area_Flight_Data_Query") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()

        # load iff sector data
        df = spark.read.csv(iff_file_path, header=False, sep=",", schema=myschema)

        # select columns
        cols = ['recType', 'recTime', 'acId', 'lat', 'lon', 'alt']
        df = df.select(*cols).filter(df['recType'] == 3).withColumn("recTime", df['recTime'].cast(IntegerType()))

        count = pd.DataFrame(index=df_new.index, columns=['ac_count_' + str(t) for t in timestamp_threshold])
        for index, row in df_new.iterrows():
            for t_threshold in timestamp_threshold:
                ev_time = int(row['tMidnightSecs'] + row['tStart'])
                lat = row['Lat']
                lon = row['Lon']

                count.loc[index, 'ac_count_{}'.format(t_threshold)] = \
                        df.filter(df['recTime'] >= ev_time - t_threshold).\
                           filter(df['recTime'] < ev_time).\
                           filter(df['lat'] >= lat - lat_threshold).\
                           filter(df['lat'] <= lat + lat_threshold). \
                           filter(df['lon'] >= lon - lon_threshold).\
                           filter(df['lon'] <= lon + lon_threshold).count()
        # concatenate dataframes
        df_new = pd.concat([df_new, count], axis=1)
        return df_new


if __name__ == '__main__':

    # need to iterate date
    start_date = 20190801
    end_date = 20190831

    for date in range(start_date, end_date + 1):

        # load data
        ev_file_path = glob.glob("/media/ypang6/paralab/Research/data/EV_ZTL/EV_ZTL_{}*.csv".format(date))[0]
        df = pd.read_csv(ev_file_path)

        # remove useless columns
        cols = ['lKey', 'tMidnightSecs', 'tStart', 'tStop', 'AcId', 'AcType', 'tEv', 'EvType',
                'Lat', 'Lon', 'aEv', 'cEv', 'vEv', 'rEv', 'DTD', 'FlD', 'DDT', 'FlT', 'EvNumInfo']
        df = df[cols]

        # prediction among two class
        df_2class = df[(df['EvType'] == 'EV_LOOP') | (df['EvType'] == 'EV_GOA')]

        # set parameters
        lat_threshold = 2
        lon_threshold = 2
        timestamp_threshold = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        #timestamp_threshold = [10, 30]

        # find the event density feature
        df_2class = event_density_feature(df, df_2class, lat_threshold, lon_threshold, timestamp_threshold)

        # find the previous event happened to the same callsign
        df_2class = previous_event_sequence_feature(df, df_2class, timestamp_threshold)

        # save csv
        df_2class.to_csv('./processed_ev_data/2class_{}.csv'.format(date), encoding='utf-8', index=False)
        #df_2class.to_csv('./processed_ev_data/2class_{}_new.csv'.format(date), encoding='utf-8', index=False)

        # load preprocessed csv
        #df_2class = pd.read_csv('2class.csv', nrows=10)

        # # find the nearby aircraft feature
        sector_iff_file_path = glob.glob("/media/ypang6/paralab/Research/data/ZTL/IFF_ZTL_{}*.csv".format(date))[0]
        df_2class_ac = aircraft_density_feature(df_2class, sector_iff_file_path, lat_threshold, lon_threshold, timestamp_threshold, geospark=False)

        # save into csv
        df_2class_ac.to_csv('./processed_ev_data/2class_ac_{}.csv'.format(date), encoding='utf-8', index=False)
        #df_2class_ac.to_csv('./processed_ev_data/2class_ac_{}_new.csv'.format(date), encoding='utf-8', index=False)

        print('Done with {}'.format(date))
