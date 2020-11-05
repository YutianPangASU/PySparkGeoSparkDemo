# spark configuration
import os
os.environ["SPARK_HOME"] = '/home/ypang6/anaconda3/lib/python3.7/site-packages/pyspark'
os.environ["PYTHONPATH"] = '/home/ypang6/anaconda3/bin/python3.7'
os.environ['PYSPARK_PYTHON'] = '/home/ypang6/anaconda3/bin/python3.7'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/ypang6/anaconda3/bin/python3.7'

import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm


def ac_density_map(iff_file_path, lat, lon, time, geospark_rdd=False, geospark_sql=False):
    import threading
    from pyspark.sql import SparkSession
    from geospark.register import GeoSparkRegistrator
    from geospark.core.formatMapper.shapefileParser import ShapefileReader
    from pyspark.sql.types import StructType, StructField, ShortType, StringType, LongType, IntegerType, DoubleType
    from geospark.utils.adapter import Adapter
    from geospark.core.geom.envelope import Envelope
    from geospark.core.enums import IndexType
    from geospark.core.spatialOperator import RangeQuery

    path_to_jars = "/home/ypang6/anaconda3/lib/python3.7/site-packages/pyspark/jars/"
    jars = ["geospark-sql_2.3-1.3.0.jar", "geospark-1.3.0.jar", "geo_wrapper_2.11-0.3.0.jar"]

    jars_string = ",".join([os.path.join(path_to_jars, el) for el in jars])

    spark = SparkSession.builder.appName("Terminal_Area_Flight_Data_Query") \
        .config("spark.jars", ",".join([os.path.join(path_to_jars, el) for el in jars])) \
        .getOrCreate()

    GeoSparkRegistrator.registerAll(spark)

    myschema = StructType([
        StructField("recType", ShortType(), True),  #1  //track point record type number
        StructField("recTime", StringType(), True),  #2  //seconds since midnigght 1/1/70 UTC
        StructField("fltKey", LongType(), True),  #3  //flight key
        StructField("bcnCode", IntegerType(), True),  #4  //digit range from 0 to 7
        StructField("cid", IntegerType(), True),  #5  //computer flight id
        StructField("Source", StringType(), True),  #6  //source of the record
        StructField("msgType", StringType(), True),  #7
        StructField("acId", StringType(), True),  #8  //call sign
        StructField("recTypeCat", StringType(), True),  #9
        StructField("lat", DoubleType(), True),  #10
        StructField("lon", DoubleType(), True),  #11
        StructField("alt", DoubleType(), True),  #12  //in 100s of feet
        StructField("significance", ShortType(), True),  #13 //digit range from 1 to 10
        StructField("latAcc", DoubleType(), True),  #14
        StructField("lonAcc", DoubleType(), True),  #15
        StructField("altAcc", DoubleType(), True),  #16
        StructField("groundSpeed", IntegerType(), True),  #17 //in knots
        StructField("course", DoubleType(), True),  #18  //in degrees from true north
        StructField("rateOfClimb", DoubleType(), True),  #19  //in feet per minute
        StructField("altQualifier", StringType(), True),  #20  //Altitude qualifier (the “B4 character”)
        StructField("altIndicator", StringType(), True),  #21  //Altitude indicator (the “C4 character”)
        StructField("trackPtStatus", StringType(), True),  #22  //Track point status (e.g., ‘C’ for coast)
        StructField("leaderDir", IntegerType(), True),  #23  //int 0-8 representing the direction of the leader line
        StructField("scratchPad", StringType(), True),  #24
        StructField("msawInhibitInd", ShortType(), True),  #25 // MSAW Inhibit Indicator (0=not inhibited, 1=inhibited)
        StructField("assignedAltString", StringType(), True),  #26
        StructField("controllingFac", StringType(), True),  #27
        StructField("controllingSec", StringType(), True),  #28
        StructField("receivingFac", StringType(), True),  #29
        StructField("receivingSec", StringType(), True),  #30
        StructField("activeContr", IntegerType(), True),  #31  // the active control number
        StructField("primaryContr", IntegerType(), True),  #32  //The primary(previous, controlling, or possible next)controller number
        StructField("kybrdSubset", StringType(), True),  #33  //identifies a subset of controller keyboards
        StructField("kybrdSymbol", StringType(), True),  #34  //identifies a keyboard within the keyboard subsets
        StructField("adsCode", IntegerType(), True),  #35  //arrival departure status code
        StructField("opsType", StringType(), True),  #36  //Operations type (O/E/A/D/I/U)from ARTS and ARTS 3A data
        StructField("airportCode", StringType(), True),  #37
        StructField("trackNumber", IntegerType(), True),  #38
        StructField("tptReturnType", StringType(), True),  #39
        StructField("modeSCode", StringType(), True),  #40
        StructField("sensorTrackNumberList", StringType(), True), #41 //a list of sensor/track number combinations
        StructField("spi", StringType(), True),  #42 // representing the Ident feature
        StructField("dvs", StringType(), True), #43 // indicate the aircraft is within a suppresion volumn area
        StructField("dupM3a", StringType(), True),  #44 // indicate 2 aircraft have the same mode 3a code
        StructField("tid", StringType(), True),  #45 //Aircraft Ident entered by pilot
    ])

    # load iff sector data
    df = spark.read.csv(iff_file_path, header=False, sep=",", schema=myschema)

    # select columns
    cols = ['recType', 'recTime', 'acId', 'lat', 'lon', 'alt']
    df = df.select(*cols).filter(df['recType'] == 3).withColumn("recTime", df['recTime'].cast(IntegerType()))

    # register pyspark df in SQL
    df.registerTempTable("pointtable")

    if time[1]-time[0] == 1:  # timeframe 1 second interval
        frames = np.zeros(shape=(len(time), len(lat) - 1, len(lon) - 1))
        for index_t in range(len(time)):
            # create shape column in geospark
            spatialdf = spark.sql(
                "SELECT ST_Point(CAST(lat AS Decimal(24, 20)), CAST(lon AS Decimal(24, 20))) AS geom, recTime, acId " \
                "FROM pointtable WHERE recTime={}".format(time[index_t]))

            spatialdf.createOrReplaceTempView("spatialdf")

            # register pyspark spatialdf in SQL
            spatialdf.registerTempTable("spatialdf")

            if geospark_sql:
                for index_lat in range(len(lat) - 1):
                    for index_lon in range(len(lon) - 1):
                        rectangular_query = "SELECT COUNT(acId) " \
                                            "FROM spatialdf " \
                                            "WHERE ST_Contains(ST_PolygonFromEnvelope({}, {}, {}, {}), geom)" \
                            .format(lat[index_lat], lon[index_lon], lat[index_lat + 1], lon[index_lon + 1])

                        frames[index_t, index_lat, index_lon] = spark.sql(rectangular_query).first()[0]

            if geospark_rdd:
                spatial_rdd = Adapter.toSpatialRdd(spatialdf, "geom")
                consider_boundary_intersection = False  # Only return gemeotries fully covered by the window
                build_on_spatial_partitioned_rdd = False  # Set to TRUE only if run join query
                using_index = True

                # build spatial Quadtree index
                spatial_rdd.buildIndex(IndexType.QUADTREE, build_on_spatial_partitioned_rdd)

                # using the geospark rdd module for range query
                for index_lat in range(len(lat) - 1):
                    for index_lon in range(len(lon) - 1):
                        range_query_window = Envelope(lon[index_lon], lon[index_lon + 1], lat[index_lat],
                                                      lat[index_lat + 1])
                        frames[index_t, index_lat, index_lon] = RangeQuery.SpatialRangeQuery(spatial_rdd,
                                                                                             range_query_window,
                                                                                             consider_boundary_intersection,
                                                                                             using_index).count()
        return frames

    else:  # timeframe interval greater than 1s
        frames = np.zeros(shape=(len(time)-1, len(lat)-1, len(lon)-1))
        for index_t in range(len(time)-1):
            # create shape column in geospark
            spatialdf = spark.sql(
                "SELECT ST_Point(CAST(lat AS Decimal(24, 20)), CAST(lon AS Decimal(24, 20))) AS geom, recTime, acId " \
                "FROM pointtable " \
                "WHERE recTime>={} AND recTime<={}" \
                .format(time[index_t], time[index_t+1]))

            #spatialdf = spatialdf.dropDuplicates(['acId'])  # get distinct acId

            spatialdf.createOrReplaceTempView("spatialdf")

            # register pyspark spatialdf in SQL
            spatialdf.registerTempTable("spatialdf")

            if geospark_sql:
                for index_lat in range(len(lat)-1):
                    for index_lon in range(len(lon)-1):
                        rectangular_query = "SELECT COUNT(acId) " \
                                            "FROM spatialdf " \
                                            "WHERE ST_Contains(ST_PolygonFromEnvelope({}, {}, {}, {}), geom)" \
                            .format(lat[index_lat], lon[index_lon], lat[index_lat+1], lon[index_lon+1])

                        frames[index_t, index_lat, index_lon] = spark.sql(rectangular_query).first()[0]

            if geospark_rdd:
                spatial_rdd = Adapter.toSpatialRdd(spatialdf, "geom")
                consider_boundary_intersection = False  # Only return gemeotries fully covered by the window
                build_on_spatial_partitioned_rdd = False  # Set to TRUE only if run join query
                using_index = True

                # build spatial Quadtree index
                spatial_rdd.buildIndex(IndexType.QUADTREE, build_on_spatial_partitioned_rdd)

                # using the geospark rdd module for range query
                for index_lat in range(len(lat)-1):
                    for index_lon in range(len(lon)-1):
                        range_query_window = Envelope(lon[index_lon], lon[index_lon+1], lat[index_lat], lat[index_lat+1])
                        frames[index_t, index_lat, index_lon] = RangeQuery.SpatialRangeQuery(spatial_rdd,
                                                                                             range_query_window,
                                                                                             consider_boundary_intersection,
                                                                                             using_index).count()
            return frames


def iff_density_map_threading_helper():
    """
    Threading helper for fast data processing on iff dataset
    :return:
    """
    pass


def ev_density_map_threading_helper():
    pass


def kNN2DDens(xv, yv, resolution, neighbours, dim=2):
    """
    """
    from scipy.spatial import cKDTree
    # Create the tree
    tree = cKDTree(np.array([xv, yv]).T)
    # Find the closest nnmax-1 neighbors (first entry is the point itself)
    grid = np.mgrid[0:resolution, 0:resolution].T.reshape(resolution**2, dim)
    dists = tree.query(grid, neighbours)
    # Inverse of the sum of distances to each grid point.
    inv_sum_dists = 1. / dists[0].sum(1)

    # Reshape
    im = inv_sum_dists.reshape(resolution, resolution)
    return im


def data_coord2view_coord(p, resolution, pmin, pmax):
    dp = pmax - pmin
    dv = (p - pmin) / dp * resolution
    return dv


def ev_density_map(ev_file_path, lat, lon, time, neighbours, smoothing):
    df = pd.read_csv(ev_file_path)

    # remove useless columns
    cols = ['tMidnightSecs', 'AcId', 'tEv', 'EvType', 'Lat', 'Lon']
    df = df[cols]

    # two class
    #df = df[(df['EvType'] == 'EV_LOOP') | (df['EvType'] == 'EV_GOA')]
    df["tEv"] = df["tEv"] + df["tMidnightSecs"]
    df["tEv"] = df[["tEv"]].astype(int)
    df = df.drop(['tMidnightSecs', 'EvType'], axis=1)

    if time[1]-time[0] == 1:  # timeframe 1 second interval
        frames = np.zeros(shape=(len(time), len(lat) - 1, len(lon) - 1))
        for index_t in range(len(time)):
            df_temp = df.loc[df['tEv'] == time[index_t]]

            print("Implementation not finished.")
    else:  # greater than 1 second interval
        frames = np.zeros(shape=(len(time)-1, resolution, resolution))
        #original_coord = np.zeros(shape=(len(time) - 1, resolution, 2))

        for time_idx in range(len(time)-1):
            temp_df = df.loc[(df['tEv'] >= time[time_idx]) & (df['tEv'] <= time[time_idx+1])]

            ys, xs = temp_df['Lat'].to_numpy(), temp_df['Lon'].to_numpy()
            extent = [lon_min, lon_max, lat_min, lat_max]
            xv = data_coord2view_coord(xs, resolution, extent[0], extent[1])
            yv = data_coord2view_coord(ys, resolution, extent[2], extent[3])

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            for ax, neighbour in zip(axes.flatten(), neighbours):

                if neighbour == 0:
                    #original_coord[time_idx, :, :] = np.concatenate((np.expand_dims(ys, axis=1), np.expand_dims(xs, axis=1)), axis=1)  # save ys and xs
                    ax.plot(xs, ys, 'k.', markersize=5)
                    ax.set_aspect('equal')
                    ax.set_title("Scatter Plot")
                    ax.set_xlim(extent[0], extent[1])
                    ax.set_ylim(extent[2], extent[3])
                else:
                    im = kNN2DDens(xv, yv, resolution, neighbour)
                    if neighbour == smoothing:
                        frames[time_idx, :, :] = im
                    ax.imshow(im, origin='lower', extent=extent, cmap=cm.Blues)
                    ax.set_title("Smoothing over %d neighbours" % neighbour)
                    ax.set_xlim(extent[0], extent[1])
                    ax.set_ylim(extent[2], extent[3])

            plt.savefig('{}/EV/{}.png'.format(date, timestamp[time_idx]), dpi=300, bbox_inches='tight')
            plt.close()

        # save frames into npy
        np.save(
            '{}/EV/ev_density_{}_res_{}_tstart_{}_tend_{}_interval_{}_smoothing_{}.npy'
                .format(date, date, resolution, t_start, t_end, interval, smoothing), frames)

pass


def save_frames(frames, timestamp, lat, lon):

    for fig_num in range(len(timestamp)-1):
        print(np.max(frames[fig_num]))
        pass
    pass


if __name__ == '__main__':
    # Threading for parallel computing
    None

    # set parameters
    date = 20190801
    t_start, t_end = 1564646400 + 3600*4, 1564646400 + 3600*8
    lat_min, lon_min, lat_max, lon_max = 32, -88, 38, -78
    interval = 60
    resolution = 128
    neighbours = [0, 4, 16, 32]
    smoothing = 16

    timestamp = np.arange(t_start, t_end + 1, interval)
    lat = np.linspace(lat_min, lat_max, resolution).tolist()
    lon = np.linspace(lon_min, lon_max, resolution).tolist()

    # load file path
    iff_file_path = glob.glob("/media/ypang6/paralab/Research/data/ZTL/IFF_ZTL_{}*.csv".format(date))[0]
    ev_file_path = glob.glob("/media/ypang6/paralab/Research/data/EV_ZTL/EV_ZTL_{}*.csv".format(date))[0]

    directory = './{}/EV'.format(date)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # process the ev density map
    ev_density_map(ev_file_path, lat, lon, timestamp, neighbours, smoothing)

    # process the ac density map
    #ac_map = ac_density_map(iff_file_path, lat, lon, timestamp, geospark_sql=True)
    #ac_map = ac_density_map(iff_file_path, lat, lon, timestamp, geospark_rdd=True)
    #np.save('ac_density_{}_res_{}_tstart_{}_tend_{}_interval_{}.npy'.format(date, resolution-1, t_start, t_end, interval), ac_map)



