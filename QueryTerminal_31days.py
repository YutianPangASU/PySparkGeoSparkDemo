# Set spark environments
import os
# os.environ["SPARK_HOME"] = '/home/ypang6/anaconda3/lib/python3.7/site-packages/pyspark'
# os.environ["PYTHONPATH"] = '/home/ypang6/anaconda3/bin/python3.7'
# os.environ['PYSPARK_PYTHON'] = '/home/ypang6/anaconda3/bin/python3.7'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/ypang6/anaconda3/bin/python3.7'

from pyspark.sql import SparkSession
from pyspark.sql.types import *
import glob
import pandas as pd

spark = SparkSession \
        .builder \
        .appName("Terminal_Area_Flight_Data_Query") \
        .config("spark.ui.port", "4041") \
        .getOrCreate()

myschema = StructType([
    StructField("recType", ShortType(), True),  #1  //track point record type number
    StructField("recTime", StringType(), True),  #2  //seconds since midnigght 1/1/70 UTC
    StructField("fltKey", LongType(), True),  #3  //flight key
    StructField("bcnCode", IntegerType(), True),  #4  //digit range from 0 to 7
    StructField("cid", IntegerType(), True),  #5  //computer flight id
    StructField("Source", StringType(), True),  #6  //source of the record
    StructField("msgType", StringType(), True),  #7
    StructField("acId", StringType(), True),  #8  //call sign
    StructField("recTypeCat", IntegerType(), True),  #9
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
    StructField("modeSCode", StringType(), True)  #40
])


def take_arrival_flights(date, point):
    # load csv into dataframe
    file_path = glob.glob("/media/ypang6/paralab/Research/data/ATL/IFF_ATL+ASDEX_{}*.csv".format(date))[0]
    df = spark.read.csv(file_path, header=False, sep=",", schema=myschema)
    cols = ['recType', 'recTime', 'acId', 'lat', 'lon', 'alt']
    df = df.select(*cols).filter(df['recType']==3).withColumn("recTime", df['recTime'].cast(IntegerType()))

    # seperate arrival flights and departure flights
    cs_dep = []
    cs_arr = []
    cs_unknown = []
    for x in df.select('acId').distinct().collect():
        temp_df = df.filter(df['acId'] == x['acId'])
        if temp_df.select(['alt']).take(1)[0][0] == 10.06:
            cs_dep.append(x['acId'])
        elif temp_df.orderBy(temp_df.recTime.desc()).select('alt').take(1)[0][0] == 10.06:
            cs_arr.append(x['acId'])
        else:
            cs_unknown.append(x['acId'])

    # create arrival flights dataframe
    df_arr = df.filter(df.acId.isin(cs_arr) == True)

    # filter points close to point and save into csv
    faf9rflight = df_arr.filter(df_arr['lat']>=point[0]-radius).filter(df_arr['lat']<=point[0]+radius).\
    filter(df_arr['lon']>=point[1]-radius).filter(df_arr['lon']<=point[1]+radius)
    faf9rflight.coalesce(1).write.csv('./faf9rflights/{}'.format(date))


if __name__ == '__main__':

    timestamp = 1567346400  # 2PM
    FAF_9L = (33.63465, -84.54984166666667)  # waypoint NIVII (FAF of KATL runway 9L)
    FAF_9R = (33.63172777777777, -84.54940555555555)  # waypoint BURNY (FAF of KATL runway 9R)
    IF_9R = (33.631397222222226, -84.71883611111112)  # waypoint GGUYY (IF of KATL runway 9R)
    IAF_9L = (33.63394722222222, -84.86316388888888)  # waypoint RYENN (IAF of KATL runway 9L)
    IAF_9R = (33.63093611111111, -84.86295)  # waypoint ANDIY (IAF of KATL runway 9R)
    IF_27R = (33.63430555555556, -84.12904722222221)  # waypoint MAASN (IF of KATL runway 27R)
    IAF_27R = (33.633874999999996, -83.99111666666667)  # waypoint YOUYU (IAF of KATL runway 27R)
    radius = 0.001

    start_date = 20190801
    end_date = 20190831

    for date in range(start_date, end_date+1):
        take_arrival_flights(date, FAF_9R)
        print("finish processing date {}".format(date))

