import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pkg_resources.extern import packaging
from scipy.spatial import cKDTree
import glob
import os
import pyproj


def parse_version(v):
    try:
        return packaging.version.Version(v)
    except packaging.version.InvalidVersion:
        return packaging.version.LegacyVersion(v)


def read_iff_file(filename, record_types=3, callsigns=None, chunksize=50000, encoding='latin-1'):
    """
    Read IFF file and return data frames for requested record types

    From IFF 2.15 specification, record types include:
    2. header
    3. track point
    4. flight plan
    5. data source program
    6. sectorization
    7. minimum safe altitude
    8. flight progress
    9. aircraft state
    Parameters
    ----------
    filename : str
        File to read
    record_types : int, sequence of ints, or 'all'
        Record types to return
    callsigns : None, string, or list of strings
        If None, return records for all aircraft callsigns.
        Otherwise, only return records that match the given callsign
        (in the case of a single string) or match one of the specified
        callsigns (in the case of a list of strings).
    chunksize: int
        Number of rows that are read at a time by pd.read_csv.  This
        limits memory usage when working with large files, as we can
        extract out the desired rows from each chunk, intead of
        reading everything into one large DataFrame and then taking a
        subset.
    encoding: str
        Encoding argument passed on to open and pd.read_csv.  Using
        'latin-1' instead of the default will suppress errors that
        might otherwise occur with minor data corruption.  See
        http://python-notes.curiousefficiency.org/en/latest/python3/text_file_processing.html

    Returns
    -------
    DataFrame or dict of DataFrames
       If record_types is a scalar, return a DataFrame containing the
       data for that record type only.  Otherwise, return a dictionary
       mapping each requested record type to a corresponding DataFrame.
    """
    # Note default record_type of 3 (track point) is used for
    # consistency with the behavior of other functions that expect
    # flight tracking data

    # Determine file format version.  This is in record type 1, which
    # for now we assume to occur on the first line.
    with open(filename, 'r') as f:
        version = parse_version(f.readline().split(',')[2])

    # Columns for each record type, from version 2.6 specification.
    cols = {0: ['recType', 'comment'],
            1: ['recType', 'fileType', 'fileFormatVersion'],
            2: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat', 'acType',
                'Orig', 'Dest', 'opsType', 'estOrig', 'estDest'],
            3: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat', 'coord1',
                'coord2', 'alt', 'significance', 'coord1Accur', 'coord2Accur', 'altAccur', 'groundSpeed', 'course',
                'rateOfClimb', 'altQualifier', 'altIndicator', 'trackPtStatus', 'leaderDir', 'scratchPad',
                'msawInhibitInd', 'assignedAltString', 'controllingFac', 'controllingSeg', 'receivingFac',
                'receivingSec', 'activeContr', 'primaryContr', 'kybrdSubset', 'kybrdSymbol', 'adsCode', 'opsType',
                'airportCode'],
            4: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat', 'acType',
                'Orig', 'Dest', 'altcode', 'alt', 'maxAlt', 'assignedAltString', 'requestedAltString', 'route',
                'estTime', 'fltCat', 'perfCat', 'opsType', 'equipList', 'coordinationTime', 'coordinationTimeType',
                'leaderDir', 'scratchPad1', 'scratchPad2', 'fixPairScratchPad', 'prefDepArrRoute', 'prefDepRoute',
                'prefArrRoute'],
            5: ['recType', 'dataSource', 'programName', 'programVersion'],
            6: ['recType', 'recTime', 'Source', 'msgType', 'rectypeCat', 'sectorizationString'],
            7: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat', 'coord1',
                'coord2', 'alt', 'significance', 'coord1Accur', 'coord2Accur', 'altAccur', 'msawtype', 'msawTimeCat',
                'msawLocCat', 'msawMinSafeAlt', 'msawIndex1', 'msawIndex2', 'msawVolID'],
            8: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat', 'acType',
                'Orig', 'Dest', 'depTime', 'depTimeType', 'arrTime', 'arrTimeType'],
            9: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat', 'coord1',
                'coord2', 'alt', 'pitchAngle', 'trueHeading', 'rollAngle', 'trueAirSpeed', 'fltPhaseIndicator'],
            10: ['recType', 'recTime', 'fltKey', 'bcnCode', 'cid', 'Source', 'msgType', 'AcId', 'recTypeCat',
                 'configType', 'configSpec']}

    # For newer versions, additional columns are supported.  However,
    # this code could be commented out, and it should still be
    # compatible with newer versions, but just ignoring the additional
    # columns.
    if version >= parse_version('2.13'):
        cols[2] += ['modeSCode']
        cols[3] += ['trackNumber', 'tptReturnType', 'modeSCode']
        cols[4] += ['coordinationPoint', 'coordinationPointType', 'trackNumber', 'modeSCode']
    if version >= parse_version('2.15'):
        cols[3] += ['sensorTrackNumberList', 'spi', 'dvs', 'dupM3a', 'tid']

    # Determine the record type of each row
    with open(filename, 'r', encoding=encoding) as f:
        # An alternative, using less memory, would be to directly
        # create skiprows indices for a particular record type, using
        # a comprehension on enumerate(f); however, that would not
        # allow handling multiple record types.
        line_record_types = [int(line.split(',')[0]) for line in f]

    # Determine which record types to retrieve, and whether the result
    # should be a scalar or dict:
    if record_types == 'all':
        record_types = np.unique(line_record_types)
        scalar_result = False
    elif hasattr(record_types, '__getitem__'):
        scalar_result = False
    else:
        record_types = [record_types]
        scalar_result = True

    if callsigns is not None:
        callsigns = list(np.atleast_1d(callsigns))

    data_frames = dict()
    for record_type in record_types:
        # Construct list of rows to skip:
        skiprows = [i for i, lr in enumerate(line_record_types) if lr != record_type]

        # Passing usecols is necessary because for some records, the
        # actual data has extraneous empty columns at the end, in which
        # case the data does not seem to get read correctly without
        # usecols
        if callsigns is None:
            df = pd.concat((chunk for chunk in
                            pd.read_csv(filename, header=None, skiprows=skiprows, names=cols[record_type],
                                        usecols=cols[record_type], na_values='?', encoding=encoding,
                                        chunksize=chunksize, low_memory=False)), ignore_index=True)
        else:
            df = pd.concat((chunk[chunk['AcId'].isin(callsigns)] for chunk in
                            pd.read_csv(filename, header=None, skiprows=skiprows, names=cols[record_type],
                                        usecols=cols[record_type], na_values='?', encoding=encoding,
                                        chunksize=chunksize, low_memory=False)), ignore_index=True)

        # For consistency with other PARA-ATM data:
        df.rename(columns={'recTime': 'time',
                           'AcId': 'callsign',
                           'coord1': 'latitude',
                           'coord2': 'longitude',
                           'alt': 'altitude',
                           'rateOfClimb': 'rocd',
                           'groundSpeed': 'tas',
                           'course': 'heading'},
                  inplace=True)

        #         if 'time' in df:
        #             df['time'] = pd.to_datetime(df['time'], unit='s')
        if 'altitude' in df:
            df['altitude'] *= 100  # Convert 100s ft to ft

        # Store to dict of data frames
        data_frames[record_type] = df

    if scalar_result:
        result = data_frames[record_types[0]]
    else:
        result = data_frames

    return result


def kNN2DDens(xv, yv, resolution, neighbours, dim=2):
    """
    """
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


if __name__ == '__main__':
    date = 20190801
    t_start, t_end = 1564646400 + 3600*4, 1564646400 + 3600*8
    lat_min, lon_min, lat_max, lon_max = 32, -88, 38, -78
    interval = 60
    resolution = 256
    neighbours = [0, 4, 16, 32]
    smoothing = 16
    lat_atl, lon_atl = 33.64, -84.82
    print("KATL before projection: Lon: {} Lat: {}".format(lon_atl, lat_atl))

    timestamp = np.arange(t_start, t_end + 1, interval)
    lat = np.linspace(lat_min, lat_max, resolution).tolist()
    lon = np.linspace(lon_min, lon_max, resolution).tolist()

    file_path = glob.glob("/media/ypang6/paralab/Research/data/ZTL/IFF_ZTL_{}*.csv".format(date))[0]
    df = read_iff_file(file_path, record_types=3, chunksize=1e7)
    df = df[['time', 'latitude', 'longitude', 'callsign']]

    # # load arrival flight list
    # arr = pd.read_csv("acId_ATL_Arr.dat", sep="'", names=['d1', 'arr', 'd2']).drop(columns=['d1', 'd2']).arr.tolist()
    # df = df[df['callsign'].isin(arr)]
    #
    # # convert wgs84 to km with a reference point
    # inProj = pyproj.Proj(init='epsg:4326')  # WGS84
    # outProj = pyproj.Proj(init='epsg:2781')  # NAD83(HARN) / Georgia West
    # df['newLon'], df['newLat'] = pyproj.transform(inProj, outProj, df['longitude'].tolist(), df['latitude'].tolist())
    # [lon_min, lon_max], [lat_min, lat_max] = pyproj.transform(inProj, outProj, [lon_min, lon_max], [lat_min, lat_max])
    # lon_atl, lat_atl = pyproj.transform(inProj, outProj, lon_atl, lat_atl)  # KATL
    # print("KATL after projection: Lon: {} Lat: {}".format(lon_atl, lat_atl))

    directory = './{}/IFF'.format(date)
    if not os.path.exists(directory):
        os.makedirs(directory)

    frames = np.zeros(shape=(len(timestamp) - 1, resolution, resolution))
    #original_coord = np.zeros(shape=(len(timestamp) - 1, resolution, 2))
    for time_idx in range(len(timestamp) - 1):
        # temp_df = df.loc[(df['time'] >= timestamp[time_idx]) & (df['time'] <= timestamp[time_idx + 1]) &
        #                  (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
        #                  (df['longitude'] >= lon_min) & (df['longitude'] >= lon_max)]

        temp_df = df.loc[(df['time'] >= timestamp[time_idx]) & (df['time'] <= timestamp[time_idx + 1])]

        ys, xs = temp_df['latitude'].to_numpy(), temp_df['longitude'].to_numpy()
        #ys, xs = temp_df['newLat'].to_numpy(), temp_df['newLon'].to_numpy()

        #extent = [np.min(xs), np.max(xs), np.min(ys), np.max(ys)]
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

        plt.savefig('{}/IFF/{}.png'.format(date, timestamp[time_idx]), dpi=300, bbox_inches='tight')
        plt.close()

    # save frames into matlab readable file
    # from scipy.io import savemat
    # savemat("demo_file.mat", {"file": frames})

    # save frames into npy
    np.save(
        '{}/IFF/ac_density_{}_res_{}_tstart_{}_tend_{}_interval_{}_smoothing_{}.npy'
            .format(date, date, resolution, t_start, t_end, interval, smoothing), frames)

    # np.save(
    #     'ac_density_{}_res_{}_tstart_{}_tend_{}_interval_{}_raw.npy'
    #         .format(date, resolution, t_start, t_end, interval), original_coord)