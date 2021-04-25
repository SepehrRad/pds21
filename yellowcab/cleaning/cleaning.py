def get_date_components(df):
    df["start_month"] = df["tpep_pickup_datetime"].dt.month
    df["start_day"] = df["tpep_pickup_datetime"].dt.day
    df["start_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["end_month"] = df["tpep_dropoff_datetime"].dt.month
    df["end_day"] = df["tpep_dropoff_datetime"].dt.day
    df["end_hour"] = df["tpep_dropoff_datetime"].dt.hour
    return df


def is_weekend(df):
    # get day of week from pickup
    df["weekend"] = df["tpep_pickup_datetime"].dt.dayofweek > 4
    return df


def get_position(df):
    df["centers"] = df["geometry"].centroid
    return df


def get_duration(df):
    df["trip_duration_minutes"] = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    return df


def to_int(df):
    df["passenger_count"] = df["passenger_count"].astype("int")
    df["RatecodeID"] = df["RatecodeID"].astype("int")
    df["payment_type"] = df["payment_type"].astype("int")
    return df


def remove_outliers(df):
    # remove voyages
    voyages = df[df["passenger_count"] <= 0]
    df = df[~df.isin(voyages).all(1)]
    # remove trips with trip_distance = 0
    trips = df[df["trip_distance"] <= 0]
    df = df[~df.isin(trips).all(1)]
    # remove invalid ratecodes
    ratecodes = df[(df["RatecodeID"] > 6) | (df["RatecodeID"] < 1)]
    df = df[~df.isin(ratecodes).all(1)]
    # remove invalid PULocations (no geojson data for zones > 263 and unknown boroughs)
    PU = df[(df["PULocationID"] > 263) | (df["PULocationID"] < 1)]
    df = df[~df.isin(PU).all(1)]
    # remove invalid DOLocations (no geojson data for zones > 263 and unknown boroughs)
    DO = df[(df["DOLocationID"] > 263) | (df["DOLocationID"] < 1)]
    df = df[~df.isin(DO).all(1)]
    # remove invalid payments types
    payment = df[(df["payment_type"] > 6) | (df["payment_type"] < 1)]
    df = df[~df.isin(payment).all(1)]
    # remove trips with fare amount < 0
    fare = df[df["fare_amount"] < 0]
    df = df[~df.isin(fare).all(1)]
    # remove trips with extra < 0
    extra = df[df["extra"] < 0]
    df = df[~df.isin(extra).all(1)]
    # remove trips with mta_tax != 0 or != 0.5
    mta = df[(df["mta_tax"] != 0) & (df["mta_tax"] != 0.5)]
    df = df[~df.isin(mta).all(1)]
    # remove trips with tips < 0
    tip = df[df["tip_amount"] < 0]
    df = df[~df.isin(tip).all(1)]
    # remove trips with tolls < 0
    tolls = df[df["tolls_amount"] < 0]
    df = df[~df.isin(tolls).all(1)]
    # remove trips with improvement_surcharge != 0 or != 0.3
    surcharge = df[(df["improvement_surcharge"] != 0) & (df["improvement_surcharge"] != 0.3)]
    df = df[~df.isin(surcharge).all(1)]
    # remove trips with congestion_surcharge < 0
    congestion = df[df["congestion_surcharge"] < 0]
    df = df[~df.isin(congestion).all(1)]
    # remove trips with driving time larger than 8 hours or smaller than 0
    time = df[(df["trip_duration_minutes"].dt.seconds > 28800) | (df["trip_duration_minutes"].dt.seconds < 0)]
    df = df[~df.isin(time).all(1)]
    return df
