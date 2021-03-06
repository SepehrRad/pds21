import os

import pandas as pd


def get_data_path():
    """
    This function finds the data directory of the project.
    ----------------------------------------------
    :returns
        path(String): The path to data directory
    """

    if os.path.isdir(os.path.join(os.getcwd(), "data")):
        path = os.path.join(os.getcwd(), "data")
    elif os.path.isdir(os.path.join(os.getcwd(), "..", "data")):
        path = os.path.join(os.getcwd(), "..", "data")
    else:
        raise FileNotFoundError
    return path


def get_zone_information(
    df,
    zone_file,
    aspect=None,
    base_path=get_data_path(),
    relative_path="input/taxi_zones",
):
    """
    This function adds zone information to the given data.
    ----------------------------------------------
    :param
        df(pd.DataFrame): The zone informaton will be added to this data frame.
        zone_file(String): The name of the zone csv file.
        aspect(String): Adds zone information based on given aspect. Allowed values are pickup or dropoff.
                        Default-> zone information is added for both aspects.
        base_path(String): Path to data directory. Defaults to wd/data.
        relative_path(String): Path to directory with the zone file. Defaults to input/taxi_zones.
    :returns
        df(pd.DataFrame): Data with zone information
    :raises
        ValueError: if the zone file can't be located.
    """
    zone_path = os.path.join(base_path, relative_path, zone_file)
    try:
        nyc_zones = pd.read_csv(zone_path)
        nyc_zones.LocationID = nyc_zones.LocationID.astype("str")
    except FileNotFoundError:
        raise ValueError(
            f"No file with name of {zone_file} could be found in {os.path.join(base_path, relative_path)}"
        )
    if aspect is None:
        df = df.merge(
            nyc_zones,
            how="left",
            left_on="PULocationID",
            right_on="LocationID",
            validate="m:1",
        )
        df = df.merge(
            nyc_zones,
            how="left",
            left_on="DOLocationID",
            right_on="LocationID",
            validate="m:1",
            suffixes=("_pickup", "_dropoff"),
        )
    else:
        if aspect == "pickup":
            df = df.merge(
                nyc_zones,
                how="left",
                left_on="PULocationID",
                right_on="LocationID",
                validate="m:1",
            )
        else:
            df = df.merge(
                nyc_zones,
                how="left",
                left_on="DOLocationID",
                right_on="LocationID",
                validate="m:1",
            )
    return df


def get_random_state():
    """
    This function defines our random state, so that it's the same in
    every application
    ----------------------------------------------
    :return:
        int: the integer used for random state parameter
    """
    RANDOM_STATE = 7
    return RANDOM_STATE


def flatten_list(not_flattened_list):
    """
    This function flattens an passed list.
    :param not_flattened_list (list): List, which is not flattened yet.
    :returns list: Flattened list.
    """
    list_flattened = []
    for element in not_flattened_list:
        if isinstance(element, list):
            list_flattened.extend(element)
        else:
            list_flattened.append(element)
    return list_flattened
