from datetime import date, datetime

import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


def add_relevant_features(data_set, date_column):
    """
    This function adds all in this module created columns
    to the input dataset, based on the passed date_column in case it is based on time.
    ----------------------------------------------
    :param
        data_set (pd.DataFrame): Dataframe to what the columns should be added.
        date_column (datetime): The column we use for comparing the dates.
    :return:
        pd.DataFrame: The passed dataframe with the new columns added.
    """
    data_set = create_holiday_column(data_set, date_column)
    data_set = create_season_column(data_set, date_column)
    data_set = create_covid_relevant_features(data_set, date_column)
    data_set = _manhattan_dist_vectorized(data_set)

    data_set["bearing_distance"] = _bearing_dist_vectorized(
        data_set["centers_lat_pickup"],
        data_set["centers_long_pickup"],
        data_set["centers_lat_dropoff"],
        data_set["centers_long_dropoff"],
    )

    data_set["haversine_distance"] = _haversine_dist_vectorized(
        data_set["centers_lat_pickup"],
        data_set["centers_long_pickup"],
        data_set["centers_lat_dropoff"],
        data_set["centers_long_dropoff"],
    )

    data_set = _get_weekday(data_set)

    return data_set


def _get_weekday(df):
    """
    This function adds a column containing the weekday of each entry to the passed dataframe.
    ----------------------------------------------
    :param
            df (pandas.DataFrame): Dataframe, where a column for the weekday should be added to.
    :return: pandas.DataFrame including a column for weekday.
    """
    df["weekday"] = df["pickup_datetime"].dt.dayofweek
    return df


def _haversine_dist_vectorized(lon1, lat1, lon2, lat2):
    """
    This function calculates the haversine distance of two passed vectors of points.
    ----------------------------------------------
    :param
           lon1 (float): The longitude of the first point.
           lat1 (float): The latitude of the first point.
           lon2 (float): The longitude of the second point.
           lat2 (float): The latitude of the second point.
    :return: float: Haversine distance of the two passed vectors of points.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = np.sin((lat2 - lat1) / 2.0) ** 2 + (
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    distance = 6371 * 2 * np.arcsin(np.sqrt(a))
    return distance


def _bearing_dist_vectorized(lon1, lat1, lon2, lat2):
    """
    This function calculates the bearing distance of two passed vectors of points.
    ----------------------------------------------
    :param
           lon1 (float): The longitude of the first point.
           lat1 (float): The latitude of the first point.
           lon2 (float): The longitude of the second point.
           lat2 (float): The latitude of the second point.
    :return: float: Bearing distance of the two passed vectors of points.
    """
    lon_diff = lon2 - lon1
    y = np.sin(lon_diff) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon_diff)
    theta = np.arctan2(y, x)
    bearing = (theta * 180 / np.pi + 360) % 360  # in degrees
    return bearing


def _manhattan_dist_vectorized(df):
    """
    This function calculates the manhattan distance of two passed vectors of points.
    ----------------------------------------------
    :param
           df (pandas.DataFrame): Dataframe, where a column for the manhattan distance should be added to.
    :return: pandas.DataFrame including a column for manhattan distance.
    """
    distance_a = _haversine_dist_vectorized(
        lat1=df["centers_lat_pickup"],
        lon1=df["centers_long_pickup"],
        lat2=df["centers_lat_pickup"],
        lon2=df["centers_long_dropoff"],
    )
    distance_b = _haversine_dist_vectorized(
        lat1=df["centers_lat_pickup"],
        lon1=df["centers_long_pickup"],
        lat2=df["centers_lat_dropoff"],
        lon2=df["centers_long_pickup"],
    )
    df["manhattan_distance"] = distance_a + distance_b
    return df


def _get_season_in_ny(date_time):
    """
    This function returns the season which the passed date belongs to.
    ----------------------------------------------
    :param
        date_time (datetime): The date we want to get the season for.
    :returns
        String: Name of the season.
    """
    seasons = [
        ("winter", (date(2020, 1, 1), date(2020, 3, 18))),
        ("spring", (date(2020, 3, 19), date(2020, 6, 19))),
        ("summer", (date(2020, 6, 20), date(2020, 9, 20))),
        ("autumn", (date(2020, 9, 21), date(2020, 12, 20))),
        ("winter", (date(2020, 12, 21), date(2020, 12, 31))),
    ]
    if isinstance(date_time, datetime):
        date_time = date_time.date()
    date_time = date_time.replace(year=2020)
    try:
        return next(
            season for season, (start, end) in seasons if start <= date_time <= end
        )
    except StopIteration:
        pass


def create_season_column(data_set, date_column):
    """
    This function adds a column stating if the date handed over is a holiday day in the US.
    ----------------------------------------------
    :param
        data_set (pandas.DataFrame): Dataframe to what the column should be added.
        date_column (datetime): The column we use for comparing the dates.

    :returns
        pandas.DataFrame: The input dataframe with a season column added.
    """
    data_set["Season"] = data_set[date_column].apply(_get_season_in_ny)
    return data_set


def create_holiday_column(data_set, date_column):
    """
    This function adds a column saying if the date is a holiday day in NY.
    ----------------------------------------------
    :param
        data_set (pandas.DataFrame): Dataframe to what the column should be added.
        date_column (datetime): The column we use for comparing the dates.

    :returns
        pandas.DataFrame: The input dataframe with a holiday day column added.
    """
    cal = calendar()
    holidays = cal.holidays(data_set[date_column].min(), data_set[date_column].max())
    data_set["Holiday"] = (
        data_set[date_column].dt.date.astype("datetime64").isin(holidays)
    )
    return data_set


def get_covid_restrictions():
    """
    Information for first covid case:
    https://www.investopedia.com/historical-timeline-of-covid-19-in-new-york-city-5071986
    https://news.google.com/covid19/map?hl=de&mid=%2Fm%2F02_286&gl=DE&ceid=DE%3Ade
    This function returns relevant covid restriction time ranges.
    Information for lockdown time-range:
    https://ny.eater.com/2020/12/30/22203053/nyc-coronavirus-timeline-restaurants-bars-2020
    Information for school restrictions time-range:
    https://ballotpedia.org/School_responses_in_New_York_to_the_coronavirus_(COVID-19)_pandemic_during_the_2020-2021_school_year
    https://mommypoppins.com/new-york-city-kids/schools/heres-the-nyc-public-school-calendar-for-2020-2021
    ----------------------------------------------
    :return
        dictionary: A dictionary including the timespans for covid restrictions each.
    """
    covid_restrictions = {
        "covid_new_cases_start": datetime(day=1, month=3, year=2020),
        "covid_new_cases_end": datetime(day=31, month=12, year=2020),
        "covid_lockdown_start": datetime(day=16, month=3, year=2020),
        "covid_lockdown_end": datetime(day=8, month=6, year=2020),
        "covid_school_restrictions_start": datetime(day=16, month=3, year=2020),
        "covid_school_restrictions_end": datetime(day=20, month=3, year=2020),
    }
    return covid_restrictions


def create_covid_relevant_features(data_set, date_column):
    """
    This function adds a column adds column for relevant covid restrictions matching the dates.
    ----------------------------------------------
    :param
        data_set (pandas.DataFrame): Dataframe to what the columns should be added.
        date_column (datetime): The column we use for comparing the dates.
    :return
        pandas.DataFrame: The passed dataframe with covid restrictions columns added.
    """
    covid_restrictions = get_covid_restrictions()
    data_set.loc[
        data_set[date_column].between(
            covid_restrictions.get("covid_new_cases_start"),
            covid_restrictions.get("covid_new_cases_end"),
        ),
        "covid_new_cases",
    ] = 1
    data_set.loc[
        data_set[date_column].between(
            covid_restrictions.get("covid_lockdown_start"),
            covid_restrictions.get("covid_lockdown_end"),
        ),
        "covid_lockdown",
    ] = 1
    data_set.loc[
        data_set[date_column].between(
            covid_restrictions.get("covid_school_restrictions_start"),
            covid_restrictions.get("covid_school_restrictions_end"),
        ),
        "covid_school_restrictions",
    ] = 1

    data_set["covid_new_cases"].fillna(0, inplace=True)
    data_set["covid_lockdown"].fillna(0, inplace=True)
    data_set["covid_school_restrictions"].fillna(0, inplace=True)

    return data_set
