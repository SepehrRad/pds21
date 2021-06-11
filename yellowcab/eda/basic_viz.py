import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import yellowcab.eda.aggregation
from yellowcab.io import get_data_path


def basic_plots(df, borough=None):
    """
    This function creates a set of six bar (sub)plots (3x monthly, 3x weekly), inspecting number of trips,
    mean trip duration and mean passenger count. The parameter borough can be a assigned to a borough name of NYC to
    inspect trip data for this specific borough or it can be left blank to inspect the whole data frame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
        borough(String): Name of a borough to inspect. If not set, all NYC trips get selected.
    :returns

    """
    if borough is not None:
        nyc_zones_df = yellowcab.io.read_geo_dataset('taxi_zones.geojson')
        borough_zones = nyc_zones_df.loc[nyc_zones_df['borough'] == borough]
        borough_loc_ids = borough_zones['LocationID'].tolist()
        borough_loc_ids = map(str, borough_loc_ids)
        df = df.loc[(df['PULocationID'].isin(borough_loc_ids)) | (df['DOLocationID'].isin(borough_loc_ids))]

    df_agg_count_monthly = yellowcab.eda.agg_stats(df['pickup_datetime'].dt.month, df['pickup_month'], ['count'])
    df_agg_mean_duration_monthly = yellowcab.eda.agg_stats(df['pickup_datetime'].dt.month, df['trip_duration_minutes'],
                                                           ['mean'])
    df_agg_mean_passenger_mothly = yellowcab.eda.agg_stats(df['pickup_datetime'].dt.month, df['passenger_count'],
                                                           ['mean'])

    df_agg_count_weekly = yellowcab.eda.agg_stats(df['pickup_datetime'].dt.week, df['pickup_month'], ['count'])
    df_agg_mean_duration_weekly = yellowcab.eda.agg_stats(df['pickup_datetime'].dt.week, df['trip_duration_minutes'],
                                                          ['mean'])
    df_agg_mean_passenger_weekly = yellowcab.eda.agg_stats(df['pickup_datetime'].dt.week, df['passenger_count'],
                                                           ['mean'])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].bar(df_agg_count_monthly.index, df_agg_count_monthly['count_pickup_month'])
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Number of trips')

    axes[0, 1].bar(df_agg_mean_duration_monthly.index, df_agg_mean_duration_monthly['mean_trip_duration_minutes'])
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Avg trip duration')

    axes[0, 2].bar(df_agg_mean_passenger_mothly.index, df_agg_mean_passenger_mothly['mean_passenger_count'])
    axes[0, 2].set_xlabel('Month')
    axes[0, 2].set_ylabel('Avg passenger count')
    axes[0, 2].set_ylim(1, 3)

    axes[1, 0].bar(df_agg_count_weekly.index, df_agg_count_weekly['count_pickup_month'])
    axes[1, 0].set_xlabel('Week')
    axes[1, 0].set_ylabel('Number of trips')

    axes[1, 1].bar(df_agg_mean_duration_weekly.index, df_agg_mean_duration_weekly['mean_trip_duration_minutes'])
    axes[1, 1].set_xlabel('Week')
    axes[1, 1].set_ylabel('Avg trip duration')

    axes[1, 2].bar(df_agg_mean_passenger_weekly.index, df_agg_mean_passenger_weekly['mean_passenger_count'])
    axes[1, 2].set_xlabel('Week')
    axes[1, 2].set_ylabel('Avg passenger count')
    axes[1, 2].set_ylim(1, 3)

    plt.subplots_adjust(left=0.1, top=0.9)
    fig.tight_layout(pad=3.0)
    if borough is not None:
        title = 'Basic plots for {boroughname}'.format(boroughname=borough)
    else:
        title = 'Basic plots for NYC'
    fig.suptitle(t=title, fontsize=18, y=0.99)


def airport_plots(df):
    """
    This function creates a set of three bar (sub)plots, inspecting number of trips, mean trip duration and
    mean passenger count of trips beginning or terminating at one of the NYC airports.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns

    """
    df_airport_trips = df.loc[(df['PULocationID'].isin(['1', '132', '138'])) | (df['DOLocationID'].isin
                                                                                (['1', '132', '138']))]

    df_agg_count_airport = yellowcab.eda.agg_stats(df_airport_trips['pickup_datetime'].dt.week,
                                                   df_airport_trips['pickup_month'], ['count'])
    df_agg_mean_duration_airport = yellowcab.eda.agg_stats(df_airport_trips['pickup_datetime'].dt.week,
                                                           df_airport_trips['trip_duration_minutes'], ['mean'])
    df_agg_mean_passenger_airport = yellowcab.eda.agg_stats(df_airport_trips['pickup_datetime'].dt.week,
                                                            df_airport_trips['passenger_count'], ['mean'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(df_agg_count_airport.index, df_agg_count_airport['count_pickup_month'])
    axes[0].set_xlabel('Week')
    axes[0].set_ylabel('Trips to airports')

    axes[1].bar(df_agg_mean_duration_airport.index, df_agg_mean_duration_airport['mean_trip_duration_minutes'])
    axes[1].set_xlabel('Week')
    axes[1].set_ylabel('Mean trip duration')

    axes[2].bar(df_agg_mean_passenger_airport.index, df_agg_mean_passenger_airport['mean_passenger_count'])
    axes[2].set_xlabel('Week')
    axes[2].set_ylabel('Mean passenger count')

    plt.subplots_adjust(left=0.1, top=0.9)
    fig.tight_layout(pad=3.0)
    title = "Basic plots for airport trips"
    fig.suptitle(t=title, fontsize=18)
