import calendar
import datetime
from collections import defaultdict
from time import strptime

import branca.colormap
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import seaborn as sns
from folium.plugins import HeatMap, HeatMapWithTime
from matplotlib.figure import Figure
from matplotlib.ticker import Formatter
from numpy import random
from panel.interact import fixed, interact
from plotly import express as px

import yellowcab
from yellowcab.io.input import read_geo_dataset
from yellowcab.io.utils import get_zone_information


def create_animated_monthly_plot(df, aspect="pickup"):
    """
    This function creates an animated plotly express plot based on different aspects of the given data.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the animated plot.
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff.
    :return:
        plotly.scatter_mapbox: The animated scatter_mapbox
    """
    data = _create_aggregator(df, aspect=aspect, animated=True)
    data = get_zone_information(data, aspect=aspect, zone_file="taxi_zones.csv")
    data = data.sort_values(by=f"{aspect}_month")
    fig = px.scatter_mapbox(
        data,
        lat=f"centers_lat_{aspect}",
        lon=f"centers_long_{aspect}",
        size=f"{aspect}_count",
        color=f"{aspect}_count",
        hover_name="Zone",
        animation_frame=f"{aspect}_month",
        color_continuous_scale="inferno",
        height=700,
        width=850,
        zoom=10,
    )
    fig = fig.update_layout(mapbox_style="carto-positron")
    fig = fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig = fig.update_traces(marker=dict(sizemin=1))
    return fig


def _create_plotly_monthly_plot(
        df, map_style="carto-positron", month=1, aspect="pickup", cmap="inferno"
):
    """
    This function creates a plotly express plot based on different aspects of the given data.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the animated plot.
        map_style(String): Tile layer style of the choropleth map.
        month(int): Used to show the data for this month only.
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff.
        cmap(String): The chosen colormap.
    :return:
        plotly.scatter_mapbox: The created scatter_mapbox
    """
    data = _create_aggregator(df, aspect=aspect, animated=True)
    data = get_zone_information(data, aspect=aspect, zone_file="taxi_zones.csv")
    data = data.sort_values(by=f"{aspect}_month")
    data = data.loc[data[f"{aspect}_month"] == month]
    fig = px.scatter_mapbox(
        data,
        lat=f"centers_lat_{aspect}",
        lon=f"centers_long_{aspect}",
        size=f"{aspect}_count",
        color=f"{aspect}_count",
        hover_name="Zone",
        color_continuous_scale=cmap,
        height=700,
        width=1200,
        zoom=10,
    )
    fig = fig.update_layout(mapbox_style=map_style)
    fig = fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig = fig.update_traces(marker=dict(sizemin=1))
    return fig


def _create_monthly_animated_tab(df):
    """
    This function creates a plotly express tab with interactive widgets.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
    :return:
        pn.Column: the created plotly express animated panel element
    """
    mapbox_tiles = [
        "carto-positron",
        "carto-darkmatter",
        "stamen-terrain",
        "stamen-toner",
        "open-street-map",
    ]
    cmap = [
        "viridis",
        "inferno",
        "balance",
        "icefire",
        "hsv",
        "mint",
        "purp",
        "ice",
        "twilight",
        "sunsetdark",
        "cividis",
        "teal",
    ]
    # Adding Dropdowns
    map_options = pn.widgets.Select(name="Tiles", options=mapbox_tiles)
    months = [calendar.month_abbr[i] for i in range(1, 13)]
    month_options = pn.widgets.Select(
        name="Month", options=dict(zip(months, range(1, 13)))
    )
    location_options = pn.widgets.Select(name="Location", options=["pickup", "dropoff"])
    cmap_option = pn.widgets.Select(name="Color Map", options=cmap)
    dashboard = interact(
        _create_plotly_monthly_plot,
        aspect=location_options,
        map_style=map_options,
        cmap=cmap_option,
        month=month_options,
        df=fixed(df),
    )
    title = pn.pane.Markdown("""# New York Monthly Map""")

    monthly_animated_tab = pn.Column(
        title, pn.Row(dashboard[1], dashboard[0], height=1300, width=1500)
    )
    return monthly_animated_tab


def _add_tile_layers(base_map=None):
    """
    This function adds four tile layers (cartodbpositron, cartodbdark_matter, stamenterrain, openstreetmap) to a given
    base map.
    ----------------------------------------------
    :param
        base_map(folium.Map): The given base map.
    """
    folium.TileLayer("cartodbpositron", name="light mode", control=True).add_to(
        base_map
    )
    folium.TileLayer("cartodbdark_matter", name="dark mode", control=True).add_to(
        base_map
    )
    folium.TileLayer("stamenterrain", name="stamenterrain", control=True).add_to(
        base_map
    )
    folium.TileLayer("openstreetmap", name="open street map", control=True).add_to(
        base_map
    )
    folium.LayerControl().add_to(base_map)


def _create_monthly_choropleth(
        df,
        month="Jan",
        aspect="pickup",
        log_count=False,
        cmap="YlGn",
        location="New York",
):
    """
    This function creates a folium choropleth based on different aspects of the given data.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
        month(String): The desired month which will be aggregated.
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff.
        log_count(bool): Shows data on log scale.
        cmap(String): The chosen colormap.
        location(String): The focus area.
    :return:
        folium.Choropleth: The created choropleth
    :raises
        ValueError: If the aggregation aspect is not 'pickup'/'dropoff'
    """
    if month == "All":
        month = None
    else:
        month = strptime(month, "%b").tm_mon
    data = _create_aggregator(
        df, month=month, aspect=aspect, log_count=log_count, choropleth=True
    )
    nyc_zones = read_geo_dataset("taxi_zones.geojson")
    nyc_zones["LocationID"] = nyc_zones["LocationID"].astype("str")
    info = ["zone", "LocationID", "borough"]
    columns = []
    if log_count:
        legend_name = f"{aspect} count (Log Scale)"
        columns.append("PULocationID") if aspect == "pickup" else columns.append(
            "DOLocationID"
        )
        columns.append(f"{aspect}_count_log")

    else:
        legend_name = f"{aspect} count"
        columns.append("PULocationID") if aspect == "pickup" else columns.append(
            "DOLocationID"
        )
        columns.append(f"{aspect}_count")
        info.append(f"{aspect}_count")
        if aspect == "pickup":
            nyc_zones = nyc_zones.merge(
                data[["PULocationID", "pickup_count"]],
                how="left",
                left_on="LocationID",
                right_on="PULocationID",
            )
        elif aspect == "dropoff":
            nyc_zones = nyc_zones.merge(
                data[["DOLocationID", "dropoff_count"]],
                how="left",
                left_on="LocationID",
                right_on="DOLocationID",
            )
        else:
            raise ValueError("Unknown aggregation aspect")

    base_map = _generate_base_map(default_location=location)
    choropleth = folium.Choropleth(
        geo_data=nyc_zones,
        name="choropleth",
        data=data,
        columns=columns,
        smooth_factor=0,
        key_on="feature.properties.LocationID",
        fill_color=cmap,
        legend_name=legend_name,
        fill_opacity=0.4,
        line_opacity=0.5,
    ).add_to(base_map)
    _add_tile_layers(base_map=base_map)
    # Display Region Label
    choropleth.geojson.add_child(folium.features.GeoJsonTooltip(info, labels=True))
    return base_map


def _create_choropleth_tab(df):
    """
    This function creates a folium choropleth tab with interactive widgets.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
    :return:
        pn.Column: The created choropleth panel element
    """
    cmap = [
        "BuGn",
        "BuPu",
        "GnBu",
        "OrRd",
        "PuBu",
        "PuBuGn",
        "PuRd",
        "RdPu",
        "YlGn",
        "YlGnBu",
        "YlOrBr",
        "YlOrRd",
    ]
    months = [calendar.month_abbr[i] for i in range(1, 13)]
    months.append("All")
    month_options = pn.widgets.Select(name="Month", options=months)
    location_options = pn.widgets.Select(name="Location", options=["pickup", "dropoff"])
    focus_area_options = pn.widgets.Select(
        name="Focus Area",
        options=[
            "New York",
            "Manhattan",
            "Brooklyn",
            "Bronx",
            "Queens",
            "Staten Island",
        ],
    )
    log_checkbox = pn.widgets.Checkbox(name="Log Scale")
    cmap_option = pn.widgets.Select(name="Color Map", options=cmap)
    dashboard = interact(
        _create_monthly_choropleth,
        location=focus_area_options,
        log_count=log_checkbox,
        cmap=cmap_option,
        month=month_options,
        aspect=location_options,
        df=fixed(df),
    )
    title = pn.pane.Markdown("""# New York Monthly Trip Choropleth""")

    monthly_choropleth_tab = pn.Column(
        title, pn.Row(dashboard[1], dashboard[0], height=1300, width=1500)
    )
    return monthly_choropleth_tab


def _generate_base_map(default_location="New York"):
    """
    This function creates a base folium map.
    ----------------------------------------------
    :param
        default_location(String): The default location of the base map.
    :return:
        folium.Map: The created base map
    """
    locations = {
        "New York": [40.693943, -73.985880],
        "Manhattan": [40.790932, -73.964016],
        "Brooklyn": [40.650002, -73.949997],
        "Bronx": [40.837048, -73.865433],
        "Queens": [40.706501, -73.823964],
        "Staten Island": [40.579021, -74.151535],
    }

    if default_location is not None and default_location not in locations:
        raise ValueError("Could not find the given location coordinates.")

    if default_location in [
        "Manhattan",
        "Brooklyn",
        "Bronx",
        "Queens",
        "Staten Island",
    ]:
        default_zoom_start = 12
    else:
        default_zoom_start = 10
    default_location = locations.get(default_location)

    base_map = folium.Map(
        location=default_location,
        control_scale=True,
        zoom_start=default_zoom_start,
        tiles=None,
    )
    return base_map


def _create_aggregator(
        df,
        month=None,
        event=None,
        aspect="pickup",
        animated=False,
        choropleth=False,
        log_count=False,
        event_heatmap=False,
):
    """
    This function aggregates the given data based on the other parameters set.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that should be aggregated.
        month(int): Shows the data for this month only.
        event(datetime tuple): The selected event as datetime tuple.
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff.
        animated(bool): If True -> the Pickup Location IDs / Drop off Location IDs and pickup/dropoff month will also be
                        added for later use in a plotly express plot.
        choropleth(bool): If True -> the Pickup Location IDs / Drop off Location IDs will also be added for later use
                          in a choropleth.
        log_count(bool): Shows data on log scale.
        event_heatmap(bool): If True -> the data will be grouped on a hourly level for a given Event.
    :return:
        df(pd.DataFrame): Aggregated data
    """
    cols_grp = [f"centers_lat_{aspect}", f"centers_long_{aspect}"]
    if month is not None:
        df = df.loc[df[f"{aspect}_month"] == month]
    if event is not None:
        df = df.loc[
            (df[f"{aspect}_datetime"] >= event[0])
            & (df[f"{aspect}_datetime"] < event[1])
            ]
    if choropleth:
        cols_grp.extend(["PULocationID"]) if aspect == "pickup" else cols_grp.extend(
            ["DOLocationID"]
        )
    if animated:
        cols_grp.extend([f"{aspect}_month"])
        cols_grp.extend(["PULocationID"]) if aspect == "pickup" else cols_grp.extend(
            ["DOLocationID"]
        )

    if event_heatmap:
        df["count"] = 1
        df_hour_list = []
        event_range = pd.date_range(event[0], event[1], freq="H")
        for stamp in event_range.delete(-1):
            month = stamp.month
            day = stamp.day
            hour = stamp.hour
            if aspect == "pickup":
                df_hour_list.append(
                    df.loc[
                        (df.pickup_hour == hour)
                        & (df.pickup_month == month)
                        & (df.pickup_day == day),
                        [f"centers_lat_{aspect}", f"centers_long_{aspect}", "count"],
                    ]
                        .groupby(cols_grp)
                        .sum()
                        .reset_index()
                        .values.tolist()
                )
            else:
                df_hour_list.append(
                    df.loc[
                        (df.dropoff_hour == hour)
                        & (df.dropoff_month == month)
                        & (df.dropoff_day == day),
                        [f"centers_lat_{aspect}", f"centers_long_{aspect}", "count"],
                    ]
                        .groupby(cols_grp)
                        .sum()
                        .reset_index()
                        .values.tolist()
                )
        return df_hour_list

    df_agg = df.groupby(cols_grp).size().to_frame(f"{aspect}_count").reset_index()

    if log_count:
        df_agg[f"{aspect}_count_log"] = np.log(df_agg[f"{aspect}_count"])
        df_agg.pop(f"{aspect}_count")
    return df_agg


def _create_inferno_cmap():
    """
    This function creates an inferno cmap for folium.
    ----------------------------------------------
    :return:
        branca.colormap: The inferno colormap.
        defaultdict: The gradient map which can be added to the map.
    """
    steps = 20
    colormap = branca.colormap.linear.YlOrRd_09.scale(0, 1).to_step(steps)
    gradient_map = defaultdict(dict)
    for i in range(steps):
        gradient_map[1 / steps * i] = colormap.rgb_hex_str(1 / steps * i)

    return colormap, gradient_map


def _create_heat_map(
        df,
        aspect="pickup",
        radius=15,
        location="New York",
        log_count=False,
        inferno_colormap=False,
):
    """
    This function creates a folium heatmap based on different aspects of the given data.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the heatmap.
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff.
        radius(int): The observed radius for each location upon creating the heatmap.
        location(String): The focus area.
        log_count(bool): Shows data on log scale.
        inferno_colormap(bool): If True -> changes the default cmap to inferno cmap.
    :return:
        folium.Heatmap: The created Heatmap
    """
    base_map = _generate_base_map(default_location=location)
    map_data = _create_aggregator(
        df, aspect=aspect, log_count=log_count
    ).values.tolist()

    if inferno_colormap:
        inferno_colormap, inferno_gradient = _create_inferno_cmap()
        HeatMap(
            data=map_data, radius=radius, gradient=inferno_gradient, name="heatmap"
        ).add_to(base_map)
        inferno_colormap.add_to(base_map)
    else:
        HeatMap(data=map_data, radius=radius, name="heatmap").add_to(base_map)
    _add_tile_layers(base_map=base_map)
    return base_map


def _create_general_heatmap_tab(df):
    """
    This function creates a folium heatmap tab with interactive widgets.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the heatmap.
    :return:
        pn.Column: The created heatmap panel element
    """
    location_options = pn.widgets.Select(name="Location", options=["pickup", "dropoff"])
    focus_area_options = pn.widgets.Select(
        name="Focus Area",
        options=[
            "New York",
            "Manhattan",
            "Brooklyn",
            "Bronx",
            "Queens",
            "Staten Island",
        ],
    )

    radius_option = pn.widgets.IntSlider(
        name="Heatmap Radius", start=5, end=20, step=1, value=15
    )
    log_checkbox = pn.widgets.Checkbox(name="Log Scale")
    inferno_checkbox = pn.widgets.Checkbox(name="Inferno colormap")
    dashboard = interact(
        _create_heat_map,
        location=focus_area_options,
        radius=radius_option,
        log_count=log_checkbox,
        inferno_colormap=inferno_checkbox,
        aspect=location_options,
        df=fixed(df),
    )
    title = pn.pane.Markdown("""# New York Trip Heatmap""")

    general_heatmap_tab = pn.Column(
        title, pn.Row(dashboard[1], dashboard[0], height=1300, width=1500)
    )
    return general_heatmap_tab


def _create_events_tab(df):
    """
    This function creates a folium choropleth tab for events with interactive widgets.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
    :return:
        pn.Column: The created choropleth panel element
    """
    cmap = [
        "BuGn",
        "BuPu",
        "GnBu",
        "OrRd",
        "PuBu",
        "PuBuGn",
        "PuRd",
        "RdPu",
        "YlGn",
        "YlGnBu",
        "YlOrBr",
        "YlOrRd",
    ]
    events = {
        "New Year": (
            datetime.datetime(2020, 1, 1, 0, 0, 0),
            datetime.datetime(2020, 1, 2, 0, 0, 0),
        ),
        "Superbowl": (
            datetime.datetime(2020, 2, 2, 18, 30, 0),
            datetime.datetime(2020, 2, 2, 22, 30, 0),
        ),
        "Independence Day": (
            datetime.datetime(2020, 7, 4, 0, 0, 0),
            datetime.datetime(2020, 7, 5, 0, 0, 0),
        ),
        "Election": (
            datetime.datetime(2020, 11, 3, 6, 0, 0),
            datetime.datetime(2020, 11, 3, 21, 0, 0),
        ),
        "Thanksgiving Parade": (
            datetime.datetime(2020, 11, 26, 9, 0, 0),
            datetime.datetime(2020, 11, 26, 12, 0, 0),
        ),
        "Black Friday": (
            datetime.datetime(2020, 11, 27, 0, 0, 0),
            datetime.datetime(2020, 11, 28, 0, 0, 0),
        ),
        "Christmas": (
            datetime.datetime(2020, 12, 24, 0, 0, 0),
            datetime.datetime(2020, 12, 27, 0, 0, 0),
        ),
        "New Years Eve": (
            datetime.datetime(2020, 12, 31, 0, 0, 0),
            datetime.datetime(2021, 1, 1, 0, 0, 0),
        ),
        "Individual Event": (
            datetime.datetime(1, 1, 1, 1, 1, 1),
            datetime.datetime(1, 1, 1, 1, 1, 1),
        ),
    }
    timespan_options = pn.widgets.DatetimeRangeInput(
        name="Individual Event Timespan",
        start=datetime.datetime(2020, 1, 1, 0, 0, 0),
        end=datetime.datetime(2021, 1, 1, 0, 0, 0),
        value=(
            datetime.datetime(2020, 1, 1, 0, 0, 0),
            datetime.datetime(2021, 1, 1, 0, 0, 0),
        ),
    )
    event_options = pn.widgets.Select(name="Event", options=events)
    location_options = pn.widgets.Select(name="Location", options=["pickup", "dropoff"])
    focus_area_options = pn.widgets.Select(
        name="Focus Area",
        options=[
            "New York",
            "Manhattan",
            "Brooklyn",
            "Bronx",
            "Queens",
            "Staten Island",
        ],
    )
    log_checkbox = pn.widgets.Checkbox(name="Log Scale")
    cmap_option = pn.widgets.Select(name="Color Map", options=cmap)
    dashboard = interact(
        _create_event_choropleth,
        location=focus_area_options,
        log_count=log_checkbox,
        cmap=cmap_option,
        event=event_options,
        timespan=timespan_options,
        aspect=location_options,
        df=fixed(df),
    )
    title = pn.pane.Markdown("""# New York Events""")

    event_tab = pn.Column(
        title, pn.Row(dashboard[1], dashboard[0], height=1300, width=1500)
    )
    return event_tab


def _create_event_choropleth(
        df,
        event=(
                datetime.datetime(2020, 1, 1, 0, 0, 0),
                datetime.datetime(2020, 1, 2, 0, 0, 0),
        ),
        timespan=(
                datetime.datetime(2020, 1, 1, 0, 0, 0),
                datetime.datetime(2021, 1, 1, 0, 0, 0),
        ),
        aspect="pickup",
        log_count=False,
        cmap="YlGn",
        location="New York",
):
    """
    This function creates a folium choropleth based on different aspects of the given data.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
        event(datetime tuple): The desired event which will be aggregated.
        timespan(datetime tuple): Simulates an individual event.
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff.
        log_count(bool): Shows data on log scale.
        cmap(String): The chosen colormap.
        location(String): The focus area.
    :return:
        folium.Choropleth: The created choropleth
    :raises
        ValueError: If the aggregation aspect is not 'pickup'/'dropoff'
    """
    if event == (
            datetime.datetime(1, 1, 1, 1, 1, 1),
            datetime.datetime(1, 1, 1, 1, 1, 1),
    ):
        event = timespan
    data = _create_aggregator(
        df, event=event, aspect=aspect, log_count=log_count, choropleth=True
    )
    nyc_zones = read_geo_dataset("taxi_zones.geojson")
    nyc_zones["LocationID"] = nyc_zones["LocationID"].astype("str")
    info = ["zone", "LocationID", "borough"]
    columns = []
    if log_count:
        legend_name = f"{aspect} count (Log Scale)"
        columns.append("PULocationID") if aspect == "pickup" else columns.append(
            "DOLocationID"
        )
        columns.append(f"{aspect}_count_log")

    else:
        legend_name = f"{aspect} count"
        columns.append("PULocationID") if aspect == "pickup" else columns.append(
            "DOLocationID"
        )
        columns.append(f"{aspect}_count")
        info.append(f"{aspect}_count")
        if aspect == "pickup":
            nyc_zones = nyc_zones.merge(
                data[["PULocationID", "pickup_count"]],
                how="left",
                left_on="LocationID",
                right_on="PULocationID",
            )
        elif aspect == "dropoff":
            nyc_zones = nyc_zones.merge(
                data[["DOLocationID", "dropoff_count"]],
                how="left",
                left_on="LocationID",
                right_on="DOLocationID",
            )
        else:
            raise ValueError("Unknown aggregation aspect")

    base_map = _generate_base_map(default_location=location)
    choropleth = folium.Choropleth(
        geo_data=nyc_zones,
        name="choropleth",
        data=data,
        columns=columns,
        smooth_factor=0,
        key_on="feature.properties.LocationID",
        fill_color=cmap,
        legend_name=legend_name,
        fill_opacity=0.4,
        line_opacity=0.5,
    ).add_to(base_map)
    _add_tile_layers(base_map=base_map)
    # Display Region Label
    choropleth.geojson.add_child(folium.features.GeoJsonTooltip(info, labels=True))
    return base_map


def _create_zone_choropleth(
        df,
        month="all",
        zone="1",
        aspect="outbound",
        log_count=False,
        cmap="YlGn",
        location="New York",
):
    """
    This function creates a folium choropleth based on different aspects of the given data.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
        month(String): The desired month which will be aggregated.
        zone(String): The selected zone.
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff.
        log_count(bool): Shows data on log scale.
        cmap(String): The chosen colormap.
        location(String): The focus area.
    :return:
        folium.Choropleth: The created choropleth
    :raises
        ValueError: If the aggregation aspect is not 'pickup'/'dropoff'
    """
    if month == "All":
        month = None
    else:
        month = strptime(month, "%b").tm_mon
    if aspect == "outbound":
        aspect_abr = "DO"
    else:
        aspect_abr = "PU"
    data = _create_zone_aggregator(
        df, month=month, zone=zone, aspect=aspect, log_count=log_count, choropleth=True
    )
    nyc_zones = read_geo_dataset("taxi_zones.geojson")
    nyc_zones["LocationID"] = nyc_zones["LocationID"].astype("str")
    info = ["zone", "LocationID", "borough"]
    columns = []
    if log_count:
        legend_name = f"{aspect} count (Log Scale)"
        columns.append(f"{aspect_abr}LocationID")
        columns.append(f"{aspect}_count_log")

    else:
        legend_name = f"{aspect} count"
        columns.append(f"{aspect_abr}LocationID")
        columns.append(f"{aspect}_count")
        info.append(f"{aspect}_count")
        if aspect in ["outbound", "inbound"]:
            nyc_zones = nyc_zones.merge(
                data[[f"{aspect_abr}LocationID", f"{aspect}_count"]],
                how="left",
                left_on="LocationID",
                right_on=f"{aspect_abr}LocationID",
            )
        else:
            raise ValueError("Unknown aggregation aspect")

    highlight_zone = nyc_zones.loc[nyc_zones["LocationID"] == zone]

    base_map = _generate_base_map(default_location=location)
    choropleth = folium.Choropleth(
        geo_data=nyc_zones,
        name="choropleth",
        data=data,
        columns=columns,
        smooth_factor=0,
        key_on="feature.properties.LocationID",
        fill_color=cmap,
        legend_name=legend_name,
        fill_opacity=0.4,
        line_opacity=0.5,
    ).add_to(base_map)
    highlighted_zone = folium.Choropleth(
        geo_data=highlight_zone,
        name="selected zone",
        columns=columns,
        smooth_factor=0,
        fill_opacity=1,
        line_color="red",
    ).add_to(base_map)
    _add_tile_layers(base_map=base_map)
    # Display Region Label
    choropleth.geojson.add_child(folium.features.GeoJsonTooltip(info, labels=True))
    highlighted_zone.geojson.add_child(
        folium.features.GeoJsonTooltip(info, labels=True)
    )
    return base_map


def _create_zone_tab(df):
    """
    This function creates a folium choropleth tab to pick a certain zone with interactive widgets.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
    :return:
        pn.Column: the created choropleth panel element
    """
    cmap = [
        "BuGn",
        "BuPu",
        "GnBu",
        "OrRd",
        "PuBu",
        "PuBuGn",
        "PuRd",
        "RdPu",
        "YlGn",
        "YlGnBu",
        "YlOrBr",
        "YlOrRd",
    ]
    months = [calendar.month_abbr[i] for i in range(1, 13)]
    months.append("All")
    month_options = pn.widgets.Select(name="Month", options=months)
    nyc_zones = read_geo_dataset("taxi_zones.geojson")
    nyc_zones["LocationID"] = nyc_zones["LocationID"].astype("str")
    zones = dict(zip(nyc_zones.zone, nyc_zones.LocationID))
    zone_options = pn.widgets.Select(name="Zone", options=zones)
    direction_options = pn.widgets.Select(
        name="Direction", options={"Outbound": "outbound", "Inbound": "inbound"}
    )
    focus_area_options = pn.widgets.Select(
        name="Focus Area",
        options=[
            "New York",
            "Manhattan",
            "Brooklyn",
            "Bronx",
            "Queens",
            "Staten Island",
        ],
    )
    log_checkbox = pn.widgets.Checkbox(name="Log Scale")
    cmap_option = pn.widgets.Select(name="Color Map", options=cmap)
    dashboard = interact(
        _create_zone_choropleth,
        location=focus_area_options,
        log_count=log_checkbox,
        cmap=cmap_option,
        month=month_options,
        zone=zone_options,
        aspect=direction_options,
        df=fixed(df),
    )
    title = pn.pane.Markdown("""# New York Inbound and Outbound""")

    zone_tab = pn.Column(
        title, pn.Row(dashboard[1], dashboard[0], height=1300, width=1500)
    )
    return zone_tab


def _create_zone_aggregator(
        df, month=None, zone=None, aspect="outbound", choropleth=False, log_count=False
):
    """
    This function aggregates the given data based on the other parameters set.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that should be aggregated.
        month(int): Shows the data for this month only.
        zone(String): The selected zone.
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff.
        choropleth(bool): If True -> the Pickup Location IDs / Drop off Location IDs will also be added for later use
                          in a choropleth.
        log_count(bool): Shows data on log scale.
    :return:
        df(pd.DataFrame): Aggregated data
    :raises:
        ValueError: If the aggregation aspect is not 'pickup'/'dropoff'
    """
    pickup_cols_grp = ["centers_lat_pickup", "centers_long_pickup"]
    dropoff_cols_grp = ["centers_lat_dropoff", "centers_long_dropoff"]

    if aspect == "outbound":
        location = "pickup"
        location_abr = "PU"
    else:
        location = "dropoff"
        location_abr = "DO"

    if zone is not None:
        if month is not None:
            df = df.loc[
                (df[f"{location}_month"] == month)
                & (df[f"{location_abr}LocationID"] == zone)
                ]
        else:
            df = df.loc[df[f"{location_abr}LocationID"] == zone]
    if choropleth:
        pickup_cols_grp.extend(["PULocationID"])
        dropoff_cols_grp.extend(["DOLocationID"])

    if location == "pickup":
        df_agg = (
            df.groupby(dropoff_cols_grp)
                .size()
                .to_frame(f"{aspect}_count")
                .reset_index()
        )
    elif location == "dropoff":
        df_agg = (
            df.groupby(pickup_cols_grp).size().to_frame(f"{aspect}_count").reset_index()
        )
    else:
        raise ValueError("Unknown aggregation aspect")

    if log_count:
        df_agg[f"{aspect}_count_log"] = np.log(df_agg[f"{aspect}_count"])
        df_agg.pop(f"{aspect}_count")
    return df_agg


def _create_event_heat_map(
        df,
        aspect="pickup",
        radius=15,
        location="New York",
        event=(
                datetime.datetime(2020, 1, 1, 0, 0, 0),
                datetime.datetime(2020, 1, 2, 0, 0, 0),
        ),
        timespan=(
                datetime.datetime(2020, 1, 1, 0, 0, 0),
                datetime.datetime(2021, 1, 1, 0, 0, 0),
        ),
):
    """
    This function creates a folium heatmap with time based on different aspects of the given data.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the heatmap.
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff.
        radius(int): The observed radius for each location upon creating the heatmap.
        location(String): The focus area.
        event(datetime tuple): The desired event which will be aggregated.
        timespan(datetime tuple): Simulates an individual event.
    :return:
        folium.HeatmapWithTime: The created HeatmapWithTime
    """
    if event == (
            datetime.datetime(1, 1, 1, 1, 1, 1),
            datetime.datetime(1, 1, 1, 1, 1, 1),
    ):
        event = timespan
    base_map = _generate_base_map(default_location=location)
    map_data = _create_aggregator(df, aspect=aspect, event=event, event_heatmap=True)
    index = (
        pd.date_range(event[0], event[1], freq="H")
            .strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
    )
    del index[-1]
    HeatMapWithTime(
        data=map_data,
        radius=radius,
        min_opacity=0.5,
        max_opacity=0.8,
        name="heatmap",
        index=index,
        use_local_extrema=True,
    ).add_to(base_map)
    _add_tile_layers(base_map=base_map)
    return base_map


def _create_event_heatmap_tab(df):
    """
    This function creates a folium heatmap with time tab with interactive widgets.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the heatmap with time.
    :return:
        pn.Column: The created heatmap with time panel element
    """
    events = {
        "New Year": (
            datetime.datetime(2020, 1, 1, 0, 0, 0),
            datetime.datetime(2020, 1, 2, 0, 0, 0),
        ),
        "Superbowl": (
            datetime.datetime(2020, 2, 2, 0, 0, 0),
            datetime.datetime(2020, 2, 3, 0, 0, 0),
        ),
        "Independence Day": (
            datetime.datetime(2020, 7, 4, 0, 0, 0),
            datetime.datetime(2020, 7, 5, 0, 0, 0),
        ),
        "Election": (
            datetime.datetime(2020, 11, 3, 0, 0, 0),
            datetime.datetime(2020, 11, 4, 0, 0, 0),
        ),
        "Thanksgiving Parade": (
            datetime.datetime(2020, 11, 26, 0, 0, 0),
            datetime.datetime(2020, 11, 27, 0, 0, 0),
        ),
        "Black Friday": (
            datetime.datetime(2020, 11, 27, 0, 0, 0),
            datetime.datetime(2020, 11, 28, 0, 0, 0),
        ),
        "Christmas": (
            datetime.datetime(2020, 12, 24, 0, 0, 0),
            datetime.datetime(2020, 12, 27, 0, 0, 0),
        ),
        "New Years Eve": (
            datetime.datetime(2020, 12, 31, 0, 0, 0),
            datetime.datetime(2021, 1, 1, 0, 0, 0),
        ),
        "Individual Event": (
            datetime.datetime(1, 1, 1, 1, 1, 1),
            datetime.datetime(1, 1, 1, 1, 1, 1),
        ),
    }
    timespan_options = pn.widgets.DatetimeRangeInput(
        name="Individual Event Timespan",
        start=datetime.datetime(2020, 1, 1, 0, 0, 0),
        end=datetime.datetime(2021, 1, 1, 0, 0, 0),
        value=(
            datetime.datetime(2020, 1, 1, 0, 0, 0),
            datetime.datetime(2021, 1, 1, 0, 0, 0),
        ),
    )
    event_options = pn.widgets.Select(name="Event", options=events)
    location_options = pn.widgets.Select(name="Location", options=["pickup", "dropoff"])
    focus_area_options = pn.widgets.Select(
        name="Focus Area",
        options=[
            "New York",
            "Manhattan",
            "Brooklyn",
            "Bronx",
            "Queens",
            "Staten Island",
        ],
    )
    radius_option = pn.widgets.IntSlider(
        name="Heatmap Radius", start=5, end=20, step=1, value=15
    )
    dashboard = interact(
        _create_event_heat_map,
        location=focus_area_options,
        radius=radius_option,
        aspect=location_options,
        event=event_options,
        timespan=timespan_options,
        df=fixed(df),
    )
    title = pn.pane.Markdown("""# New York Trip Heatmap with Time""")

    general_heatmap_tab = pn.Column(
        title, pn.Row(dashboard[1], dashboard[0], height=1300, width=1500)
    )
    return general_heatmap_tab


def monthly_visualization(df, month=None, hist=None, xlim=None):
    """
    This function visualizes the monthly distribution of trip duration of the given DataFrame, including comparison to
    a normal distribution & returns the distribution plot.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame with trip data.
        month(String): Name of selected month that gets observed.
        hist(bool): Plot histogram bars or not.
        xlim(int): Limit of x-axis.
    :returns
        pn.pane.Matplotlib(): Monthly distribution plot.
    """
    month = strptime(month, "%b").tm_mon
    df_m = df.loc[df["pickup_month"] == month]

    fig = Figure(figsize=(8, 6))
    ax = fig.subplots()
    l1 = sns.distplot(df_m["trip_duration_minutes"], ax=ax, hist=hist)
    l2 = sns.distplot(
        random.normal(size=5000, loc=df_m["trip_duration_minutes"].mean()),
        hist=hist,
        ax=ax,
    )
    l3 = ax.axvline(df_m["trip_duration_minutes"].mean(), linestyle="dashed")
    ax.legend(
        [l1, l2, l3],
        labels=["Original distribution", "Normal distribution", "Mean"],
        loc="upper right",
        borderaxespad=0.3,
    )
    plt.setp(ax, xlim=(0, xlim))
    ax.set_xlabel("Trip duration (minutes)")
    ax.set_ylabel("Density")
    mpl_pane = pn.pane.Matplotlib(fig, tight=True)
    return mpl_pane


def _create_duration_distribution_tab(df):
    """
    This function creates plots for monthly trip duration distributions and a comparison to normal distribution.

    ----------------------------------------------

    :param
        df(pd.DataFrame): Data that is used to make the distribution plot.
    :return:
        pn.pane.Matplotlib(): Monthly distribution plot.

    """
    months = [calendar.month_abbr[i] for i in range(1, 13)]
    month_options = pn.widgets.Select(name="Month", options=months)
    hist_options = pn.widgets.Checkbox(name="Histogram")
    xlim_options = pn.widgets.IntSlider(
        name="Range x-axis (minutes)", start=20, end=80, step=10, value=40
    )

    dashboard = interact(
        monthly_visualization,
        month=month_options,
        hist=hist_options,
        xlim=xlim_options,
        df=fixed(df),
    )
    title = pn.pane.Markdown("""# Monthly trip duration distribution""")

    duration_distribution_tab = pn.Column(
        title, pn.Row(dashboard[1], dashboard[0], height=1300, width=1500)
    )
    return duration_distribution_tab


def basic_plots(df, borough, pu_do, feature):
    """
    This function creates a Figure of 4 subplots (barplots) which give a basic overview about trip data and features.
    The left sided plots show information regarding the total trip number (monthly, weekly) of the selected borough.
    The ride sided plots show information regarding the selected feature.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
        borough(String): Name of a borough to inspect. If not set, all NYC trips get selected.
        pu_do(String): 'pickup' or 'dropoff' trips get investigated.
        feature(String): Feature-coloumn to be investigated in the right sided barplots.
    :returns
        pn.pane.Matplotlib(): Basic plots.
    """
    geojson_df = read_geo_dataset("taxi_zones.geojson")

    if borough != 'NYC complete':
        if borough != 'Airports':
            if pu_do == 'pickup':
                df = df.loc[df['pickup_borough'] == borough]
            else:
                df = df.loc[df['dropoff_borough'] == borough]
        if borough == 'Airports':
            if pu_do == 'pickup':
                df = df.loc[(df['pickup_zone'].str.contains('Airport'))]
            else:
                df = df.loc[(df['dropoff_zone'].str.contains('Airport'))]

    # number of trips dataframes
    df_agg_count_monthly = yellowcab.eda.agg_stats(df['pickup_datetime'].dt.month, df['pickup_month'], ['count'])
    df_agg_count_weekly = yellowcab.eda.agg_stats(df['pickup_datetime'].dt.week, df['pickup_month'], ['count'])

    # feature dataframes
    if feature != 'start_or_destination':
        # mean plots
        df_agg_mean_monthly = yellowcab.eda.agg_stats(df['pickup_datetime'].dt.month, df[feature], ['mean'])
        df_agg_mean_weekly = yellowcab.eda.agg_stats(df['pickup_datetime'].dt.week, df[feature], ['mean'])
    else:
        # PU-DO-Plots
        if pu_do == 'pickup':
            col1 = 'DOLocationID'
            col2 = 'dropoff_borough'
        else:
            col1 = 'PULocationID'
            col2 = 'pickup_borough'

        dopu_loc_ser_ids = df[col1].value_counts()
        dopu_loc_ser_ids.sort_values(ascending=False, inplace=True)
        dopu_loc_ser_ids = dopu_loc_ser_ids[:10]
        df_dopu_ids = dopu_loc_ser_ids.to_frame().reset_index()
        df_dopu_ids = df_dopu_ids.rename(columns={'index': col1,
                                                  col1: 'Count'})
        dopu_loc_ser_borough = df[col2].value_counts()
        dopu_loc_ser_borough.sort_values(ascending=False, inplace=True)
        dopu_loc_ser_borough = dopu_loc_ser_borough[:10]
        df_dopu_boroughs = dopu_loc_ser_borough.to_frame().reset_index()
        df_dopu_boroughs = df_dopu_boroughs.rename(columns={'index': col2,
                                                            col2: 'Count'})

    # create figure column names
    if pu_do == 'pickup':
        col_name_start_or_destination = 'Dropoff areas of trips started at {boroughname}'.format(boroughname=borough)
    else:
        col_name_start_or_destination = 'Pickup areas of trips ended at {boroughname}'.format(boroughname=borough)

    featureDict = {
        'trip_duration_minutes': 'Trip Duration (minutes)',
        'trip_distance': 'Trip Distance (miles)',
        'total_amount': 'Total Amount ($)',
        'tip_amount': 'Tip Amount ($)',
        'passenger_count': 'Passenger Count',
        'start_or_destination': col_name_start_or_destination}

    col_feature = '{featurename}'.format(featurename=featureDict[feature])
    col_total_number = 'Total number of trips with {pudo} at {boroughname}'.format(pudo=pu_do, boroughname=borough)
    col_list = [col_total_number, col_feature]

    fig = Figure(figsize=(15, 10))
    axes = fig.subplots(2, 2)
    # number of trips plots
    axes[0, 0].bar(df_agg_count_monthly.index, df_agg_count_monthly['count_pickup_month'], color='orange')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Number of trips')
    axes[0, 0].ticklabel_format(useOffset=False, style='plain', axis='y')

    axes[1, 0].bar(df_agg_count_weekly.index, df_agg_count_weekly['count_pickup_month'], color='orange')
    axes[1, 0].set_xlabel('Week')
    axes[1, 0].set_ylabel('Number of trips')
    axes[1, 0].ticklabel_format(useOffset=False, style='plain', axis='y')

    if feature != 'start_or_destination':
        # mean plots
        axes[0, 1].bar(df_agg_mean_monthly.index, df_agg_mean_monthly['mean_{featurename}'.format(featurename=feature)])
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Mean {featurename}'.format(featurename=featureDict[feature]))
        axes[0, 1].ticklabel_format(useOffset=False, style='plain', axis='y')

        axes[1, 1].bar(df_agg_mean_weekly.index, df_agg_mean_weekly['mean_{featurename}'.format(featurename=feature)])
        axes[1, 1].set_xlabel('Week')
        axes[1, 1].set_ylabel('Mean {featurename}'.format(featurename=featureDict[feature]))
        axes[1, 1].ticklabel_format(useOffset=False, style='plain', axis='y')

        col_feature = 'Mean {featurename}'.format(featurename=featureDict[feature])

    else:
        # count plots
        axes[0, 1].bar(df_dopu_ids[col1], df_dopu_ids['Count'])
        axes[0, 1].set_xlabel('Location IDs')
        axes[0, 1].set_ylabel('Number of trips')
        axes[0, 1].ticklabel_format(useOffset=False, style='plain', axis='y')

        axes[1, 1].bar(df_dopu_boroughs[col2], df_dopu_boroughs['Count'])
        axes[1, 1].set_xlabel('Boroughs')
        axes[1, 1].set_ylabel('Number of trips')
        axes[1, 1].ticklabel_format(useOffset=False, style='plain', axis='y')

    for ax, col in zip(axes[0], col_list):
        ax.set_title(col)

    fig.subplots_adjust(left=0.1, top=0.9)
    fig.tight_layout(pad=3.0)
    title = 'Basic plots for {boroughname}'.format(boroughname=borough)
    fig.suptitle(t=title, fontsize=18, y=0.99)

    mpl_pane = pn.pane.Matplotlib(fig, tight=True)
    return mpl_pane


def _create_basic_plots_tab(df):
    """
    This function creates a Figure of 4 subplots (barplots) which give a basic overview about trip data and features.
    The left sided plots show information regarding the total trip number (monthly, weekly) of the selected borough.
    The ride sided plots show information regarding the selected feature.

    ----------------------------------------------

    :param
        df(pd.DataFrame): Data that is used to make the distribution plot.
    :return:
        pn.pane.Matplotlib(): Basic plots.

    """
    df_geo = read_geo_dataset("taxi_zones.geojson")
    df_geo["LocationID"] = df_geo["LocationID"].astype("str")

    boroughs = list(df_geo['borough'].unique())
    boroughs = np.insert(boroughs, 0, "NYC complete")
    boroughs = np.insert(boroughs, 0, "Airports")
    # delete EWR airport
    boroughs = boroughs[boroughs != 'EWR']
    boroughs_list = boroughs.tolist()
    borough_options = pn.widgets.Select(name="Borough", options=boroughs_list)

    feature_list = ['trip_duration_minutes',
                    'trip_distance',
                    'total_amount',
                    'tip_amount',
                    'passenger_count',
                    'start_or_destination']
    feature_options = pn.widgets.Select(name="Features", options=feature_list)

    pu_do_list = ['pickup',
                  'dropoff']
    pu_do_options = pn.widgets.Select(name="Trips with pickup or dropoff in the selected borough", options=pu_do_list)

    # add start and end zone + borough to trip dataframe
    df_viz = df.merge(df_geo[['LocationID', 'zone', 'borough']], how='left', left_on='PULocationID',
                      right_on='LocationID')
    df_viz.rename(columns={'zone': 'pickup_zone', 'borough': 'pickup_borough'}, inplace=True)
    df_viz = df_viz.drop(['LocationID'], axis=1)
    df_viz = df_viz.merge(df_geo[['LocationID', 'zone', 'borough']], how='left', left_on='DOLocationID',
                          right_on='LocationID')
    df_viz.rename(columns={'zone': 'dropoff_zone', 'borough': 'dropoff_borough'}, inplace=True)
    df_viz = df_viz.drop(['LocationID'], axis=1)

    dashboard = interact(
        basic_plots,
        borough=borough_options,
        feature=feature_options,
        pu_do=pu_do_options,
        df=fixed(df_viz),
    )
    title = ""
    basic_plots_tab = pn.Column(
        title, pn.Row(dashboard[1], dashboard[0], height=1300, width=1500)
    )
    return basic_plots_tab


def create_dashboard(df):
    """
    This function creates an interactive panel dashboard.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the interactive.
    :return:
        pn.Tabs: The interactive panel dashboard
    """

    heatmap_general = ("Heatmap General", _create_general_heatmap_tab(df))
    choropleth_monthly = ("Choropleth Monthly", _create_choropleth_tab(df))
    plotly_express_animated_monthly = (
        "Monthly Scatter Plot",
        _create_monthly_animated_tab(df),
    )
    duration_distribution_monthly = (
        "Duration distribution",
        _create_duration_distribution_tab(df),
    )
    basic_plots_viz = (
        "Basic plots",
        _create_basic_plots_tab(df),
    )
    events = ("Events", _create_events_tab(df))
    zones = ("Zone", _create_zone_tab(df))
    event_heatmap = ("Event Heatmap", _create_event_heatmap_tab(df))
    dashboard = pn.Tabs(
        heatmap_general,
        choropleth_monthly,
        event_heatmap,
        plotly_express_animated_monthly,
        duration_distribution_monthly,
        basic_plots_viz,
        zones,
        events,
    )
    return dashboard
