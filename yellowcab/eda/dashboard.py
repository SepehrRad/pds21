import calendar
import datetime
from collections import defaultdict
from time import strptime

import branca.colormap
import folium
import numpy as np
import panel as pn
from folium.plugins import HeatMap
from panel.interact import fixed, interact

from yellowcab.io.input import read_geo_dataset


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
    :returns
        folium.Choropleth: The created choropleth
    :raises
        ValueError: if the aggregation aspect is not 'pickup'/'dropoff'
    """
    month = strptime(month, "%b").tm_mon
    data = _create_aggregator(
        df, month=month, aspect=aspect, log_count=log_count, choropleth=True
    )
    nyc_zones = read_geo_dataset("taxi_zones.geojson")
    nyc_zones["LocationID"] = nyc_zones["LocationID"].astype("str")
    info = ["zone", "LocationID", "borough"]
    columns = []
    if log_count:
        legend_name = f"{aspect} Count (Log Scale)"
        columns.append("PULocationID") if aspect == "pickup" else columns.append(
            "DOLocationID"
        )
        columns.append(f"{aspect}_count_log")

    else:
        legend_name = f"{aspect} Count"
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
    folium.TileLayer('cartodbpositron', name="light mode", control=True).add_to(base_map)
    folium.TileLayer('cartodbdark_matter', name="dark mode", control=True).add_to(base_map)
    folium.TileLayer('stamenterrain', name="stamenterrain", control=True).add_to(base_map)
    folium.TileLayer('openstreetmap', name="openstreetmap", control=True).add_to(base_map)
    folium.LayerControl().add_to(base_map)
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
    month_options = pn.widgets.Select(name="Month", options=months)
    location_options = pn.widgets.Select(name="Location", options=["pickup", "dropoff"])
    focus_area_options = pn.widgets.Select(
        name="Focus Area", options=["New York", "Manhattan", "Brooklyn", "Bronx", "Queens", "Staten Island"]
    )
    log_checkbox = pn.widgets.Checkbox(name="Log scale")
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
        folium.Map: the created base map
    """
    locations = {
        "New York": [40.693943, -73.985880],
        "Manhattan": [40.790932, -73.964016],
        "Brooklyn": [40.650002, -73.949997],
        "Bronx": [40.837048, -73.865433],
        "Queens": [40.706501, -73.823964],
        "Staten Island": [40.579021, -74.151535]
    }

    if default_location is not None and default_location not in locations:
        raise ValueError("Could not find the given location coordinates.")

    if default_location in ["Manhattan", "Brooklyn", "Bronx", "Queens", "Staten Island"]:
        default_zoom_start = 12
    else:
        default_zoom_start = 10
    default_location = locations.get(default_location)

    base_map = folium.Map(
        location=default_location,
        control_scale=True,
        zoom_start=default_zoom_start,
        tiles=None
    )
    return base_map


def _create_aggregator(
        df, month=None, event=None, aspect="pickup", choropleth=False, log_count=False
):
    """
    This function creates a base folium map.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that should be aggregated
        month(int): Shows the data for this month only
        event(datetime tuple): The selected event as datetime tuple
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff
        choropleth(bool): If True -> the Pickup Location IDs / Drop off Location IDs will also be added for later use
                          in a choropleth.
        log_count(bool): Shows data on log scale
    :return:
        df(pd.DataFrame): Aggregated Data
    :raises:
        ValueError: if the aggregation aspect is not 'pickup'/'dropoff'
    """
    pickup_cols_grp = ["centers_lat_pickup", "centers_long_pickup"]
    dropoff_cols_grp = ["centers_lat_dropoff", "centers_long_dropoff"]
    if month is not None:
        df = df.loc[df[f"{aspect}_month"] == month]
    if event is not None:
        df = df.loc[(df[f"{aspect}_datetime"] >= event[0]) & (df[f"{aspect}_datetime"] < event[1])]
    if choropleth:
        pickup_cols_grp.extend(["PULocationID"])
        dropoff_cols_grp.extend(["DOLocationID"])

    if aspect == "pickup":
        df_agg = (
            df.groupby(pickup_cols_grp).size().to_frame("pickup_count").reset_index()
        )
    elif aspect == "dropoff":
        df_agg = (
            df.groupby(dropoff_cols_grp).size().to_frame("dropoff_count").reset_index()
        )
    else:
        raise ValueError("Unknown aggregation aspect")

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
    if log_count:
        map_data = _create_aggregator(df, aspect=aspect, log_count=True).values.tolist()
    else:
        map_data = _create_aggregator(df, aspect=aspect).values.tolist()

    if inferno_colormap:
        inferno_colormap, inferno_gradient = _create_inferno_cmap()
        HeatMap(data=map_data, radius=radius, gradient=inferno_gradient, name="heatmap").add_to(
            base_map
        )
        inferno_colormap.add_to(base_map)
    else:
        HeatMap(data=map_data, radius=radius, name="heatmap").add_to(base_map)
    folium.TileLayer('cartodbpositron', name="light mode", control=True).add_to(base_map)
    folium.TileLayer('cartodbdark_matter', name="dark mode", control=True).add_to(base_map)
    folium.TileLayer('stamenterrain', name="stamenterrain", control=True).add_to(base_map)
    folium.TileLayer('openstreetmap', name="openstreetmap", control=True).add_to(base_map)
    folium.LayerControl().add_to(base_map)
    return base_map


def _create_general_heatmap_tab(df):
    """
    This function creates a folium heatmap tab with interactive widgets.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the heatmap.
    :return:
        pn.Column: the created heatmap panel element
    """
    location_options = pn.widgets.Select(name="Location", options=["pickup", "dropoff"])
    focus_area_options = pn.widgets.Select(
        name="Focus Area", options=["New York", "Manhattan", "Brooklyn", "Bronx", "Queens", "Staten Island"]
    )

    radius_option = pn.widgets.IntSlider(
        name="Heatmap Radius", start=5, end=20, step=1, value=15
    )
    log_checkbox = pn.widgets.Checkbox(name="Log scale")
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
    This function creates a folium choropleth tab with interactive widgets.
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
    events = {
        "New Year": (datetime.datetime(2020, 1, 1, 0, 0, 0), datetime.datetime(2020, 1, 2, 0, 0, 0)),
        "Superbowl": (datetime.datetime(2020, 2, 2, 18, 30, 0), datetime.datetime(2020, 2, 2, 22, 30, 0)),
        "Independence Day": (datetime.datetime(2020, 7, 4, 0, 0, 0), datetime.datetime(2020, 7, 5, 0, 0, 0)),
        "Election": (datetime.datetime(2020, 11, 3, 6, 0, 0), datetime.datetime(2020, 11, 3, 21, 0, 0)),
        "Thanksgiving Parade": (datetime.datetime(2020, 11, 26, 9, 0, 0), datetime.datetime(2020, 11, 26, 12, 0, 0)),
        "Black Friday": (datetime.datetime(2020, 11, 27, 0, 0, 0), datetime.datetime(2020, 11, 28, 0, 0, 0)),
        "Christmas": (datetime.datetime(2020, 12, 24, 0, 0, 0), datetime.datetime(2020, 12, 27, 0, 0, 0)),
        "New Years Eve": (datetime.datetime(2020, 12, 31, 0, 0, 0), datetime.datetime(2021, 1, 1, 0, 0, 0)),
        "Individual Event": (datetime.datetime(1, 1, 1, 1, 1, 1), datetime.datetime(1, 1, 1, 1, 1, 1))
    }
    timespan_options = pn.widgets.DatetimeRangeInput(
        name='Individual Event Timespan',
        start=datetime.datetime(2020, 1, 1, 0, 0, 0), end=datetime.datetime(2021, 1, 1, 0, 0, 0),
        value=(datetime.datetime(2020, 1, 1, 0, 0, 0), datetime.datetime(2021, 1, 1, 0, 0, 0)),
        # callback_throttle=3000
    )
    event_options = pn.widgets.Select(name="Event", options=events)
    location_options = pn.widgets.Select(name="Location", options=["pickup", "dropoff"])
    focus_area_options = pn.widgets.Select(
        name="Focus Area", options=["New York", "Manhattan", "Brooklyn", "Bronx", "Queens", "Staten Island"]
    )
    log_checkbox = pn.widgets.Checkbox(name="Log scale")
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
        event=(datetime.datetime(2020, 1, 1, 0, 0, 0), datetime.datetime(2020, 1, 2, 0, 0, 0)),
        timespan=(datetime.datetime(2020, 1, 1, 0, 0, 0), datetime.datetime(2021, 1, 1, 0, 0, 0)),
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
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff
        log_count(bool): Shows data on log scale
        cmap(String): The chosen colormap
        location(String): The focus area
    :returns
        folium.Choropleth: The created choropleth
    :raises
        ValueError: if the aggregation aspect is not 'pickup'/'dropoff'
    """
    if event == (datetime.datetime(1, 1, 1, 1, 1, 1), datetime.datetime(1, 1, 1, 1, 1, 1)):
        event = timespan
    data = _create_aggregator(
        df, event=event, aspect=aspect, log_count=log_count, choropleth=True
    )
    nyc_zones = read_geo_dataset("taxi_zones.geojson")
    nyc_zones["LocationID"] = nyc_zones["LocationID"].astype("str")
    info = ["zone", "LocationID", "borough"]
    columns = []
    if log_count:
        legend_name = f"{aspect} Count (Log Scale)"
        columns.append("PULocationID") if aspect == "pickup" else columns.append(
            "DOLocationID"
        )
        columns.append(f"{aspect}_count_log")

    else:
        legend_name = f"{aspect} Count"
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
    folium.TileLayer('cartodbpositron', name="light mode", control=True).add_to(base_map)
    folium.TileLayer('cartodbdark_matter', name="dark mode", control=True).add_to(base_map)
    folium.TileLayer('stamenterrain', name="stamenterrain", control=True).add_to(base_map)
    folium.TileLayer('openstreetmap', name="openstreetmap", control=True).add_to(base_map)
    folium.LayerControl().add_to(base_map)
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
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff
        log_count(bool): Shows data on log scale
        cmap(String): The chosen colormap
        location(String): The focus area
    :return:
        folium.Choropleth: The created choropleth
    :raises
        ValueError: if the aggregation aspect is not 'pickup'/'dropoff'
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
        legend_name = f"{aspect} Count (Log Scale)"
        columns.append(f"{aspect_abr}LocationID")

    else:
        legend_name = f"{aspect} Count"
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
        data=data,
        columns=columns,
        smooth_factor=0,
        key_on="feature.properties.LocationID",
        fill_opacity=0.8,
        line_opacity=1.0,
    ).add_to(base_map)
    folium.TileLayer('cartodbpositron', name="light mode", control=True).add_to(base_map)
    folium.TileLayer('cartodbdark_matter', name="dark mode", control=True).add_to(base_map)
    folium.TileLayer('stamenterrain', name="stamenterrain", control=True).add_to(base_map)
    folium.TileLayer('openstreetmap', name="openstreetmap", control=True).add_to(base_map)
    folium.LayerControl().add_to(base_map)
    # Display Region Label
    choropleth.geojson.add_child(folium.features.GeoJsonTooltip(info, labels=True))
    highlighted_zone.geojson.add_child(folium.features.GeoJsonTooltip(info, labels=True))
    return base_map


def _create_zone_tab(df):
    """
    This function creates a folium choropleth tab with interactive widgets.
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
    zones = list(df["PULocationID"].unique())
    zones.sort(key=int)
    zone_options = pn.widgets.Select(name="Zone", options=zones)
    direction_options = pn.widgets.Select(name="Direction", options={"Outbound": "outbound", "Inbound": "inbound"})
    focus_area_options = pn.widgets.Select(
        name="Focus Area", options=["New York", "Manhattan", "Brooklyn", "Bronx", "Queens", "Staten Island"]
    )
    log_checkbox = pn.widgets.Checkbox(name="Log scale")
    cmap_option = pn.widgets.Select(name="Color Map", options=cmap)
    dashboard = interact(
        _create_zone_choropleth,
        location=focus_area_options,
        log_count=log_checkbox,
        cmap=cmap_option,
        month=month_options,
        zone=zone_options,
        aspect=direction_options,
        df=fixed(df)
    )
    title = pn.pane.Markdown("""# New York Inbound and Outbound""")

    zone_tab = pn.Column(
        title, pn.Row(dashboard[1], dashboard[0], height=1300, width=1500)
    )
    return zone_tab


def _create_zone_aggregator(
        df, month=None, zone=None, aspect="outbound", choropleth=False, log_count=False):
    """
    This function creates a base folium map.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that should be aggregated
        month(int): Shows the data for this month only
        zone(String): The selected zone.
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff
        choropleth(bool): If True -> the Pickup Location IDs / Drop off Location IDs will also be added for later use
                          in a choropleth.
        log_count(bool): Shows data on log scale
    :return:
        df(pd.DataFrame): Aggregated Data
    :raises:
        ValueError: if the aggregation aspect is not 'pickup'/'dropoff'
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
            df = df.loc[(df[f"{location}_month"] == month) & (df[f"{location_abr}LocationID"] == zone)]
        else:
            df = df.loc[df[f"{location_abr}LocationID"] == zone]
    if choropleth:
        pickup_cols_grp.extend(["PULocationID"])
        dropoff_cols_grp.extend(["DOLocationID"])

    if location == "pickup":
        df_agg = (
            df.groupby(dropoff_cols_grp).size().to_frame(f"{aspect}_count").reset_index()
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


def create_dashboard(df):
    """
    This function creates an interactive panel dashboard.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the interactive.
    :return:
        pn.Tabs: The interactive panel dashboard.
    """

    heatmap_general = ("Heatmap General", _create_general_heatmap_tab(df))
    choropleth_monthly = ("Choropleth Monthly", _create_choropleth_tab(df))
    events = ("Events", _create_events_tab(df))
    zones = ("Zone", _create_zone_tab(df))
    dashboard = pn.Tabs(heatmap_general, choropleth_monthly, events, zones)
    return dashboard
