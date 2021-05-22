from folium.plugins import HeatMap
import branca.colormap
import numpy as np
from collections import defaultdict
import folium
import calendar
from time import strptime
import panel as pn
from panel.interact import interact, fixed

from yellowcab.io.input import read_geo_dataset


def _create_monthly_choropleth(df, month='Jan', aspect='pickup', log_count=False,
                               cmap='YlGn', zoom=11, map_style='cartodbpositron', location="New York"):
    """
    This function creates a folium choropleth based on different aspects of the given data.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the choropleth.
        month(String): The desired month which will be aggregated.
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff
        log_count(bool): Shows data on log scale
        cmap(String): The chosen colormap
        zoom(int): Starting zoom of the choropleth
        map_style(String): Tile layer style of the choropleth map
        location(String): The focus area
    :returns
        folium.Choropleth: The created choropleth
    :raises
        ValueError: if the aggregation aspect is not 'pickup'/'dropoff'
    """
    month = strptime(month, '%b').tm_mon
    data = _create_aggregator(df, month=month, aspect=aspect, log_count=log_count, choropleth=True)
    nyc_zones = read_geo_dataset('taxi_zones.geojson')
    nyc_zones['LocationID'] = nyc_zones['LocationID'].astype('str')
    info = ['zone', 'LocationID', 'borough']
    columns = []
    legend_name = ''
    if log_count:
        legend_name = f'{aspect} Count (Log Scale)'
        columns.append("PULocationID") if aspect == 'pickup' else columns.append("DOLocationID")
        columns.append(f"{aspect}_count_log")

    else:
        legend_name = f'{aspect} Count'
        columns.append("PULocationID") if aspect == 'pickup' else columns.append("DOLocationID")
        columns.append(f"{aspect}_count")
        info.append(f'{aspect}_count')
        if aspect == 'pickup':
            nyc_zones = nyc_zones.merge(data[['PULocationID', 'pickup_count']], how="left", left_on='LocationID',
                                        right_on='PULocationID')
        elif aspect == "dropoff":
            nyc_zones = nyc_zones.merge(data[['DOLocationID', 'dropoff_count']], how="left", left_on='LocationID',
                                        right_on='DOLocationID')
        else:
            raise ValueError("Unknown aggregation aspect")

    base_map = _generate_base_map(default_location=location, map_style=map_style, default_zoom_start=zoom)
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
        line_opacity=0.5
    ).add_to(base_map)
    folium.LayerControl().add_to(base_map)
    # Display Region Label
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(info, labels=True)
    )
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
    folium_tiles = ['cartodbpositron', 'cartodbdark_matter', 'stamenterrain', 'openstreetmap']
    cmap = ['BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 'RdPu',
            'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']
    months = [calendar.month_abbr[i] for i in range(1, 13)]
    month_options = pn.widgets.Select(name='Month', options=months)
    map_options = pn.widgets.Select(name='Tiles', options=folium_tiles)
    location_options = pn.widgets.Select(name='Location', options=['pickup', 'dropoff'])
    focus_area_options = pn.widgets.Select(name='Focus Area', options=['New York', 'Manhattan'])
    zoom_option = pn.widgets.IntSlider(name='Zoom Factor', start=5, end=20, step=1, value=10)
    log_checkbox = pn.widgets.Checkbox(name='Log scale')
    cmap_option = pn.widgets.Select(name='Color Map', options=cmap)
    dashboard = (interact(_create_monthly_choropleth, location=focus_area_options,
                          map_style=map_options,
                          zoom=zoom_option,
                          log_count=log_checkbox,
                          cmap=cmap_option,
                          month=month_options,
                          aspect=location_options,
                          df=fixed(df)))
    title = pn.pane.Markdown('''# New York Monthly Trip Choropleth''')

    monthly_choropleth_tab = pn.Column(title, pn.Row(dashboard[1], dashboard[0], height=1300, width=1500))
    return monthly_choropleth_tab


def _generate_base_map(default_location="New York", map_style='cartodbpositron', default_zoom_start=10):
    """
    This function creates a base folium map.
    ----------------------------------------------
    :param
        default_location(String): The default location of the base map.
        map_style(String): The default map style for the base map.
        default_zoom_start(int): The starting zoom for the base map.
    :return:
        folium.Map: the created base map
    """
    locations = {"New York": [40.693943, -73.985880],
                 "Manhattan": [40.754932, -73.984016]}

    if default_location is not None and default_location not in locations:
        raise ValueError("Could not find the given location coordinates.")

    if default_location == "Manhattan":
        default_zoom_start = 12

    default_location = locations.get(default_location)

    base_map = folium.Map(location=default_location, tiles=map_style, control_scale=True, zoom_start=default_zoom_start)
    return base_map


def _create_aggregator(df, month=None, aspect='pickup', choropleth=False, log_count=False):
    """
    This function creates a base folium map.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that should be aggregated
        month(int): Shows the data for this month only
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff
        choropleth(bool): If True -> the Pickup Location IDs / Drop off Location IDs will also be added for later use
                          in a choropleth.
        log_count(bool): Shows data on log scale

    :return:
        df(pd.DataFrame): Aggregated Data
    :raises:
        ValueError: if the aggregation aspect is not 'pickup'/'dropoff'
    """
    pickup_cols_grp = ['centers_lat_pickup', 'centers_long_pickup']
    dropoff_cols_grp = ['centers_lat_dropoff', 'centers_long_dropoff']
    if month is not None:
        df = df.loc[df[f'{aspect}_month'] == month]
    if choropleth:
        pickup_cols_grp.extend(['PULocationID'])
        dropoff_cols_grp.extend(['DOLocationID'])

    if aspect == "pickup":
        df_agg = df.groupby(pickup_cols_grp).size().to_frame('pickup_count').reset_index()
    elif aspect == "dropoff":
        df_agg = df.groupby(dropoff_cols_grp).size().to_frame('dropoff_count').reset_index()
    else:
        raise ValueError("Unknown aggregation aspect")

    if log_count:
        df_agg[f'{aspect}_count_log'] = np.log(df_agg[f'{aspect}_count'])
        df_agg.pop(f'{aspect}_count')
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


def _create_heat_map(df, aspect='pickup', radius=15, zoom=11, map_style='cartodbpositron', location="New York",
                     log_count=False, inferno_colormap=False):
    """
    This function creates a folium heatmap based on different aspects of the given data.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the heatmap.
        aspect(String): Aggregates data based on given aspect. Allowed values are pickup or dropoff.
        radius(int): The observed radius for each location upon creating the heatmap.
        log_count(bool): Shows data on log scale.
        inferno_colormap(bool): If True -> changes the default cmap to inferno cmap
        zoom(int): Starting zoom of the heatmap
        map_style(String): Tile layer style of the heatmap
        location(String): The focus area
    :returns
        folium.Heatmap: The created Heatmap
    """
    base_map = _generate_base_map(default_location=location, map_style=map_style, default_zoom_start=zoom)
    map_data = df
    if log_count:
        map_data = _create_aggregator(df, aspect=aspect, log_count=True).values.tolist()
    else:
        map_data = _create_aggregator(df, aspect=aspect).values.tolist()

    if inferno_colormap:
        inferno_colormap, inferno_gradient = _create_inferno_cmap()
        HeatMap(data=map_data, radius=radius, max_zoom=zoom, gradient=inferno_gradient).add_to(base_map)
        inferno_colormap.add_to(base_map)
    else:
        HeatMap(data=map_data, radius=radius, max_zoom=zoom).add_to(base_map)

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
    folium_tiles = ['cartodbpositron', 'cartodbdark_matter', 'stamenterrain', 'openstreetmap']
    map_options = pn.widgets.Select(name='Tiles', options=folium_tiles)
    location_options = pn.widgets.Select(name='Location', options=['pickup', 'dropoff'])
    focus_area_options = pn.widgets.Select(name='Focus Area', options=['New York', 'Manhattan'])
    zoom_option = pn.widgets.IntSlider(name='Zoom Factor', start=5, end=20, step=1, value=10)
    radius_option = pn.widgets.IntSlider(name='Heatmap Radius', start=5, end=20, step=1, value=8)
    log_checkbox = pn.widgets.Checkbox(name='Log scale')
    inferno_checkbox = pn.widgets.Checkbox(name='Inferno colormap')
    dashboard = (interact(_create_heat_map, location=focus_area_options,
                          map_style=map_options,
                          zoom=zoom_option,
                          radius=radius_option,
                          log_count=log_checkbox,
                          inferno_colormap=inferno_checkbox,
                          aspect=location_options,
                          df=fixed(df)))
    title = pn.pane.Markdown('''# New York Trip Heatmap''')

    general_heatmap_tab = pn.Column(title, pn.Row(dashboard[1], dashboard[0], height=1300, width=1500))
    return general_heatmap_tab


def create_dashboard(df):
    """
    This function creates an interactive panel dashboard.
    ----------------------------------------------
    :param
        df(pd.DataFrame): Data that is used to make the interactive.
    :return:
        pn.Tabs: The interactive panel dashboard.
    """

    heatmap_general = ('Heatmap General', _create_general_heatmap_tab(df))
    choropleth_monthly = ('Choropleth Monthly', _create_choropleth_tab(df))
    dashboard = pn.Tabs(heatmap_general, choropleth_monthly)
    return dashboard
