import os
import re
import json
import time
import socket
import tempfile
import folium
import ipysheet
import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray as rxr
import geopandas as gpd
import branca.colormap as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ipywidgets as widgets

import ipyleaflet
from glob import glob
from pathlib import Path
from folium import plugins
from pyproj import Transformer
from ipyfilechooser import FileChooser
from ipysheet import from_dataframe
from rasterio.warp import calculate_default_transform, reproject, Resampling
from localtileserver import TileClient, get_leaflet_tile_layer, examples
from ipyleaflet import WidgetControl
from ipyfilechooser import FileChooser
from IPython.display import display
from ipyleaflet import (
    Map,
    FullScreenControl,
    LayersControl,
    DrawControl,
    MeasureControl,
    ScaleControl,
    Marker,
    basemaps,
    AwesomeIcon,
    LegendControl,
    MarkerCluster,
    WidgetControl
)


os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = \
    f"{os.environ['JUPYTERHUB_SERVICE_PREFIX'].lstrip('/')}/proxy/{{port}}"
chunks = {"band": 1, "x": 2048, "y": 2048}

import localtileserver
from localtileserver import get_folium_tile_layer, TileClient

def add_raster(
            m,
            in_raster,
            style=None,
            data_bands=[5, 7, 2],
            cmap=None,
            layer_name="Raster"
        ):
    """
    Adds a raster layer to the map.
    Args:
        in_shp (str): The file path to the input shapefile.
        style (dict, optional): The style dictionary. Defaults to None.
        layer_name (str, optional): The layer name for the raster layer.
    TODO:
        - widget for bands to choose to show
        - bands as a parameter
        - delete raster layer
    """
    raster_client = TileClient(in_raster)
    m.add_layer(
        get_leaflet_tile_layer(
            raster_client, show=False, band=data_bands,
            cmap=cmap, name=layer_name))
    m.center = raster_client.center()
    m.zoom = raster_client.default_zoom

    
def add_markers(m, points_df, cmap, classes, icons):
    """
    Add markers to the map
    """
    markers_list = []
    for index, point in points_df.iterrows():
        
        coordinates = (point['geometry'].y, point['geometry'].x)
        type_color = cmap[point['predicted']]
        type_pred = classes[point['predicted']]

        radio_pred_widget = widgets.RadioButtons(
            options=classes,
            value=type_pred, # Defaults to 'pineapple'
            layout={'width': 'max-content'}, # If the items' names are long
            description='Prediction:',
            disabled=True
        )

        radio_check_widget = widgets.RadioButtons(
            options=classes,
            value=classes[0], # Defaults to first class
            layout={'width': 'max-content'}, # If the items' names are long
            description='Validation:',
            disabled=False
        )
        point_id_widget = widgets.IntText(
            value=index,
            description='ID:',
            disabled=True
        )
        checked_widget = widgets.Checkbox(
            value=False,
            description='Verified',
            disabled=False
        ) 
        popup = widgets.VBox([
            point_id_widget, radio_pred_widget,
            radio_check_widget, checked_widget
        ])
        
        marker = Marker(
            name=str(index),
            location=coordinates,
            draggable=False,
            keyboard=True,
            icon=AwesomeIcon(
                name=icons[point['predicted']],
                marker_color=type_color,
                icon_color=type_color,
            ),
            popup=popup
        )
        
        # assign value to ipysheet
        cell = ipysheet.cell(index, 2, type_pred)
        widgets.jslink((cell, 'value'), (radio_pred_widget, 'value'))
        widgets.jslink((radio_pred_widget, 'value'), (cell, 'value'))
        cell = ipysheet.cell(index, 3, 'other')
        widgets.jslink((cell, 'value'), (radio_check_widget, 'value'))
        widgets.jslink((radio_check_widget, 'value'), (cell, 'value'))
        cell = ipysheet.cell(index, 4, False)#, choice=)
        widgets.jslink((cell, 'value'), (checked_widget, 'value'))
        
        # append to group of markers
        markers_list.append(marker)

    marker_cluster = MarkerCluster(
        markers=tuple(markers_list),
        name="validation"
    )
    # marker_cluster.on_click(handle_click)

    m.add_layer(marker_cluster)


def gen_points(in_raster, n_points):
    """
    Generate random points over raster mask.
    """
    # read prediction raster
    raster_prediction = rxr.open_rasterio(
        in_raster, chunks=chunks)
    raster_prediction.name = "predicted"
    raster_crs = raster_prediction.rio.crs

    # convert to dataframe and filter no-data
    raster_prediction = \
        raster_prediction.squeeze().to_dataframe().reset_index()

    # drop some unnecesary bands
    raster_prediction = raster_prediction.drop(
        ['band', 'spatial_ref'], axis=1)

    # filter no-data from dataframe
    raster_prediction = raster_prediction[
        raster_prediction['predicted'] >= 0]
    raster_prediction = raster_prediction[
        raster_prediction['predicted'] < 255]

    # change type to integer
    raster_prediction = raster_prediction.astype(
        {'predicted': 'int'})

    # create random points
    unique_counts = raster_prediction['predicted'].value_counts()
    for class_id, class_count in unique_counts.iteritems():
        raster_prediction = raster_prediction.drop(
            raster_prediction[
                raster_prediction['predicted'] == class_id].sample(
                n=class_count - n_points).index
        )

    # convert to geodataframe
    geometry = gpd.points_from_xy(
        raster_prediction.x, raster_prediction.y)

    # add geometry and crs to dataframe
    raster_prediction = gpd.GeoDataFrame(
        raster_prediction,
        crs=raster_crs,
        geometry=geometry).reset_index(drop=True)

    # dataframe to match data_client crs
    raster_prediction = raster_prediction.to_crs(4326)

    # add columns for validation process
    raster_prediction['operator'] = 0
    raster_prediction['verified'] = 'false'

    return raster_prediction

def main_toolbar(m, sheet):

    padding = "0px 0px 0px 5px"  # upper, right, bottom, left

    # button to open the toolbar - done
    toolbar_button = widgets.ToggleButton(
        value=False,
        tooltip="Toolbar",
        icon="wrench",
        layout=widgets.Layout(width="28px", height="28px", padding=padding),
    )

    # button to close the toolbar widget - done
    close_button = widgets.ToggleButton(
        value=False,
        tooltip="Close the tool",
        icon="times",
        button_style="primary",
        layout=widgets.Layout(height="28px", width="28px", padding=padding),
    )

    # locate the toolbar on top - done
    toolbar = widgets.HBox([toolbar_button])

    # action to close toolbar - done
    def close_click(change):
        if change["new"]:
            toolbar_button.close()
            close_button.close()
            toolbar.close()

    # observer option to close toolbar - done
    close_button.observe(close_click, "value")

    # creating grid from inside toolbar
    rows = 2
    cols = 2
    grid = widgets.GridspecLayout(
        rows, cols, grid_gap="0px", layout=widgets.Layout(width="62px")
    )
    icons = ["folder-open", "map-marker", "map", "gears"]

    # add the options to the grid
    for i in range(rows):
        for j in range(cols):
            grid[i, j] = widgets.Button(
                description="",
                button_style="primary",
                icon=icons[i * rows + j],
                layout=widgets.Layout(width="28px", padding="0px"),
            )

    # add the full toolbar
    toolbar = widgets.VBox([toolbar_button])

    # observe the tool bar on click
    def toolbar_click(change):
        if change["new"]:
            toolbar.children = [widgets.HBox([close_button, toolbar_button]), grid]
        else:
            toolbar.children = [toolbar_button]

    # observer routing for toolbar
    toolbar_button.observe(toolbar_click, "value")

    # define the toolbar control
    toolbar_ctrl = WidgetControl(widget=toolbar, position="topright")

    # add the toolbar control to the map
    m.add_control(toolbar_ctrl)

    # define the output control from the widget
    output = widgets.Output()
    output_ctrl = WidgetControl(widget=output, position="topright")

    buttons = widgets.ToggleButtons(
        value=None,
        options=["Apply", "Reset", "Close"],
        tooltips=["Apply", "Reset", "Close"],
        button_style="primary",
    )
    buttons.style.button_width = "80px"

    # data_dir = os.path.abspath("./data")
    data_dir = os.path.abspath(m.data_dir)
    mask_dir = os.path.abspath(m.mask_dir)

    # data chooser
    fc = FileChooser(data_dir)
    fc.use_dir_icons = True
    fc.filter_pattern = ["*.tif", "*.gpkg", "*.shp", "*.geojson"]
    
    bands = [
        ('Coastal Blue', 1), ('Blue', 2), ('Green', 3),  ('Yellow', 4),
        ('Red', 5), ('Red Edge', 6), ('NIR1', 7), ('NIR2', 8)]
    
    # bands chooser
    red = widgets.Dropdown(
        options=bands,
        value=5,
        description='Red:',
    )
    green = widgets.Dropdown(
        options=bands,
        value=7,
        description='Green:',
    )
    blue = widgets.Dropdown(
        options=bands,
        value=2,
        description='Blue:',
    )

    bands_widget = widgets.HBox([red, green, blue])
    filechooser_widget = widgets.VBox([fc, bands_widget, buttons])

    def button_click(change):
        if change["new"] == "Apply" and fc.selected is not None:
            if fc.selected.endswith(".tif"):
                add_raster(
                    m, fc.selected, data_bands=[red.value, green.value, blue.value], layer_name="Raster")
            elif fc.selected.endswith(".shp"):
                m.add_shapefile(fc.selected, layer_name="Shapefile")
            elif fc.selected.endswith(".geojson"):
                m.add_geojson(fc.selected, layer_name="GeoJSON")
        elif change["new"] == "Reset":
            fc.reset()
        elif change["new"] == "Close":
            fc.reset()
            m.remove_control(output_ctrl)
            buttons.value = None

    buttons.observe(button_click, "value")

    def tool_click(b):
        with output:
            output.clear_output()
            
            # icon #1, folder-open to load data array
            if b.icon == "folder-open":
                display(filechooser_widget)
                m.add_control(output_ctrl)
                
            # icon #2, map-marker tool to generate dataframe
            elif b.icon == "map-marker":
                
                fc = FileChooser(mask_dir)
                fc.use_dir_icons = True
                fc.filter_pattern = ["*.tif", ".gpkg"]

                x_widget = widgets.Dropdown(
                    description="X:",
                    layout=widgets.Layout(width="122px", padding="0px"),
                    style={"description_width": "initial"},
                )
                y_widget = widgets.Dropdown(
                    description="Y:",
                    layout=widgets.Layout(width="122px", padding="0px"),
                    style={"description_width": "initial"},
                )
                
                n_points_widget = widgets.IntSlider(
                        value=50,
                        min=0,
                        max=200,
                        step=1,
                        description='Points per Class:',
                        disabled=False,
                        continuous_update=False,
                        orientation='horizontal',
                        readout=True,
                        readout_format='d'
                )

                label_widget = widgets.Dropdown(
                    description="Label:",
                    layout=widgets.Layout(width="248px", padding="0px"),
                    style={"description_width": "initial"},
                )

                layer_widget = widgets.Text(
                    description="Layer name: ",
                    value="Marker cluster",
                    layout=widgets.Layout(width="248px", padding="0px"),
                    style={"description_width": "initial"},
                )

                btns = widgets.ToggleButtons(
                    value=None,
                    options=["Read data", "Display", "Close"],
                    tooltips=["Read data", "Display", "Close"],
                    button_style="primary",
                )
                btns.style.button_width = "80px"

                def btn_click(change):
                    if change["new"] == "Read data" and fc.selected is not None:
                        
                        #import pandas as pd

                        #df = pd.read_csv(fc.selected)
                        #col_names = df.columns.values.tolist()
                        #x_widget.options = col_names
                        #y_widget.options = col_names
                        #label_widget.options = col_names

                        #if "longitude" in col_names:
                        #    x_widget.value = "longitude"

                        #if "latitude" in col_names:
                        #    y_widget.value = "latitude"

                        #if "name" in col_names:
                        #    label_widget.value = "name"
                        
                        cmap = ['lightgray', 'green', 'orange', 'red']
                        cmap_hex = [mcolors.rgb2hex(cmap[i]) for i in range(len(cmap))]
                        classes = ['other', 'tree', 'crop', 'burn']
                        icons = ['cog', 'tree', 'crop', 'fire']

                        # add mask raster to map
                        add_raster(
                            m,
                            fc.selected,
                            data_bands=[1],
                            layer_name='Label', #label_mask_widget.value,
                            cmap=cmap_hex
                        )
                        
                        # generate random points
                        random_points_df = gen_points(
                            fc.selected, n_points_widget.value)
                        
                        sheet = ipysheet.sheet(
                            ipysheet.from_dataframe(
                                random_points_df.to_crs(4326).drop(
                                    ['geometry'], axis=1)
                        ))
                        widgets.Dropdown.value.tag(sync=True)
                        display(sheet)
                        
                        # show markers
                        add_markers(m, random_points_df, cmap, classes, icons)

                    elif change["new"] == "Display":

                        if x_widget.value is not None and (y_widget.value is not None):
                            m.add_points_from_csv(
                                fc.selected,
                                x=x_widget.value,
                                y=y_widget.value,
                                label=label_widget.value,
                                layer_name=layer_widget.value,
                            )

                    elif change["new"] == "Close":
                        fc.reset()
                        m.remove_control(output_ctrl)

                btns.observe(btn_click, "value")

                csv_widget = widgets.VBox(
                    [
                        fc,
                        widgets.HBox([x_widget, y_widget]),
                        n_points_widget,
                        label_widget,
                        layer_widget,
                        btns,
                    ]
                )

                display(csv_widget)
                m.add_control(output_ctrl)

            # icon #3 tool box from whitebox
            elif b.icon == "gears":
                import whiteboxgui.whiteboxgui as wbt

                if hasattr(m, "whitebox") and m.whitebox is not None:
                    if m.whitebox in m.controls:
                        m.remove_control(m.whitebox)

                tools_dict = wbt.get_wbt_dict()
                wbt_toolbox = wbt.build_toolbox(
                    tools_dict, max_width="800px", max_height="500px"
                )

                wbt_control = WidgetControl(widget=wbt_toolbox, position="bottomright")

                m.whitebox = wbt_control
                m.add_control(wbt_control)
            
            # icon #3 tool box from whitebox
            #elif b.icon == "map":
                

    for i in range(rows):
        for j in range(cols):
            tool = grid[i, j]
            tool.on_click(tool_click)


def gen_map(widget_dialog):
    
    data_dialog = widget_dialog.children[0]

    data_bands = []
    for item in widget_dialog.children[1].children:
        data_bands.append(item.value)

    label_dialog = widget_dialog.children[2]
    val_points_per_class = widget_dialog.children[3].value
    
    cmap = []
    for item in widget_dialog.children[4].children[0].children:
        cmap.append(item.value)
    
    classes = []
    for item in widget_dialog.children[4].children[1].children:
        classes.append(item.value)
    
    tiles_basemap: str = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
    icons: list = ['cog', 'tree', 'wheat', 'fire']

    # read prediction raster
    raster_prediction = rxr.open_rasterio(
        label_dialog.selected, chunks={"band": 1, "x": 2048, "y": 2048})
    raster_prediction.name = "predicted"
    raster_crs = raster_prediction.rio.crs

    # convert to dataframe and filter no-data
    raster_prediction = raster_prediction.squeeze().to_dataframe().reset_index()  # convert array to dataframe
    raster_prediction = raster_prediction.drop(['band', 'spatial_ref'], axis=1)  # drop some unecessary columns
    raster_prediction = raster_prediction[raster_prediction['predicted'] >= 0]  # only select appropiate values, remove no-data
    raster_prediction = raster_prediction.astype({'predicted': 'int'})  # convert mask into int

    # create random points
    unique_counts = raster_prediction['predicted'].value_counts()
    for class_id, class_count in unique_counts.iteritems():
        raster_prediction = raster_prediction.drop(
            raster_prediction[raster_prediction['predicted'] == class_id].sample(
                n=class_count - val_points_per_class).index
        )

    geometry = gpd.points_from_xy(raster_prediction.x, raster_prediction.y)
    raster_prediction = gpd.GeoDataFrame(raster_prediction, crs=raster_crs, geometry=geometry).reset_index(drop=True)
    
    # Client - initial client to localize zoom
    color_list = [mcolors.rgb2hex(cmap[i]) for i in range(len(cmap))]
    data_client = TileClient(data_dialog.selected)
    
    # conversion to 16bit
    label_raster = rxr.open_rasterio(label_dialog.selected).astype(np.int16)
    label_raster = label_raster.rio.write_nodata(-10001, encoded=True, inplace=True)
    label_filename = Path(label_dialog.selected)
    label_output_filename = os.path.join(
        label_filename.parent.absolute(),
        f'{label_filename.stem}-16.tif'
    )
    label_raster.rio.to_raster(
        label_output_filename,
        BIGTIFF="IF_SAFER",
        compress='LZW',
        driver='GTiff',
        dtype='int16'
    )

    #label_client = TileClient(label_dialog.selected)
    label_client = TileClient(label_output_filename)

    # dataframe to match data_client crs
    raster_prediction = raster_prediction.to_crs(4326)#(data_client.default_projection).split(':')[-1])
    raster_prediction['operator'] = 0
    raster_prediction['verified'] = 'false'

    # Create ipyleaflet TileLayer from that server
    data_layer = get_leaflet_tile_layer(
        data_client, show=False, band=data_bands, name="data")

    label_layer = get_leaflet_tile_layer(
        label_client, show=False, cmap=color_list, name="label")

    # Create ipyleaflet map, add tile layer, and display
    m = Map(
        center=data_client.center(),
        zoom=data_client.default_zoom,
        basemap=basemaps.Esri.WorldImagery,
        scroll_wheel_zoom=True,
        keyboard=True
    )
    m.add_layer(data_layer)
    m.add_layer(label_layer)

    validation_sheet = ipysheet.sheet(from_dataframe(
        raster_prediction.to_crs(4326).drop(['geometry'], axis=1)
    ))

    widgets.Dropdown.value.tag(sync=True)

    # Iterate through list and add a marker
    markers_list = []
    for index, point in raster_prediction.iterrows():

        coordinates = (point['geometry'].y, point['geometry'].x)
        type_color = cmap[point['predicted']]
        type_pred = classes[point['predicted']]

        radio_pred_widget = widgets.RadioButtons(
            options=classes,
            value=type_pred, # Defaults to 'pineapple'
            layout={'width': 'max-content'}, # If the items' names are long
            description='Prediction:',
            disabled=True
        )

        radio_check_widget = widgets.RadioButtons(
            options=classes,
            value=classes[0], # Defaults to 'pineapple'
            layout={'width': 'max-content'}, # If the items' names are long
            description='Validation:',
            disabled=False
        )
        point_id_widget = widgets.IntText(
            value=index,
            description='ID:',
            disabled=True
        )
        checked_widget = widgets.Checkbox(
            value=False,
            description='Verified',
            disabled=False
        ) 
        popup = widgets.VBox([
            point_id_widget, radio_pred_widget,
            radio_check_widget, checked_widget
        ])

        marker = Marker(
            name=str(index),
            location=coordinates,
            draggable=False,
            keyboard=True,
            icon=AwesomeIcon(
                name=icons[point['predicted']],
                marker_color=type_color,
                icon_color=type_color,
                # spin=True
            ),
            popup=popup
        )

        cell = ipysheet.cell(index, 2, type_pred)
        widgets.jslink((cell, 'value'), (radio_pred_widget, 'value'))
        widgets.jslink((radio_pred_widget, 'value'), (cell, 'value'))
        cell = ipysheet.cell(index, 3, 'other')
        widgets.jslink((cell, 'value'), (radio_check_widget, 'value'))
        widgets.jslink((radio_check_widget, 'value'), (cell, 'value'))
        cell = ipysheet.cell(index, 4, False)#, choice=)
        widgets.jslink((cell, 'value'), (checked_widget, 'value'))

        # append to group of markers
        markers_list.append(marker)

    marker_cluster = MarkerCluster(
        markers=tuple(markers_list),
        name="validation"
    )
    # marker_cluster.on_click(handle_click)

    m.add_layer(marker_cluster);
    m.add_control(ScaleControl(position='bottomleft'))
    m.add_control(LayersControl(position='topright'))
    m.add_control(FullScreenControl())

    display(m)
    display(validation_sheet)
    return m, validation_sheet

def file_chooser_widget(data_dir=None, mask_dir=None):
    
    hostname = socket.gethostname()
    
    if socket.gethostname()[:3] == 'gpu':
        data_dir = '/explore/nobackup/projects/3sl/data/Tappan'
        mask_dir = '/explore/nobackup/projects/ilab/projects/' + \
            'Senegal/3sl/products/land_cover/dev/trees.v2/Tappan'
    else:
        data_dir = '/home/jovyan/efs/projects/3sl/data/Tappan'
        mask_dir = '/home/jovyan/efs/projects/3sl/products/otcb.v1/Tappan'

    data_dir = os.path.abspath(data_dir)
    mask_dir = os.path.abspath(mask_dir)

    data_dialog = FileChooser(
        data_dir,
        filename='Tappan01_WV02_20110430_M1BS_103001000A27E100_data.tif',
        title='<b>Select raster filename:</b>',
        filter_pattern='*.tif',
        show_hidden=False,
        select_default=True,
        show_only_dirs=False,
        use_dir_icons=True
    )
    mask_dialog = FileChooser(
        mask_dir,
        filename='Tappan01_WV02_20110430_M1BS_103001000A27E100_data.otcb.tif',
        title='<b>Select prediction filename:</b>',
        filter_pattern='*.tif',
        show_hidden=False,
        select_default=True,
        show_only_dirs=False,
        use_dir_icons=True
    )

    buttons = widgets.ToggleButtons(
        value=None,
        options=["Apply", "Reset"],
        tooltips=["Apply", "Reset"],
        button_style="primary",
    )
    buttons.style.button_width = "80px"
    
    bands = [
        ('Coastal Blue', 1), ('Blue', 2), ('Green', 3),  ('Yellow', 4),
        ('Red', 5), ('Red Edge', 6), ('NIR1', 7), ('NIR2', 8)]
    
    # bands chooser
    red = widgets.Dropdown(
        options=bands,
        value=5,
        description='Red:',
    )
    green = widgets.Dropdown(
        options=bands,
        value=7,
        description='Green:',
    )
    blue = widgets.Dropdown(
        options=bands,
        value=2,
        description='Blue:',
    )

    bands_widget = widgets.HBox([red, green, blue])

    """
    n_points_widget = widgets.IntSlider(
            value=50,
            min=0,
            max=200,
            step=1,
            description='Points per Class:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout={'width': 'max-content'},
            align_items='stretch', 
            style= {'description_width': 'initial'}
    )
    """
    
    n_points_widget =  widgets.BoundedIntText(
        value=50,
        min=0,
        max=200,
        step=1,
        description='Points per Class:',
        disabled=False,
        layout={'width': 'max-content'},
        align_items='stretch', 
        style= {'description_width': 'initial'}
    )
    
    # colors chooser
    colors_widget_list = []
    cmap = ['lightgray', 'green', 'orange', 'red']
    for index, color in enumerate(cmap):
        colors_widget_list.append(
            widgets.Dropdown(
                options=cmap,
                value=color,
                description=f'Class {index}:',
            )
        )

    # classes chooser
    classes_widget_list = []
    classes = ['other', 'tree', 'crop', 'burn']
    for index, c in enumerate(classes):
        classes_widget_list.append(
            widgets.Dropdown(
                options=classes,
                value=c,
                description=f'Class {index}:',
            )
        )

    colors_class_widget = widgets.HBox([
        widgets.VBox(colors_widget_list),
         widgets.VBox(classes_widget_list),
    ])

    filechooser_widget = widgets.VBox([
        data_dialog, bands_widget, mask_dialog, n_points_widget,
        colors_class_widget
    ])

    display(filechooser_widget)    
    return filechooser_widget


def save_file(filename_widget, validation_sheet):
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    hostname = socket.gethostname()
    
    if hostname[:3] == 'gpu':
        username = os.environ['USER']
        output_dir = '/explore/nobackup/projects/3sl/development/scratch'
    else:
        username = hostname.split('-')[-1]
        output_dir = '/home/jovyan/efs/projects/3sl/validation'
    
    # get filename
    filename = Path(filename_widget.children[2].selected).stem

    # full output_dir
    full_dirs = filename_widget.children[2].selected.split('/')
    output_dir = os.path.join(output_dir, username, full_dirs[-3], full_dirs[-2])
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.join(output_dir, f"{timestr}-{username}-{filename}.gpkg")

    saved_sheet = ipysheet.to_dataframe(validation_sheet)
    gdf = gpd.GeoDataFrame(
        saved_sheet, geometry=gpd.points_from_xy(saved_sheet.y, saved_sheet.x))
    gdf.to_file(output_filename, layer='validation', driver="GPKG")
    return