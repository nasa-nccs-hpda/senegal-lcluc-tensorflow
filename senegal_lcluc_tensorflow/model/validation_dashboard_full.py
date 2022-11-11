import ipyleaflet
from ipyleaflet import (
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

import os
import ipysheet
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import ipywidgets as widgets
import whiteboxgui.whiteboxgui as wbt
import matplotlib.colors as mcolors
from ipyfilechooser import FileChooser
from IPython.display import display

os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = \
    f"{os.environ['JUPYTERHUB_SERVICE_PREFIX'].lstrip('/')}/proxy/{{port}}"

from localtileserver import get_leaflet_tile_layer, TileClient

class ValidationDashboard():
    
    def __init__(self, **kwargs):
        
        self.val_map = ValidationDashboardMap()
        display(self.val_map)

class ValidationDashboardMap(ipyleaflet.Map):
    """
    This Map class inherits the ipyleaflet Map class.
    Modified: https://github.com/giswqs/geodemo/blob/master/geodemo/geodemo.py
    Args:
        ipyleaflet (ipyleaflet.Map): An ipyleaflet map.
    """

    def __init__(self, **kwargs):

        if "center" not in kwargs:
            kwargs["center"] = [40, -100]

        if "zoom" not in kwargs:
            kwargs["zoom"] = 4

        if "scroll_wheel_zoom" not in kwargs:
            kwargs["scroll_wheel_zoom"] = True

        if "keyboard" not in kwargs:
            kwargs["keyboard"] = True

        if "basemap" not in kwargs:
            kwargs["basemap"] = basemaps.Esri.WorldImagery

        super().__init__(**kwargs)

        if "height" not in kwargs:
            self.layout.height = "600px"
        else:
            self.layout.height = kwargs["height"]

        self.add_control(FullScreenControl())
        self.add_control(LayersControl(position="topright"))
        self.add_control(DrawControl(position="topleft"))
        self.add_control(MeasureControl())
        self.add_control(ScaleControl(position="bottomleft"))

        if "data_dir" not in kwargs:
            self.data_dir = '/explore/nobackup/projects/3sl/data/Tappan'
        else:
            self.data_dir = kwargs["data_dir"]
        assert os.path.exists(self.data_dir), f'{self.data_dir} does not exist'

        if "mask_dir" not in kwargs:
            self.mask_dir = '/explore/nobackup/projects/ilab/projects/' + \
                'Senegal/3sl/products/land_cover/dev/trees.v2/Tappan'
        assert os.path.exists(self.mask_dir), f'{self.mask_dir} does not exist'

        if "output_dir" not in kwargs:
            self.output_dir = \
                '/explore/nobackup/projects/3sl/development/scratch'
        else:
            self.output_dir = kwargs["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        if "chunks" not in kwargs:
            self.chunks = {"band": 1, "x": 2048, "y": 2048}
        else:
            self.chunks = kwargs["chunks"]

        self.validation_sheet = None
        self.random_points = None

        self._main_toolbar()

    def add_raster(
                self,
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
        self.add_layer(
            get_leaflet_tile_layer(
                raster_client, show=False, band=data_bands,
                cmap=cmap, name=layer_name))
        self.center = raster_client.center()
        self.zoom = raster_client.default_zoom

        # read prediction raster
        raster_prediction = rxr.open_rasterio(
            in_raster, chunks=self.chunks)
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
                    n=class_count - 100).index
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

        cmap = ['lightgray', 'green', 'orange', 'red']
        classes = ['other', 'tree', 'crop', 'burn']
        cmap = [mcolors.rgb2hex(cmap[i]) for i in range(len(cmap))]
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

            #cell = ipysheet.cell(index, 2, type_pred)
            #widgets.jslink((cell, 'value'), (radio_pred_widget, 'value'))
            #widgets.jslink((radio_pred_widget, 'value'), (cell, 'value'))
            #cell = ipysheet.cell(index, 3, 'other')
            #widgets.jslink((cell, 'value'), (radio_check_widget, 'value'))
            #widgets.jslink((radio_check_widget, 'value'), (cell, 'value'))
            #cell = ipysheet.cell(index, 4, False)#, choice=)
            #widgets.jslink((cell, 'value'), (checked_widget, 'value'))

            # append to group of markers
            markers_list.append(marker)

        marker_cluster = MarkerCluster(
            markers=tuple(markers_list),
            name="validation"
        )
        # marker_cluster.on_click(handle_click)

        self.add_layer(marker_cluster)


    def add_markers(self, df):
        print(df)
        """
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
        """
        return

    def gen_points(self, in_raster, n_points):
        """
        Generate random points over raster mask.
        """
        print(in_raster, n_points)
        
        # read prediction raster
        raster_prediction = rxr.open_rasterio(
            in_raster, chunks=self.chunks)
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
        
        gpd.to_csv(raster_prediction, 'out.gpkg')

        cmap = ['lightgray', 'green', 'orange', 'red']
        classes = ['other', 'tree', 'crop', 'burn']
        cmap = [mcolors.rgb2hex(cmap[i]) for i in range(len(cmap))]
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

            #cell = ipysheet.cell(index, 2, type_pred)
            #widgets.jslink((cell, 'value'), (radio_pred_widget, 'value'))
            #widgets.jslink((radio_pred_widget, 'value'), (cell, 'value'))
            #cell = ipysheet.cell(index, 3, 'other')
            #widgets.jslink((cell, 'value'), (radio_check_widget, 'value'))
            #widgets.jslink((radio_check_widget, 'value'), (cell, 'value'))
            #cell = ipysheet.cell(index, 4, False)#, choice=)
            #widgets.jslink((cell, 'value'), (checked_widget, 'value'))

            # append to group of markers
            markers_list.append(marker)

        marker_cluster = MarkerCluster(
            markers=tuple(markers_list),
            name="validation"
        )
        # marker_cluster.on_click(handle_click)

        self.add_layer(marker_cluster)

    #return raster_prediction

    # def df_to_gpkg()
    # def gpkg_to_df()

    def _main_toolbar(self):

        padding = "0px 0px 0px 5px"  # upper, right, bottom, left

        toolbar_button = widgets.ToggleButton(
            value=False,
            tooltip="Toolbar",
            icon="wrench",
            layout=widgets.Layout(
                width="28px", height="28px", padding=padding),
        )

        close_button = widgets.ToggleButton(
            value=False,
            tooltip="Close the tool",
            icon="times",
            button_style="primary",
            layout=widgets.Layout(
                height="28px", width="28px", padding=padding),
        )

        toolbar = widgets.HBox([toolbar_button])

        def close_click(change):
            if change["new"]:
                toolbar_button.close()
                close_button.close()
                toolbar.close()

        close_button.observe(close_click, "value")

        rows = 2
        cols = 2
        grid = widgets.GridspecLayout(
            rows, cols, grid_gap="0px", layout=widgets.Layout(width="62px")
        )

        icons = ["folder-open", "map-marker", "map", "gears"]

        for i in range(rows):
            for j in range(cols):
                grid[i, j] = widgets.Button(
                    description="",
                    button_style="primary",
                    icon=icons[i * rows + j],
                    layout=widgets.Layout(width="28px", padding="0px"),
                )

        toolbar = widgets.VBox([toolbar_button])

        def toolbar_click(change):
            if change["new"]:
                toolbar.children = [
                    widgets.HBox([close_button, toolbar_button]), grid]
            else:
                toolbar.children = [toolbar_button]

        toolbar_button.observe(toolbar_click, "value")
        toolbar_ctrl = WidgetControl(widget=toolbar, position="topright")

        self.add_control(toolbar_ctrl)

        output = widgets.Output()
        output_ctrl = WidgetControl(widget=output, position="topright")

        buttons = widgets.ToggleButtons(
            value=None,
            options=["Apply", "Reset", "Close"],
            tooltips=["Apply", "Reset", "Close"],
            button_style="primary",
        )
        buttons.style.button_width = "80px"

        data_dir = os.path.abspath(self.data_dir)
        mask_dir = os.path.abspath(self.mask_dir)
        output_dir = os.path.abspath(self.output_dir)

        fc = FileChooser(data_dir)
        fc.use_dir_icons = True
        fc.filter_pattern = ["*.tif", "*.gpkg", "*.shp", "*.geojson"]

        filechooser_widget = widgets.VBox([fc, buttons])

        def button_click(change):
            if change["new"] == "Apply" and fc.selected is not None:
                if fc.selected.endswith(".tif"):
                    self.add_raster(fc.selected, layer_name="Raster")
                elif fc.selected.endswith(".shp"):
                    self.add_shapefile(fc.selected, layer_name="Shapefile")
                elif fc.selected.endswith(".geojson"):
                    self.add_geojson(fc.selected, layer_name="GeoJSON")
            elif change["new"] == "Reset":
                fc.reset()
            elif change["new"] == "Close":
                fc.reset()
                self.remove_control(output_ctrl)
                buttons.value = None

        buttons.observe(button_click, "value")

        def tool_click(b):
            
            with output:
                
                output.clear_output()

                if b.icon == "folder-open":
                    display(filechooser_widget)
                    self.add_control(output_ctrl)

                elif b.icon == "gears":
                    if hasattr(self, "whitebox") and self.whitebox is not None:
                        if self.whitebox in self.controls:
                            self.remove_control(self.whitebox)

                    tools_dict = wbt.get_wbt_dict()
                    wbt_toolbox = wbt.build_toolbox(
                        tools_dict, max_width="800px", max_height="500px"
                    )

                    wbt_control = WidgetControl(
                        widget=wbt_toolbox, position="bottomright")

                    self.whitebox = wbt_control
                    self.add_control(wbt_control)

                elif b.icon == "map-marker":

                    #print(f"HEYYYY map-marker")
                    
                    # TODO: REMOVE CONTROL HERE to avoid error when close not
                    # pressed, I think there are two elements in the control
                    # print("CONTROL", type(output_ctrl), output_ctrl)
                    fc = FileChooser(mask_dir)
                    fc.use_dir_icons = True
                    fc.filter_pattern = ["*.tif"]

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

                    label_mask_widget = widgets.Text(
                        description="Layer name: ",
                        value="Mask",
                        layout=widgets.Layout(width="248px", padding="0px"),
                        style={"description_width": "initial"},
                    )

                    map_marker_btns = widgets.ToggleButtons(
                        value=None,
                        options=["Generate", "Display", "Close"],
                        tooltips=["Generate", "Display", "Close"],
                        button_style="primary",
                    )
                    map_marker_btns.style.button_width = "80px"
                    
                    #print(f"HEYYYY added marker_btn_click")

                    def marker_btn_click(change):
                        
                        print(f"HEYYYY marker_btn_click, {change}")

                        if change["new"] == "Generate" and \
                                fc.selected is not None:
                            
                            print(f"HEYYYY GENERATE")

                            """
                            HERE WE GENERATE THE POINTS
                            """
                            cmap = ['lightgray', 'green', 'orange', 'red']
                            classes = ['other', 'tree', 'crop', 'burn']
                            cmap = [mcolors.rgb2hex(cmap[i]) for i in range(len(cmap))]

                            # add mask raster to map
                            self.add_raster(
                                fc.selected,
                                data_bands=[1],
                                layer_name=label_mask_widget.value,
                                cmap=cmap
                            )

                            # generate random points over map
                            self.gen_points(label_mask_widget.value, n_points_widget.value).to_crs(4326)
                            """
                            markers_list = []
                            for index, point in self.random_points.iterrows():

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

                                #cell = ipysheet.cell(index, 2, type_pred)
                                #widgets.jslink((cell, 'value'), (radio_pred_widget, 'value'))
                                #widgets.jslink((radio_pred_widget, 'value'), (cell, 'value'))
                                #cell = ipysheet.cell(index, 3, 'other')
                                #widgets.jslink((cell, 'value'), (radio_check_widget, 'value'))
                                #widgets.jslink((radio_check_widget, 'value'), (cell, 'value'))
                                #cell = ipysheet.cell(index, 4, False)#, choice=)
                                #widgets.jslink((cell, 'value'), (checked_widget, 'value'))

                                # append to group of markers
                                markers_list.append(marker)

                            marker_cluster = MarkerCluster(
                                markers=tuple(markers_list),
                                name="validation"
                            )
                            # marker_cluster.on_click(handle_click)

                            self.add_layer(marker_cluster);
                            """

                            #self.validation_sheet = ipysheet.sheet(
                            #    ipysheet.from_dataframe(
                            #        random_points.to_crs(4326).drop(
                            #            ['geometry'], axis=1)
                            #))

                            #widgets.Dropdown.value.tag(sync=True)
                            # display(validation_sheet)
                            #display(validation_sheet)
                            #display(random_points)
                            
                            #self.random_points = random_points

                            # add markers to map
                            #self.add_markers(random_points)

                            """

                            df = pd.read_csv(fc.selected)
                            col_names = df.columns.values.tolist()
                            x_widget.options = col_names
                            y_widget.options = col_names
                            label_widget.options = col_names

                            if "longitude" in col_names:
                                x_widget.value = "longitude"

                            if "latitude" in col_names:
                                y_widget.value = "latitude"

                            if "name" in col_names:
                                label_widget.value = "name"
                            """

                        elif change["new"] == "Display":

                            """
                            HERE WE SHOW THE MARKERS AND RASTER
                            """
                            if x_widget.value is not None and \
                                    (y_widget.value is not None):
                                self.add_points_from_csv(
                                    fc.selected,
                                    x=x_widget.value,
                                    y=y_widget.value,
                                    label=label_widget.value,
                                    layer_name=layer_widget.value,
                                )

                        elif change["new"] == "Close":
                            fc.reset()
                            self.remove_control(output_ctrl)
                    
                    points_widget = widgets.VBox(
                        [
                            fc,
                            n_points_widget,
                            label_mask_widget,
                            map_marker_btns,
                        ]
                    )
                            
                    map_marker_btns.observe(marker_btn_click, "value")

                    display(points_widget)
                    self.add_control(output_ctrl)

                elif b.icon == "map":
                    fc = FileChooser(data_dir)
                    fc.use_dir_icons = True
                    fc.filter_pattern = ["*.csv"]

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
                        if change["new"] == "Read data" and \
                                fc.selected is not None:

                            df = pd.read_csv(fc.selected)
                            col_names = df.columns.values.tolist()
                            x_widget.options = col_names
                            y_widget.options = col_names
                            label_widget.options = col_names

                            if "longitude" in col_names:
                                x_widget.value = "longitude"

                            if "latitude" in col_names:
                                y_widget.value = "latitude"

                            if "name" in col_names:
                                label_widget.value = "name"

                        elif change["new"] == "Display":

                            if x_widget.value is not None and \
                                    (y_widget.value is not None):
                                self.add_points_from_csv(
                                    fc.selected,
                                    x=x_widget.value,
                                    y=y_widget.value,
                                    label=label_widget.value,
                                    layer_name=layer_widget.value,
                                )

                        elif change["new"] == "Close":
                            fc.reset()
                            self.remove_control(output_ctrl)

                    btns.observe(btn_click, "value")

                    csv_widget = widgets.VBox(
                        [
                            fc,
                            widgets.HBox([x_widget, y_widget]),
                            label_widget,
                            layer_widget,
                            btns,
                        ]
                    )

                    display(csv_widget)
                    self.add_control(output_ctrl)

        for i in range(rows):
            for j in range(cols):
                tool = grid[i, j]
                tool.on_click(tool_click)
