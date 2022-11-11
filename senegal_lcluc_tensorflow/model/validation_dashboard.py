import os
import time
import math
import socket
import ipysheet
import ipyleaflet
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import ipywidgets as widgets
import whiteboxgui.whiteboxgui as wbt
import matplotlib.colors as mcolors
from pathlib import Path
from ipyfilechooser import FileChooser
from IPython.display import display
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

os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = \
    f"{os.environ['JUPYTERHUB_SERVICE_PREFIX'].lstrip('/')}/proxy/{{port}}"

from localtileserver import get_leaflet_tile_layer, TileClient


class ValidationDashboard(ipyleaflet.Map):
    """
    This Map class inherits the ipyleaflet Map class.
    Modified: https://github.com/giswqs/geodemo/blob/master/geodemo/geodemo.py
    Args:
        ipyleaflet (ipyleaflet.Map): An ipyleaflet map.
    """

    def __init__(self, **kwargs):

        # Define center start
        if "center" not in kwargs:
            kwargs["center"] = [14, -14]

        # Define initial zoom level
        if "zoom" not in kwargs:
            kwargs["zoom"] = 6
            
        # Define max_zoom options
        if "max_zoom" not in kwargs:
            kwargs["max_zoom"] = 20
            self.default_zoom = 18

        # Define zoom capabilities
        if "scroll_wheel_zoom" not in kwargs:
            kwargs["scroll_wheel_zoom"] = True

        # Allow keyboard movements
        if "keyboard" not in kwargs:
            kwargs["keyboard"] = True

        # Define basemap options
        if "basemap" not in kwargs:
            kwargs["basemap"] = basemaps.Esri.WorldImagery

        super().__init__(**kwargs)

        # Define the height of the layout
        if "height" not in kwargs:
            self.layout.height = "600px"
        else:
            self.layout.height = kwargs["height"]

        # Add control widgets
        self.add_control(FullScreenControl())
        self.add_control(MeasureControl())
        self.add_control(LayersControl(position="topright"))
        self.add_control(DrawControl(position="topleft"))
        self.add_control(ScaleControl(position="bottomleft"))
        
        # Retrieve hostname
        self.hostname = socket.gethostname()

        # Define data directory, if no argument given, proceed with defaults
        if "data_dir" not in kwargs:
            # Definition for Explore and SMCE clusters
            if self.hostname[:3] == 'gpu':
                self.data_dir = '/explore/nobackup/projects/3sl/data/Tappan'
            else:
                self.data_dir = '/home/jovyan/efs/projects/3sl/data/Tappan'
        else:
            self.data_dir = kwargs['data_dir']
        assert os.path.exists(self.data_dir), f'{self.data_dir} does not exist'

        # Define labels directory, if no argument given, proceed with defaults
        if "mask_dir" not in kwargs:
            # Definition for Explore and SMCE clusters
            if self.hostname[:3] == 'gpu':
                self.mask_dir = '/explore/nobackup/projects/ilab/projects/' + \
                    'Senegal/3sl/products/land_cover/dev/otcb.v1/Tappan'
            else:
                self.mask_dir = '/home/jovyan/efs/projects/3sl/products/otcb.v1/Tappan'
        else:
            self.mask_dir = kwargs['mask_dir']
        assert os.path.exists(self.mask_dir), f'{self.mask_dir} does not exist'
        
        # timestr = time.strftime("%Y%m%d-%H%M%S")
    
        # Define output directory, if no argument given, proceed with defaults
        if "output_dir" not in kwargs:
            # Definition for Explore and SMCE clusters
            if self.hostname[:3] == 'gpu':
                self.username = os.environ['USER']
                self.output_dir = '/explore/nobackup/projects/3sl/development/scratch'
            else:
                self.username = self.hostname.split('-')[-1]
                self.output_dir = '/home/jovyan/efs/projects/3sl/validation'
        else:
            self.output_dir = kwargs['output_dir']

        #print(self.username)
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cleanup data_dir and mask_dir variables
        self.data_dir = os.path.abspath(self.data_dir)
        self.mask_dir = os.path.abspath(self.mask_dir)
        
        # Define available raster bands
        if "default_bands" not in kwargs:
            self.default_bands = [
                ('Coastal Blue', 1), ('Blue', 2), ('Green', 3),  ('Yellow', 4),
                ('Red', 5), ('Red Edge', 6), ('NIR1', 7), ('NIR2', 8)
            ]
        else:
            self.default_bands = kwargs['default_bands']

        # Define default RGB bands
        if "rgb_bands" not in kwargs:
            self.rgb_bands = [7, 3, 2]
        else:
            self.rgb_bands = kwargs['rgb_bands']
            assert len(self.rgb_bands) == 3, f"rgb_bands requires list of 3 items"
        
        # Define wether we allow users to define their own bands
        if "rgb_disabled" not in kwargs:
            self.rgb_disabled = True
        else:
            self.rgb_disabled = kwargs['rgb_disabled']
        
        # Define validation classes
        if "validation_classes" not in kwargs:
            self.validation_classes = [
                'other', 'trees/shrub', 'cropland', 'other vegetation',
                'water', 'build'
            ]
        else:
            self.validation_classes = kwargs['validation_classes']
            
        # Define mask classes
        if "mask_classes" not in kwargs:
            self.mask_classes = ['other', 'tree', 'crop', 'burn']
        else:
            self.mask_classes = kwargs['mask_classes']

        # Define if validation points need to be generated
        if "gen_points" not in kwargs:
            self.gen_points = True
        else:
            self.gen_points = kwargs['gen_points']
        
        if "expected_accuracies" not in kwargs:
            self.expected_accuracies = [0.90, 0.90, 0.90, 0.90]
        else:
            self.expected_accuracies = kwargs['expected_accuracies']

        if "expected_standard_error" not in kwargs:
            self.expected_standard_error = 0.01
        else:
            self.expected_standard_error = kwargs['expected_standard_error']
            
        if "product_name" not in kwargs:
            self.product_name = 'otcb'
        else:
            self.product_name = kwargs['product_name']

        if "chunks" not in kwargs:
            self.chunks = {"band": 1, "x": 2048, "y": 2048}
        else:
            self.chunks = kwargs["chunks"]

        self.output_filename = None
        self._validation_sheet = None
        self._markers_dict = dict()
        self._marker_counter = -1
        self.raster_crs = None

        # Start main toolbar
        self._main_toolbar()


    def add_raster(
                self,
                in_raster: str,
                style: str = None,
                data_bands: list = [5, 7, 2],
                cmap: list = None,
                sigma: int = 2,
                layer_name: str = "Raster"
            ):
        """
        Adds a raster layer to the map.
        Args:
        """
        # create TileClient object
        raster_client = TileClient(in_raster)
        
        style_list = []
        for bid, pid in zip(data_bands, ['#f00', '#0f0', '#00f']):
            band_stats = raster_client.rasterio.statistics(bid)
            newmin = band_stats.mean - (band_stats.std * sigma)
            newmax = band_stats.mean + (band_stats.std * sigma)
            style_list.append(
                {'band': bid, 'palette': pid, 'min': newmin, 'max': newmax})
        
        self.add_layer(
            get_leaflet_tile_layer(
                raster_client, show=False, band=data_bands,
                cmap=cmap, max_zoom=int(self.max_zoom),
                max_native_zoom=int(self.max_zoom),
                name=layer_name, scheme='linear',
                dtype='uint16', style={'bands': style_list}
            )
        )
        self.center = raster_client.center()  # center Map around raster
        self.zoom = raster_client.default_zoom  # zoom to raster center

    def generate_points(self, mask_filename: str):
        """
        Generate points
        """
        # read prediction raster
        raster_prediction = rxr.open_rasterio(
            mask_filename, chunks=self.chunks)
        raster_prediction.name = "predicted"
        raster_prediction = raster_prediction.rio.reproject("EPSG:4326")
        self.raster_crs = raster_prediction.rio.crs

        # convert to dataframe and filter no-data
        raster_prediction = raster_prediction.squeeze().to_dataframe().reset_index()
        raster_prediction = raster_prediction.drop(
            ['band', 'spatial_ref'], axis=1)  # drop some unecessary columns
        # only select appropiate values, remove no-data
        raster_prediction = raster_prediction[raster_prediction['predicted'] >= 0]
        #print("BEFORE", raster_prediction.shape)
        raster_prediction = raster_prediction[raster_prediction['predicted'] < 255]
        #print("AFTER", raster_prediction.shape)
        # convert mask into int
        raster_prediction = raster_prediction.astype({'predicted': 'int'})
        
        unique_counts = raster_prediction['predicted'].value_counts()
        original_shape = raster_prediction.shape[0]
        
        percentage_counts, standard_deviation = [], []
        for class_id, class_count in unique_counts.iteritems():
            #print("CLASS ID", class_id, class_count)
            percentage_counts.append(class_count / original_shape)
            standard_deviation.append(
                math.sqrt(
                    self.expected_accuracies[class_id] * \
                    (1 - self.expected_accuracies[class_id]))
            )
            
        unique_counts = unique_counts.to_frame()
        unique_counts['percent'] = percentage_counts
        unique_counts['standard_deviation'] = standard_deviation
        unique_counts = unique_counts.round(2)

        val_total_points = round((
            (unique_counts['percent'] * unique_counts['standard_deviation']).sum() / \
                self.expected_standard_error) ** 2)
        
        for class_id, row in unique_counts.iterrows():
            val_points = round(row['percent'] * val_total_points)
            raster_prediction = raster_prediction.drop(
                raster_prediction[raster_prediction['predicted'] == class_id].sample(
                    n=int(row['predicted'] - val_points), random_state=24).index
            )
        #TEMP
        #print(raster_prediction.shape)
        #raster_prediction = raster_prediction.iloc[:100 , :]
        #print(raster_prediction.shape)
        #print("original CRS", self.raster_crs)


        geometry = gpd.points_from_xy(raster_prediction.x, raster_prediction.y)
        raster_prediction = gpd.GeoDataFrame(
            raster_prediction, crs=self.raster_crs, geometry=geometry).reset_index(drop=True)
        return raster_prediction

    def add_markers(self, in_raster: str, gen_points: bool = True):
        
        # Extract label filename from data filename
        mask_filename = os.path.join(
            self.mask_dir, f'{Path(in_raster).stem}.{self.product_name}.tif')
        
        if not gen_points or os.path.isfile(self.output_filename):
            #print("NOT GEN POINTS")
            validation_points = self.load_gpkg(self.output_filename)
            #print(validation_points.crs)
            validation_points = validation_points.to_crs(epsg=4326)
            #print(validation_points.crs)
        else:
            #print("GEN POINTS")
            validation_points = self.generate_points(mask_filename)
            validation_points = validation_points.drop(['predicted'], axis=1).to_crs(4326)
            validation_points['operator'] = 'other'
            validation_points['burnt'] = 0
            validation_points['confidence'] = 1
            validation_points['verified'] = 'false'
            
        #print("VAL POINTS AFTER TO CRS", validation_points.head())
        #print("VAL POINTS AFTER TO CRS", validation_points.crs)

        
        self._validation_sheet = ipysheet.sheet(
            ipysheet.from_dataframe(
                validation_points.to_crs(4326).drop(['geometry'], axis=1)))
        widgets.Dropdown.value.tag(sync=True)
        
        #print(validation_points.head())


        for index, point in validation_points.iterrows():
            
            #print(index)
            
            # get coordinates
            coordinates = (point['geometry'].y, point['geometry'].x)
            #print(point)
            if point['verified'] == 'false' or not point['verified']:
                verified_option = False
            else:
                verified_option = True
            #print(coordinates, point['operator'], verified_option)
        
            radio_check_widget = widgets.RadioButtons(
                options=self.validation_classes,
                value=point['operator'],
                layout={'width': 'max-content'}, # If the items' names are long
                description='Validation:',
                disabled=False
            )
            #print("BEFORE FORCING radio_check_widget", radio_check_widget.value)
            #radio_check_widget.value = point['operator']
            #print("AFTER FORCING radio_check_widget", radio_check_widget.value)
            
            radio_burn_widget = widgets.RadioButtons(
                options=[('not-burnt', 0), ('burnt', 1)],
                value=point['burnt'],
                layout={'width': 'max-content'}, # If the items' names are long
                description='Burnt:',
                disabled=False
            )
            #print("BEFORE FORCING radio_burn_widget", radio_burn_widget.value)
            #radio_check_widget.value = point['burnt']
            #print("AFTER FORCING radio_burn_widget", radio_burn_widget.value)
            
            
            radio_confidence_widget = widgets.RadioButtons(
                options=[('high-confidence', 1), ('medium-confidence', 2), ('low-confidence', 3)],
                value=point['confidence'],
                layout={'width': 'max-content'}, # If the items' names are long
                description='Confidence:',
                disabled=False
            )
            point_id_widget = widgets.IntText(
                value=index,
                description='ID:',
                disabled=True
            )
            checked_widget = widgets.Checkbox(
                value=verified_option,
                description='Verified:',
                disabled=False
            )

            popup = widgets.VBox([
                point_id_widget,
                radio_check_widget,
                radio_burn_widget,
                radio_confidence_widget,
                checked_widget
            ])
            
            #### 
            marker = Marker(
                name=str(index),
                location=coordinates,
                draggable=False,
                keyboard=True,
                popup=popup
            )
            
            # IF THIS POINT WAS VERIFIED, ADD MARKER AS WHEN CHECKEDDDDD#
            if verified_option:
                #print("YESSSS WAS VERIFIED")
                marker.icon = AwesomeIcon(
                    name='check-square',
                    marker_color='green',
                    icon_color='black'
                )

            cell = ipysheet.cell(index, 2, point['operator'])
            widgets.jslink((cell, 'value'), (radio_check_widget, 'value'))
            cell = ipysheet.cell(index, 3, point['burnt'])
            widgets.jslink((cell, 'value'), (radio_burn_widget, 'value'))
            cell = ipysheet.cell(index, 4, point['confidence'])
            widgets.jslink((cell, 'value'), (radio_confidence_widget, 'value'))
            cell = ipysheet.cell(index, 5, verified_option)
            widgets.jslink((cell, 'value'), (checked_widget, 'value'))
            
            self._markers_dict[tuple(marker.location)] = marker
   
            def handle_marker_on_click(*args, **kwargs):
                
                self._markers_dict[tuple(kwargs['coordinates'])].icon = AwesomeIcon(
                    name='check-square',
                    marker_color='green',
                    icon_color='black'
                )
                
                self.save_gpkg(
                    ipysheet.to_dataframe(self._validation_sheet),
                    self.output_filename
                )
            marker.on_click(handle_marker_on_click)
                            
        marker_cluster = MarkerCluster(
            markers=tuple(list(self._markers_dict.values())),
            name="validation"
        )
        self.add_layer(marker_cluster)
        
        return

    def save_gpkg(self, df, output_filename, layer="validation"):
        gdf = gpd.GeoDataFrame(
            df, crs=self.raster_crs,
            geometry=gpd.points_from_xy(df.x, df.y))
        gdf.to_file(output_filename, layer=layer, driver="GPKG")
        return

    def load_gpkg(self, input_filename):
        gdf = gpd.read_file(input_filename).drop(['index'], axis=1)
        self.raster_crs = gdf.crs
        self._marker_counter = gdf[
            'verified'][gdf['verified'] == True].last_valid_index()
        return gdf

    def _main_toolbar(self):
        """
        Generate main toolbar widget
        """
        # Define toolbar padding
        padding = "0px 0px 0px 5px"  # upper, right, bottom, left

        # Define toolbar_button inside Map
        toolbar_button = widgets.ToggleButton(
            value=False,
            tooltip="Toolbar",
            icon="wrench",
            layout=widgets.Layout(
                width="28px", height="28px", padding=padding),
        )

        # Define close_button inside Map
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

        icons = ["folder-open", "gears", "arrow-left", "arrow-right"] #"map-marker",

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
        
        # bands chooser
        red = widgets.Dropdown(
            options=self.default_bands,
            value=self.rgb_bands[0],
            description='Red:',
            disabled=self.rgb_disabled
        )
        green = widgets.Dropdown(
            options=self.default_bands,
            value=self.rgb_bands[1],
            description='Green:',
            disabled=self.rgb_disabled
        )
        blue = widgets.Dropdown(
            options=self.default_bands,
            value=self.rgb_bands[2],
            description='Blue:',
            disabled=self.rgb_disabled
        )

        bands_widget = widgets.HBox([red, green, blue])

        buttons.style.button_width = "80px"

        # Define FileChooser widget
        fc = FileChooser(self.data_dir)
        fc.use_dir_icons = True
        fc.filter_pattern = ["*.tif", "*.gpkg", "*.shp", "*.geojson"]
        filechooser_widget = widgets.VBox([fc, bands_widget, buttons])

        def button_click(change):
            """
            Observer for filechooser_widget button_click
            """
            if change["new"] == "Apply" and fc.selected is not None:
                
                self._selected_filename = fc.selected

                if self._selected_filename.endswith(".tif"):
                    
                    # Get current bands from widget, add raster to Map
                    data_bands = [item.value for item in bands_widget.children]                    
                    self.add_raster(
                        self._selected_filename, layer_name="Raster", data_bands=data_bands)
                    
                    # Set output_filename
                    full_dirs = self._selected_filename.split('/')
                    output_dir = os.path.join(
                        self.output_dir, self.username, full_dirs[-3], full_dirs[-2])
                    os.makedirs(output_dir, exist_ok=True)

                    self.output_filename = os.path.join(
                        output_dir, f"{self.username}-{Path(self._selected_filename).stem}.gpkg")
                    
                    # Visualize or generate markers for validation
                    self.add_markers(self._selected_filename, gen_points=self.gen_points)

                elif self._selected_filename.endswith(".shp"):
                    self.add_shapefile(self._selected_filename, layer_name="Shapefile")
                elif self._selected_filename.endswith(".geojson"):
                    self.add_geojson(self._selected_filename, layer_name="GeoJSON")

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

                elif b.icon == "arrow-right":
                    self._marker_counter = self._marker_counter + 1
                    if len(list(self._markers_dict)) <= self._marker_counter:
                        self._marker_counter = 0

                    self.center = tuple(list(self._markers_dict)[self._marker_counter])
                    self.zoom = self.default_zoom
                    
                elif b.icon == "arrow-left":
                    self._marker_counter = self._marker_counter - 1
                    if self._marker_counter < 0:
                        self._marker_counter = len(list(self._markers_dict)) - 1

                    self.center = tuple(list(self._markers_dict)[self._marker_counter])
                    self.zoom = self.default_zoom

        for i in range(rows):
            for j in range(cols):
                tool = grid[i, j]
                tool.on_click(tool_click)
