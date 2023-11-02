import os
import logging
import tqdm
import rasterio
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import pandas as pd
import numpy as np

from vhr_composite.model.metrics import calculate_mode

TIME: str = "time"
X: str = "x"
Y: str = "y"
BAND: int = 1
CLOUD_CLEAR_VALUE: int = 0
NO_DATA: int = 255
OTHER_DATA: int = 10
BURN_AREA_VALUE: int = 3


class Composite(object):

    def __init__(self,
                 name: str,
                 epoch: str,
                 grid_geopackage_path: str,
                 model_output_geopackage_path: str,
                 output_dir: str,
                 logger: logging.Logger) -> None:
        """
        Initializes the Composite object
        """
        self._name = name
        self._epoch = epoch
        self._experiment_name = f'{self._name}.{self._epoch}'
        self._grid_geopackage_path = grid_geopackage_path
        self._model_output_geopackage_path = model_output_geopackage_path
        self._logger = logger

        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir

        if not os.path.exists(self._grid_geopackage_path) or \
                not os.path.exists(self._model_output_geopackage_path):
            msg = '{} does not exist'.format(self._grid_geopackage_path)
            raise FileNotFoundError(msg)

        self._grid = gpd.read_file(self._grid_geopackage_path)
        self._model_output = \
            gpd.read_file(self._model_output_geopackage_path)

    def generate_grid(self, calculate_mode: bool = False,
                      classes: dict = None) -> None:
        """
        Generate the gridded zarrs from the strips that intersect
        each grid cell
        """
        self._logger.info('Calculating intersect')
        intersection = self._get_intersection()
        for tile in tqdm.tqdm(intersection.keys()):
            tile_df = self._grid[self._grid['tile'] == tile]
            tiles_df = [tile_df for _ in range(len(intersection[tile]))]
            strips = intersection[tile]
            landcovers = strips['landcover']
            cloudmasks = strips['cloudmask']
            datetimes = strips['datetime']
            arrays = list(map(Composite.strip_to_grid,
                              landcovers, cloudmasks,
                              datetimes, tiles_df))
            arrays = [array for array in arrays if array is not None]
            name = 'CAS.M1BS.{}'.format(tile)
            concat_array = xr.concat(arrays, dim='time', fill_value=10)
            concat_dataset = concat_array.to_dataset(name=name)
            tile_path = os.path.join(self._output_dir, f'{name}.zarr')
            concat_dataset.to_zarr(tile_path)
            if calculate_mode and classes:
                self.calculate_mode_per_tile(tile_path=tile_path,
                                             classes=classes,
                                             tile_dataset_input=concat_dataset,
                                             )
        return None

    def _get_intersection(self) -> dict:
        """
        For each grid cell, find the strips that intersect
        """
        intersection_dict = {}
        for i in tqdm.tqdm(range(len(self._grid))):
            intersects = self._model_output['geometry'].apply(
                lambda shp: shp.intersects(self._grid.iloc[i]['geometry']))
            does_intersect = intersects[intersects]
            if len(does_intersect) > 0:
                intersection_dict[self._grid.iloc[i]['tile']] = \
                    self._model_output.loc[list(does_intersect.index)]
        self._logger.info(f'Found {len(intersection_dict)} intersections')
        return intersection_dict

    @staticmethod
    def strip_to_grid(
            land_cover_path: str,
            cloud_mask_path: str,
            timestamp: pd._libs.tslibs.timestamps.Timestamp,
            grid_geodataframe: gpd.GeoDataFrame) -> xr.DataArray:
        """
        This function opens a strip and clips it to a given geometry
        and performs some basic quality assurance checking.

        :param strip_path: str, the file path for the strip to read and clip
        :param cloud_mask_path: str, the file path for the corresponding
        cloud mask
        :param timestamp: the time-stamp for the
        :param grid_geodataframe: geopandas.DataFrame, the dataframe whose
        geometry is used to clip the strip to
        :return: rioxarray.DataArray, the clipped DataArray strip
        """
        try:
            strip_data_array = rxr.open_rasterio(land_cover_path)
            cloud_mask_data_array = rxr.open_rasterio(cloud_mask_path)
        except rasterio.errors.RasterioIOError:
            return None

        geometry_to_clip = grid_geodataframe.geometry.values
        geometry_crs = grid_geodataframe.crs
        strip_data_array = strip_data_array.rio.clip(geometry_to_clip,
                                                     crs=geometry_crs)

        strip_data_array = strip_data_array.assign_coords(time=timestamp)
        strip_data_array = strip_data_array.expand_dims(dim=TIME)
        strip_data_array = strip_data_array.where((
            (cloud_mask_data_array == 0) &
            (strip_data_array != BURN_AREA_VALUE) &
            (strip_data_array != NO_DATA)
        ), other=OTHER_DATA)

        return strip_data_array

    def calculate_mode_per_tile(self,
                                tile_path: str,
                                classes: dict,
                                tile_dataset_input: xr.Dataset = None) -> None:
        """
        Given a landcover zarr or dataset, calculate the mode
        and write to GTiff
        """
        tile_dataset = tile_dataset_input if tile_dataset_input \
            else xr.open_zarr(tile_path)
        variable_name = os.path.basename(tile_path).split('.zarr')[0]
        # Select the array without the band, transpose to time-last format
        tile_array = tile_dataset[variable_name].sel(
            band=BAND).transpose(Y, X, TIME)
        mode = self._calculate_mode(tile_array, classes, logger=self._logger)
        self._logger.info('Compute mode - Configuring mode to data array')
        # Add the band to the mode
        mode_with_band = np.zeros((BAND, mode.shape[0], mode.shape[1]))
        mode_with_band[0, :, :] = mode

        # Make the coordinates that will be used to make the mode ndarray
        # a xr.DataArray
        coords = dict(
            band=tile_dataset.band,
            y=tile_dataset.y,
            x=tile_dataset.x,
            spatial_ref=tile_dataset.spatial_ref,
        )
        name = '{}.mode.{}'.format(variable_name, self._experiment_name)
        self._logger.info(f'Compute mode - Appending {name} to {tile_path}')
        mode_data_array = self._make_data_array_from_mode(mode_with_band,
                                                          coords,
                                                          name)

        # Write to GTiff
        tile_raster_path = tile_path.replace('.zarr', f'{name}.tif')
        mode_data_array.rio.to_raster(tile_raster_path)
        return None

    def _make_data_array_from_mode(
            self,
            ndarray: np.ndarray,
            coords: dict,
            name: str) -> xr.DataArray:
        """
        Given a ndarray, make it a Xarray DataArray
        """
        data_array = xr.DataArray(
            data=ndarray,
            dims=['band', 'y', 'x'],
            coords=coords,
            attrs=dict(
                description="Mode of model results"
            ),
        )
        data_array.name = name
        return data_array

    def _calculate_mode(self, tile_array: xr.DataArray,
                        classes: dict,
                        logger: logging.Logger = None):
        """
        Object-oriented wrapper for mode calculation function.
        """
        return calculate_mode(tile_array, classes, logger=logger)
