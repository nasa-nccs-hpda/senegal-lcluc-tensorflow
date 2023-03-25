import os
import ee
import re
import sys
import logging
import requests
import rasterio
import rioxarray as rxr
from glob import glob
from typing import Literal
from pathlib import Path
from shapely.geometry import box
from multiprocessing import Pool, cpu_count


class LandUseClassification(object):

    def __init__(
                self,
                config_filename: str = None,
                logger=None,
                gee_session=None
            ):

        # TODO:
        # slurm filename in output dir

        # Set logger
        self.logger = logger if logger is not None else self._set_logger()

        # Set GEE Session
        self._gee_session = gee_session

        # Set polarization placeholder
        self._polarization = None

        # Set s1_db placeholder
        self._s1_db = None

        # Set months placeholder
        self._months = None

        # Configuration file intialization
        # self.conf = self._read_config(config_filename)

        # Set output directories and locations
        # self.images_dir = os.path.join(self.conf.data_dir, 'images')
        # os.makedirs(self.images_dir, exist_ok=True)
        # self.logger.info(f'Images dir: {self.images_dir}')

        # self.labels_dir = os.path.join(self.conf.data_dir, 'labels')
        # os.makedirs(self.labels_dir, exist_ok=True)
        # self.logger.info(f'Images dir: {self.labels_dir}')

        # self.model_dir = self.conf.model_dir
        # os.makedirs(self.model_dir, exist_ok=True)

        # Seed everything
        # seed_everything(self.conf.seed)

    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------
    def setup(
                self,
                label_regex: str,
                gee_account: str,
                gee_key: str,
                polarization: Literal["VV", "VH"] = 'VH',
                output_dir: str = 'output'
            ):

        assert label_regex is not None, \
            'label_regex should not be None. ' + \
            'If working from the CLI specify --label-regex'

        # get label filenames
        label_filenames = glob(label_regex)
        logging.info(f'Processing {len(label_filenames)} filenames.')

        # setup GEE auth
        # service account and key filename for the service account
        self._set_gee_session(gee_account, gee_key)

        # set output_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # set polarization for global use
        self._polarization = polarization
        logging.info(f'Downloading {self._polarization} polarization.')

        # iterate over label regex filenames
        for filename in label_filenames:

            logging.info(f'Downloading Sentinel-1 for {filename}')

            # Get year from filename
            year = self.get_date_from_filename(filename)['year']
            logging.info(f'Using {year} to download Sentinel-1 data.')

            # get bounds from filename
            gee_aoi = self.get_raster_gee_bounds(filename)
            logging.info(f'Bounds to use on GEE: {gee_aoi.getInfo()}')

            # request sentinel-1 data
            self.s1_db = self.get_sentinel1(gee_aoi, polarization, year)
            logging.info(
                'S1 ImageCollection: ' +
                f'{len(self.s1_db.toBands().bandNames().getInfo())}')

            # preprocess s1_db with logarithmic power
            self.s1_db = self.s1_db.map(self.compute_gee_power)
            logging.info(
                'Processed logarithmic power for ' +
                f'{len(self.s1_db.toBands().bandNames().getInfo())} bands')

            # compute monthly median across the dataset, convert
            # into ee.Image object with 12 bands, one per month
            months = ee.List.sequence(1, 12)
            self.s1_db = self.compute_gee_month_sequence(months, self.s1_db)
            self.s1_db = self.compute_gee_clip(self.s1_db, gee_aoi)
            logging.info(
                'Processed monthly mean, resulted in ' +
                f'{len(self.s1_db.bandNames().getInfo())} bands')

            # output_filename for all bands
            output_filename = os.path.join(
                self.output_dir,
                f'{Path(filename).stem}_{self._polarization}.tif'
            )

            # join all rasters into a single filename
            self.get_raster_stack(output_filename)

        return

    def preprocess(self):
        return

    def train(self):
        return

    def predict(self):
        return

    def get_sentinel1(
                self,
                gee_aoi: ee.geometry.Geometry,
                polarization: str,
                year: str,
                gee_s1_path: str = 'COPERNICUS/S1_GRD',
                gee_instrument_mode: str = 'IW'
            ):
        gee_image_collection = ee.ImageCollection(gee_s1_path) \
            .filter(ee.Filter.eq('instrumentMode', gee_instrument_mode)) \
            .filterBounds(gee_aoi) \
            .select(polarization) \
            .filterDate(f'{year}-01-01', f'{year}-12-31')
        return gee_image_collection

    def get_raster_gee_bounds(self, filename: str) -> ee.geometry.Geometry:
        raster = rxr.open_rasterio(filename).rio.reproject("EPSG:4326")
        xx, yy = box(*raster.rio.bounds()).exterior.coords.xy
        image_bounds = [[x, y] for x, y in zip(xx, yy)]
        return ee.Geometry.Polygon(image_bounds)

    def get_raster_gee(self, band: str):
        url = self.s1_db.getDownloadUrl({
            'bands': band,
            'scale': 10,
            'format': 'GEO_TIFF'
        })
        response = requests.get(url)
        output_filename = os.path.join(self.output_dir, f'{band}.tif')
        with open(output_filename, 'wb') as fd:
            fd.write(response.content)
        return

    def get_raster_stack(self, output_filename: str):

        # get band names for parallel download
        band_names = self.s1_db.bandNames().getInfo()
        p = Pool(processes=cpu_count())
        p.map(self.get_raster_gee, band_names)
        p.close()
        p.join()

        # Temporary output filenames
        file_list = [
            os.path.join(self.output_dir, f'{b}.tif') for b in band_names]

        # Read metadata of first file
        with rasterio.open(file_list[0]) as src0:
            meta = src0.meta

        # Update meta to reflect the number of layers
        meta.update(count=len(file_list))

        # Read each layer and write it to stack
        with rasterio.open(output_filename, 'w', **meta) as dst:
            for id, layer in enumerate(file_list, start=1):
                with rasterio.open(layer) as src1:
                    dst.write_band(id, src1.read(1))

        # Remove temporary files
        list(map(os.remove, file_list))

        return

    def get_date_from_filename(self, filename: str) -> re.Match:
        date_match = re.search(
            r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})', filename)
        return date_match

    def compute_segmentation(self):
        return

    def compute_gee_power(self, image: ee.image.Image) -> ee.element.Element:
        return ee.Image(10).pow(image.divide(10)).rename(
            self._polarization).copyProperties(image, ['system:time_start'])

    def compute_gee_month_sequence(
                self,
                months: ee.ee_list.List,
                s1_db: ee.imagecollection.ImageCollection
            ) -> ee.imagecollection.ImageCollection:
        image_collection = ee.ImageCollection.fromImages(
            months.map(
                lambda month: s1_db.filter(
                    ee.Filter.calendarRange(month, month, 'month')
                ).median().set('month', month)
            )
        )
        return image_collection

    def compute_gee_clip(
                self,
                s1_db: ee.imagecollection.ImageCollection,
                gee_aoi: ee.geometry.Geometry
            ) -> ee.image.Image:
        return s1_db.toBands().clip(gee_aoi)

    def _set_gee_session(self, gee_account: str, gee_key: str) -> None:
        credentials = ee.ServiceAccountCredentials(gee_account, gee_key)
        self._gee_session = ee.Initialize(credentials)
        logging.info('GEE authentication set.')

    def _set_logger(self) -> logging.RootLogger:
        """
        Set logger configuration.
        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    @property
    def polarization(self) -> str:
        return self._polarization

    @polarization.setter
    def polarization(self, value: str) -> None:
        self._polarization = value
