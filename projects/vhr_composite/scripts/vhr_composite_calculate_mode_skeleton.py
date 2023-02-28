import sys
import logging
import glob
import os
import tqdm

import pandas as pd
import geopandas as gpd

sys.path.append('.')
from vhr_composite.model.composite import Composite
from vhr_composite.model.utils import TqdmLoggingHandler


#
# To run:
# ssh gpulogin1
# salloc -G 4 -J composite_test
# module load anaconda
# 
def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('composite-skeleton.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(TqdmLoggingHandler())

    # * Set some (hardcoded for now) variables
    region = 'CAS'  # Assume for now we are doing one region at a time
    testName = 'testv3'
    epoch = '2016'
    region = 'CAS'
    modelName = 'otcb.v5'

    inStripShp = '/explore/nobackup/people/mwooten3/Senegal_LCLUC/' + \
        'VHR/_footprints/CAS_M1BS_metadataFootprints.shp'
    testName = 'testv3'
    epoch = '2010'

    # * Using 8-bit outputs I made for now as I no longer have access to older
    # output directory - eventually use the .cog files?
    lcDir = '/explore/nobackup/projects/3sl/development/' + \
        'cnn_landcover/normalization/otcb.v5'
    cloudDir = '/explore/nobackup/projects/3sl/products/' + \
        'cloudmask/v1/{}'.format(region)  # CHanging to explore soon

    grid_path = '/explore/nobackup/people/mwooten3/Senegal_LCLUC/' + \
        'Shapefiles/Grid/Senegal_Grid__all.shp'
    grid_gdf = gpd.read_file(grid_path)

    # Get gdf with strips of interest
    gdf = gpd.read_file(inStripShp)
    gdf = gdf.loc[gdf[testName] == epoch]

    # Set as columns in geodataframe
    gdf['landcover'] = list(map(lambda f: os.path.join(
        lcDir,
        '{}-toa.otcb.tif'.format(f)),
        gdf['strip_id']))
    gdf['cloudmask'] = list(map(lambda f: os.path.join(
        cloudDir,
        '{}-toa.cloudmask.tif'.format(f)),
        gdf['strip_id']))
    gdf['datetime'] = pd.to_datetime(gdf['acq_time'])
    gdf = gdf.to_crs(grid_gdf.crs)
    output_dir = '.'
    logger.info("Reading {} files from .shp {}".format(len(gdf), inStripShp))
    model_output_gdf_name = f'{region}.{modelName}.{testName}.{epoch}.gpkg'
    gdf.to_file(model_output_gdf_name, driver="GPKG")

    grid_output_path = '/explore/nobackup/people/cssprad1/projects/3sl/' + \
        '3.0.0/data/v3.2010'
    grid_files = sorted(glob.glob(os.path.join(grid_output_path, '*.zarr')))
    logger.info(grid_files)
    logger.info(f'Processing {len(grid_files)} tiles')

    composite = Composite(name=testName,
                          epoch=epoch,
                          grid_geopackage_path=grid_path,
                          model_output_geopackage_path=model_output_gdf_name,
                          output_dir=output_dir,
                          logger=logger)
    for tile_path in tqdm.tqdm(grid_files):
        composite.calculate_skeleton_per_tile(
            tile_path)


if __name__ == '__main__':
    sys.exit(main())
