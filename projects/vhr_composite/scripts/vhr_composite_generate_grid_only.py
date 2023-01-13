import sys
import logging
import os
import pandas as pd
import geopandas as gpd

from vhr_composite.model.composite import Composite
from vhr_composite.model.utils import TqdmLoggingHandler


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('grid-generation.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(TqdmLoggingHandler())

    region = 'CAS'
    modelName = 'otcb.v5'
    testName = 'testv3'
    epoch = '2010'

    inStripShp = '/explore/nobackup/people/mwooten3/Senegal_LCLUC/' + \
        'VHR/_footprints/CAS_M1BS_metadataFootprints.shp'
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
    model_output_gdf_path = os.path.join(output_dir, model_output_gdf_name)
    gdf.to_file(model_output_gdf_path, driver="GPKG")

    composite = Composite(name=testName,
                          epoch=epoch,
                          grid_geopackage_path=grid_path,
                          model_output_geopackage_path=model_output_gdf_name,
                          output_dir=output_dir,
                          logger=logger)
    composite.generate_grid()


if __name__ == '__main__':
    sys.exit(main())
