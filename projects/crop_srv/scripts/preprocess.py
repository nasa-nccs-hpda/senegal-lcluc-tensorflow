# --------------------------------------------------------------------------
# Preprocessing script to extract training tiles from SRV data.
# export PYTHONPATH="/adapt/nobackup/people/jacaraba/development/tensorflow-caney"
# python scripts/preprocess.py --data-regex '/explore/nobackup/projects/3sl/data/VHR/SRV/M1BS/*-toa.tif' --label-regex '/explore/nobackup/projects/3sl/labels/landcover/srv/*.gpkg' --output-dir '/explore/nobackup/projects/3sl/development/cnn_landcover/crop.srv.v1/metadata'
# python scripts/preprocess.py --data-regex '/explore/nobackup/projects/3sl/data/VHR/SRV/M1BS/*-toa.tif' --label-regex '/explore/nobackup/projects/3sl/labels/landcover/srv/*.gpkg' --output-dir '/explore/nobackup/projects/3sl/development/cnn_landcover/crop.srv.v4' --tile-size 256 --footprint --tiles
# --------------------------------------------------------------------------
import os
import re
import sys
import time
import logging
import argparse
import geopandas as gpd
from multiprocessing import Pool, cpu_count

# import cuspatial
from glob import glob
from pathlib import Path
from tensorflow_caney.utils.raster.extract import get_raster_total_bounds
from tensorflow_caney.utils.vector.extract import filter_gdf_by_list, \
    extract_tiles


def run(conf) -> None:

    # define output directories
    metadata_output_dir = os.path.join(conf.output_dir, 'metadata')
    images_output_dir = os.path.join(conf.output_dir, 'images')
    labels_output_dir = os.path.join(conf.output_dir, 'labels')

    # create output directories
    for odir in [images_output_dir, labels_output_dir, metadata_output_dir]:
        os.makedirs(odir, exist_ok=True)

    # generate footprints from matching polygon and raster datasets
    if conf.footprint:

        # get total bounds of all rasters available for training
        raster_gdf = get_raster_total_bounds(conf.data_regex)
        raster_year_list = raster_gdf['acq_year'].unique()
        logging.info(raster_gdf.head())
        logging.info(f'{raster_gdf.shape}, {raster_year_list}')

        # get label filenames
        label_filenames = glob(conf.label_regex)

        # iterate over each label filename based on the years available
        # from the raster gdf, this will allow to only use available years
        for year in sorted(raster_year_list):

            # get label filename based on year
            label_filename = list(
                filter(lambda v: re.findall(
                    f'({str(year)})', v), label_filenames)
            )

            # skip if list is empty
            if not label_filename:
                continue

            # get element from list and convert to string
            label_filename = label_filename[0]

            # read label geopackage filename, includes field polygons
            polygon_gdf = gpd.read_file(label_filename)
            polygon_gdf = polygon_gdf.to_crs(raster_gdf.crs)
            logging.info(f"{label_filename}, {polygon_gdf.shape[0]} polygons")

            # filter original raster gdf to only match the year of the label
            filtered_raster_gdf = filter_gdf_by_list(
                raster_gdf, 'acq_year', [int(year)])

            # join and intersect the polygon and raster gdf's based on geometry
            polygon_gdf = gpd.sjoin(
                polygon_gdf, filtered_raster_gdf[
                    ['study_area', 'scene_id', 'acq_year', 'geometry']],
                how='left', op='intersects'
            )
            polygon_gdf = polygon_gdf.dropna(
                axis=0,
                how='any',
                thresh=None,
                subset=None,
            )
            polygon_gdf['acq_year'] = polygon_gdf['acq_year'].astype(int)

            # output the intersected polygon for later use
            output_filename = os.path.join(
                metadata_output_dir, f'{Path(label_filename).stem}.gpkg')

            # save geopackage file within the output_dir
            polygon_gdf.to_file(
                output_filename, driver='GPKG', layer='Intersection')
            logging.info(f"{output_filename}, {polygon_gdf.shape[0]} polygons")

    # iterate over the previously saved intersection files
    # extract tiles fully in parallel with multi-core parallelization
    if conf.tiles:

        # list dataset filenames from disk
        intersection_filenames = glob(
            os.path.join(metadata_output_dir, '*.gpkg'))
        assert len(intersection_filenames) > 0, \
            f"No gpkg files found under {metadata_output_dir}."

        for intersection_filename in intersection_filenames:

            # open the filename
            dataset_gdf = gpd.read_file(intersection_filename)

            # since we want to add paralelization, include some elements
            # in the greater array to merge a single object
            dataset_gdf['tile_size'] = conf.tile_size
            dataset_gdf['images_output_dir'] = images_output_dir
            dataset_gdf['labels_output_dir'] = labels_output_dir
            dataset_gdf['clustering_band'] = 6

            p = Pool(processes=conf.num_procs)
            p.map(extract_tiles, dataset_gdf.iterrows())
            p.close()
            p.join()

    return


def main() -> None:

    # Process command-line args.
    desc = 'Use this application to map LCLUC in Senegal using WV data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--cli', dest='cli_only', action='store_true', default=False)

    parser.add_argument(
        '--footprint', dest='footprint', action='store_true', default=False)

    parser.add_argument(
        '--tiles', dest='tiles', action='store_true', default=True)

    parser.add_argument(
        '-p', '--num-procs', dest='num_procs', type=int, default=cpu_count(),
        help='Number of parallel processes')

    parser.add_argument('-d',
                        '--data-regex',
                        type=str,
                        required=True,
                        dest='data_regex',
                        help='Regex to WorldView TIF files (e.g. /disk/*.tif)')

    parser.add_argument('-l',
                        '--label-regex',
                        type=str,
                        required=True,
                        dest='label_regex',
                        help='Regex to label GPKG files (e.g. /disk/*.tif)')

    parser.add_argument('-o',
                        '--output-dir',
                        type=str,
                        required=True,
                        dest='output_dir',
                        help='Output directory to store metadata')

    parser.add_argument('-ts',
                        '--tile-size',
                        type=int,
                        required=True,
                        dest='tile_size',
                        help='Tile size for data tiles to extract')

    args = parser.parse_args()

    # Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Call run for preprocessing steps
    timer = time.time()
    run(args)
    logging.info(
        f'Done with preprocessing, took {(time.time()-timer)/60.0:.2f} min.')

    return


# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    main()
