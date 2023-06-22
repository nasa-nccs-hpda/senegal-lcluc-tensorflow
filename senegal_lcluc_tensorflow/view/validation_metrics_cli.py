import os
import sys
import logging
import warnings
import argparse
import rasterio
import xarray as xr
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from glob import glob
from pathlib import Path
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error,
    explained_variance_score,
)
from rasterstats import zonal_stats

warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------
# main
#
# python validation_metrics_cli.py \
# --rasters-regex \
# '/explore/nobackup/people/mcarrol2/Lake_depth/Caleb_results/*.tif' \
# --points-regex \
# '/explore/nobackup/projects/ilab/scratch/mcarrol2/Lake_Depth/*.shp' \
# --output-filename val.csv
# -----------------------------------------------------------------------------
def extract_points(
            raster_data: xr.DataArray,
            points_df: gpd.GeoDataFrame
        ) -> pd.DataFrame:
    """
    Extract validation results from raster based on shapefile.
    :param raster_data: _description_
    :type raster_data: xr.DataArray
    :param points_df: _description_
    :type points_df: gpd.GeoDataFrame
    """
    validation_values = []  # List to store validated values
    raster_data = raster_data[0, :, :]  # Lower dimension to y,x
    for _, df_row in points_df.iterrows():
        # Extract validation point
        validation_value = raster_data.sel(
            x=df_row['geometry'].x,
            y=df_row['geometry'].y,
            method="nearest"
        )
        # Add validation value to list
        validation_values.append(validation_value.values)
    return pd.DataFrame(validation_values)


def assign_predicted_value(
            points_df,
            raster_filename,
            predicted_column_name,
            predicted_values,

        ):

    for index, row in points_df.iterrows():

        row_short_filename = '_'.join(
            row['short_filename'].split('_')[:5])

        raster_short_filename = '_'.join(
            Path(raster_filename).stem.split('_')[:5])

        if raster_short_filename == row_short_filename:
            points_df.at[index, predicted_column_name] = \
                predicted_values[index]
            points_df.at[
                index, f'{predicted_column_name}_Filename'] = \
                Path(raster_filename).stem
    return points_df


def run(args: argparse.Namespace) -> None:
    """
    Run validation of rasters.
    :param args: arguments given from argparse CLI interface
    :type args: argparse.Namespace
    """
    # -------------------------------------------------------------------------
    # Setup filenames to process
    # -------------------------------------------------------------------------
    raster_filenames = sorted(glob(args.rasters_regex))
    logging.info(f'Processing {len(raster_filenames)} rasters.')

    point_filenames = sorted(glob(args.points_regex))
    logging.info(f'Processing {len(point_filenames)} shapefile.')

    # Setup output directory
    if os.path.dirname(args.output_filename):
        os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)

    # -------------------------------------------------------------------------
    # Start processing filenames
    # -------------------------------------------------------------------------
    for point_filename in point_filenames:

        # open dataframe
        points_df = gpd.read_file(point_filename)
        points_df = points_df.to_crs(
            rxr.open_rasterio(raster_filenames[0]).rio.crs)
        logging.info(
            f'Processing {point_filename}, {points_df.shape} points.')

        # buffer geometry to extract zonal stat?

        # adding temporary columns for prediction
        points_df[args.predicted_column_name] = 10  # default for nodata
        points_df[f'{args.predicted_column_name}_Filename'] = \
            'filename'  # filename of prediction

        # iterate over rasters
        for raster_filename in raster_filenames:

            logging.info(f'Predicting raster: {Path(raster_filename).stem}')

            if points_df.geom_type[0] == 'Polygon':

                src_zonal_stats = zonal_stats(
                    points_df, raster_filename,
                    stats="majority"
                )
                predicted_values = [d['majority'] for d in src_zonal_stats]

            else:

                src = rasterio.open(raster_filename)

                coord_list = []
                for x, y in zip(
                        points_df['geometry'].x, points_df['geometry'].y):
                    coord_list.append((x, y))
                predicted_values = [x.item() for x in src.sample(coord_list)]

            # print(points_df.head(1))
            points_df = assign_predicted_value(
                points_df,
                raster_filename,
                args.predicted_column_name,
                predicted_values,
            )

    # output GPKG
    points_df.to_file(
        Path(args.output_filename).with_suffix('.gpkg'),
        driver='GPKG', layer='validation')

    # output CSV
    points_df.drop('geometry', axis=1).to_csv(
        Path(args.output_filename).with_suffix('.csv'))

    # -------------------------------------------------------------------------
    # Aggregate some metrics from the validation dataset
    # -------------------------------------------------------------------------

    # points_df = points_df[
    #    points_df[f'{args.predicted_column_name}_Filename'].str.contains(
    #        'filename' ) == False
    # ]

    """

    metrics_metadata = [
        r2_score,
        mean_absolute_error,
        mean_squared_error,
        mean_squared_log_error,
        mean_absolute_percentage_error,
        median_absolute_error,
        max_error,
        explained_variance_score
    ]

    metrics_df = pd.DataFrame(
        list(points_df.filter(regex=args.predicted_column_name)),
        columns=["Measure_Type"])

    for metric in metrics_metadata:
        temp_metrics_list = []
        for column_name in list(
                    points_df.filter(regex=args.predicted_column_name)):
            column_df = points_df.loc[points_df[column_name] >= 0]
            temp_metrics_list.append(
                metric(
                    column_df[args.mask_column_name],
                    column_df[column_name]
                )
            )
        metrics_df[metric.__name__] = temp_metrics_list

    metrics_output_filename = os.path.join(
        os.path.dirname(args.output_filename),
        f'{Path(args.output_filename).stem}_metrics.csv')
    metrics_df.to_csv(metrics_output_filename)
    """
    return


def main():

    # -------------------------------------------------------------------------
    # Process command-line args.
    # -------------------------------------------------------------------------
    desc = 'Use this application to preprocess label files.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--rasters-regex", type=str, required=True, dest='rasters_regex',
        help="Regex to load image files")

    parser.add_argument(
        "--points-regex", type=str, required=True, dest='points_regex',
        help="Regex to load label files")

    parser.add_argument(
        "--output-filename", type=str, required=True, dest='output_filename',
        help="Output filename to store validation output")

    parser.add_argument(
        "--predicted-column-name", type=str, required=False,
        dest='predicted_column_name', default="Pred_Class",
        help="Predicted value column name")

    parser.add_argument(
        "--mask-column-name", type=str, required=False,
        dest='mask_column_name', default="aggregated_class",
        help="True mask value column name")

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

    run(args)
    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
