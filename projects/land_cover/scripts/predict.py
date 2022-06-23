# --------------------------------------------------------------------------
# Prediction of vhr data. This assumes you provide
# a configuration file with required parameters and files.
# --------------------------------------------------------------------------
import os
import sys
import time
import atexit
import logging
import argparse
import omegaconf
import rasterio
from glob import glob
from pathlib import Path

import numpy as np
import cupy as cp
import pandas as pd
import xarray as xr
import rioxarray as rxr
import tensorflow as tf

from tensorflow_caney.config.cnn_config import Config
from tensorflow_caney.utils.system import seed_everything
from tensorflow_caney.utils.model import load_model

from tensorflow_caney.utils.data import modify_bands, \
    get_mean_std_metadata
from tensorflow_caney.utils import indices
from tensorflow_caney.inference import inference


# ---------------------------------------------------------------------------
# script train.py
# ---------------------------------------------------------------------------
def run(
            args: argparse.Namespace,
            conf: omegaconf.dictconfig.DictConfig
        ) -> None:
    """
    Run training steps.
    Possible additions to this process:
        - TBD
    """
    logging.info('Starting prediction stage')

    # Set and create model directory
    os.makedirs(conf.inference_save_dir, exist_ok=True)

    # Load model for inference
    model = load_model(
        model_filename=conf.model_filename,
        model_dir=os.path.join(conf.data_dir, 'model')
    )

    # Gather filenames to predict
    data_filenames = sorted(glob(conf.inference_regex))
    assert len(data_filenames) > 0, \
        f'No files under {conf.inference_regex}.'
    logging.info(f'{len(data_filenames)} files to predict')

    # Retrieve mean and std, there should be a more ideal place
    # TEMPORARY FILE FOR METADATA, TIRED AND NEED TO LEAVE THIS RUNNING
    # CHANGE TO MODEL_DIR IN THE OTHER SCRIPTS
    if conf.standardization in ["global", "mixed"]:
        mean, std = get_mean_std_metadata(
            os.path.join(
                conf.model_dir, f'mean-std-{conf.experiment_name}.csv')
        )
        logging.info(f'Mean: {mean}, Std: {std}')
    else:
        mean = None
        std = None

    # iterate files, create lock file to avoid predicting the same file
    for filename in data_filenames:

        start_time = time.time()

        # output filename to save prediction on
        output_filename = os.path.join(
            conf.inference_save_dir,
            f'{Path(filename).stem}.{conf.experiment_type}.tif'
        )

        # lock file for multi-node, multi-processing
        lock_filename = f'{output_filename}.lock'

        # predict only if file does not exist and no lock file
        if not os.path.isfile(output_filename) and \
                not os.path.isfile(lock_filename):

            try:

                logging.info(f'Starting to predict {filename}')

                # create lock file
                open(lock_filename, 'w').close()

                # open filename
                image = rxr.open_rasterio(filename)
                logging.info(f'Prediction shape: {image.shape}')

            except rasterio.errors.RasterioIOError:
                logging.info(f'Skipped {filename}, probably corrupted.')
                continue

            # Calculate indices and append to the original raster
            image = indices.add_indices(
                xraster=image, input_bands=conf.input_bands,
                output_bands=conf.output_bands)

            # Modify the bands to match inference details
            image = modify_bands(
                xraster=image, input_bands=conf.input_bands,
                output_bands=conf.output_bands)
            logging.info(f'Prediction shape after modf: {image.shape}')

            # Transpose the image for channel last format
            image = image.transpose("y", "x", "band")

            # Remove no-data values to account for edge effects
            # temporary_tif = image.values
            temporary_tif = xr.where(image > -100, image, 600)

            prediction = inference.sliding_window_tiler_multiclass(
                xraster=temporary_tif,
                model=model,
                n_classes=conf.n_classes,
                overlap=0.50,
                batch_size=conf.pred_batch_size,
                standardization=conf.standardization,
                mean=mean,
                std=std,
                normalize=conf.normalize
            )
            # logging.info(f'Prediction unique values {np.unique(prediction)}')
            # logging.info(f'Done with prediction')

            # Call inference function
            # prediction = inference.sliding_window(
            #    xraster=temporary_tif.data,
            #    model=model,
            #    window_size=conf.window_size,
            #    tile_size=conf.tile_size,
            #    inference_overlap=conf.inference_overlap,
            #    inference_treshold=conf.inference_treshold,
            #    batch_size=conf.pred_batch_size,
            #    mean=conf.mean,
            #    std=conf.std,
            #    n_classes=conf.n_classes,
            #    standardization=conf.standardization,
            #    normalize=conf.normalize
            # )

            # Drop image band to allow for a merge of mask
            image = image.drop(
                dim="band",
                labels=image.coords["band"].values[1:],
                drop=True
            )

            # Get metadata to save raster
            prediction = xr.DataArray(
                np.expand_dims(prediction, axis=-1),
                name=conf.experiment_type,
                coords=image.coords,
                dims=image.dims,
                attrs=image.attrs
            )
            prediction.attrs['long_name'] = (conf.experiment_type)
            prediction.attrs['model_name'] = (conf.model_filename)
            prediction = prediction.transpose("band", "y", "x")

            # Set nodata values on mask
            nodata = prediction.rio.nodata
            prediction = prediction.where(image != nodata)
            prediction.rio.write_nodata(nodata, encoded=True, inplace=True)

            # Save COG file to disk
            prediction.rio.to_raster(
                output_filename, BIGTIFF="IF_SAFER", compress='LZW',
                num_threads='all_cpus'  # , driver='COG'
            )

            del prediction

            # delete lock file
            try:
                os.remove(lock_filename)
            except FileNotFoundError:
                logging.info(f'Lock file not found {lock_filename}')
                continue

            logging.info(f'Finished processing {output_filename}')
            logging.info(f"{(time.time() - start_time)/60} min")

        # This is the case where the prediction was already saved
        else:
            logging.info(f'{output_filename} already predicted.')

    # Close multiprocessing Pools from the background
    # atexit.register(gpu_strategy._extended._collective_ops._pool.close)

    return


def main() -> None:

    # Process command-line args.
    desc = 'Use this application to map LCLUC in Senegal using WV data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file')

    parser.add_argument('-d',
                        '--data-csv',
                        type=str,
                        required=False,
                        dest='data_csv',
                        help='Path to the data CSV configuration file')

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

    # Configuration file intialization
    schema = omegaconf.OmegaConf.structured(Config)
    conf = omegaconf.OmegaConf.load(args.config_file)
    try:
        conf = omegaconf.OmegaConf.merge(schema, conf)
    except BaseException as err:
        sys.exit(f"ERROR: {err}")

    # Seed everything
    seed_everything(conf.seed)

    # Call run for preprocessing steps
    run(args, conf)

    return


# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    main()
