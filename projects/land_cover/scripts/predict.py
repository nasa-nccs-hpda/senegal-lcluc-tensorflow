# --------------------------------------------------------------------------
# Prediction of vhr data. This assumes you provide
# a configuration file with required parameters and files.
# --------------------------------------------------------------------------
import os
import sys
import time
import logging
import argparse
import omegaconf
from glob import glob
from pathlib import Path

import numpy as np
import cupy as cp
import pandas as pd
import xarray as xr
import rioxarray as rxr
import tensorflow as tf


sys.path.append('/adapt/nobackup/people/jacaraba/development/tensorflow-caney')

from tensorflow_caney.config.cnn_config import Config
from tensorflow_caney.utils.system import seed_everything, set_gpu_strategy
from tensorflow_caney.utils.system import set_mixed_precision, set_xla
from tensorflow_caney.utils.data import get_dataset_filenames
from tensorflow_caney.utils.segmentation_tools import SegmentationDataLoader

from tensorflow_caney.networks.unet import unet_batchnorm as unet
from tensorflow_caney.networks.loss import get_loss

from tensorflow_caney.utils.data import modify_bands
from tensorflow_caney.utils import indices
from tensorflow_caney.inference import inference

# ---------------------------------------------------------------------------
# script train.py
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace, conf: omegaconf.dictconfig.DictConfig) -> None:
    """
    Run training steps.

    Possible additions to this process:
        - TBD
    """
    logging.info('Starting prediction stage')
    
    # Set and create model directory
    os.makedirs(conf.inference_save_dir, exist_ok=True)
    
    # TODO: get last model if no model filename was given
    # try:
    #    self.conf.model_filename
    # except AttributeError:
    #    models_list = glob.glob(os.path.join(self.model_dir, '*.pt'))
    #    self.model_filename = max(models_list, key=os.path.getctime)
    # logging.info(f'Loading {self.model_filename}')

    # Loading the trained model
    assert os.path.isfile(conf.model_filename), \
        f'{conf.model_filename} does not exist.'

    # Set hardware acceleration options
    gpu_strategy = set_gpu_strategy(conf.gpu_devices)

    with gpu_strategy.scope():

        model = tf.keras.models.load_model(
            conf.model_filename, custom_objects={}
            #    # "_iou": self._iou,
            #    "TverskyLoss": TverskyLoss()
            #    }
        )

    # Gather filenames to predict
    data_filenames = sorted(glob(conf.inference_regex))
    assert len(data_filenames) > 0, \
        f'No files under {conf.inference_regex}.'
    logging.info(f'{len(data_filenames)} files to predict')

    #if self.conf.standardize:
    #    self.conf.mean = np.load(
    #        os.path.join(
    #            self.conf.data_dir,
    #            f'mean-{self.conf.experiment_name}.npy')).tolist()
    #    self.conf.std = np.load(
    #        os.path.join(
    #            self.conf.data_dir,
    #            f'std-{self.conf.experiment_name}.npy')).tolist()
    #logging.info(f'Mean: {self.conf.mean}, Std: {self.conf.std}')

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

            logging.info(f'Starting to predict {filename}')

            # create lock file
            open(lock_filename, 'w').close()

            # open filename
            image = rxr.open_rasterio(filename)
            image = image.transpose("y", "x", "band")
            logging.info(f'Prediction shape: {image.shape}')

            # TODO: CALCULATE INDICES

            image = modify_bands(
                xraster=image, input_bands=conf.input_bands,
                output_bands=conf.output_bands)
            logging.info(f'Prediction shape after modf: {image.shape}')

            prediction = inference.sliding_window(
                xraster=image.values,
                model=model,
                window_size=conf.window_size,
                tile_size=conf.tile_size,
                inference_overlap=conf.inference_overlap,
                inference_treshold=conf.inference_treshold,
                batch_size=conf.pred_batch_size,
                mean=conf.mean,
                std=conf.std,
                n_classes=conf.n_classes
            )

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
            prediction = prediction.transpose("band", "y", "x")

            # Set nodata values on mask
            nodata = prediction.rio.nodata
            prediction = prediction.where(image != nodata)
            prediction.rio.write_nodata(nodata, encoded=True, inplace=True)

            # Save COG file to disk
            prediction.rio.to_raster(
                output_filename, BIGTIFF="IF_SAFER", compress='LZW',
                num_threads='all_cpus')#, driver='COG')

            del prediction

            # delete lock file
            os.remove(lock_filename)

            logging.info(f"{(time.time() - start_time)/60} min")

        # This is the case where the prediction was already saved
        else:
            logging.info(f'{output_filename} already predicted.')

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
