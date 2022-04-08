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
from pathlib import Path

import numpy as np
import cupy as cp
import pandas as pd
import xarray as xr
import rioxarray as rxr
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau

sys.path.append('/adapt/nobackup/people/jacaraba/development/tensorflow-caney')

from tensorflow_caney.config.cnn_config import Config
from tensorflow_caney.utils.system import seed_everything, set_gpu_strategy
from tensorflow_caney.utils.system import set_mixed_precision, set_xla
from tensorflow_caney.utils.data import get_dataset_filenames
from tensorflow_caney.utils.segmentation_tools import SegmentationDataLoader

from tensorflow_caney.networks.unet import unet_batchnorm as unet
from tensorflow_caney.utils.losses import get_loss
from tensorflow_caney.utils.optimizers import get_optimizer
from tensorflow_caney.utils.metrics import get_metrics

# ---------------------------------------------------------------------------
# script train.py
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace, conf: omegaconf.dictconfig.DictConfig) -> None:
    """
    Run training steps.

    Possible additions to this process:
        - callbacks from functions
        - model from functions
    """
    logging.info('Starting training stage')

    # set data variables for directory management
    images_dir = os.path.join(conf.data_dir, 'images')
    labels_dir = os.path.join(conf.data_dir, 'labels')
    
    # Set and create model directory
    model_dir = os.path.join(conf.data_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    # Set hardware acceleration options
    gpu_strategy = set_gpu_strategy(conf.gpu_devices)
    set_mixed_precision(conf.mixed_precision)
    set_xla(conf.xla)

    # Get data and label filenames for training
    data_filenames = get_dataset_filenames(images_dir)
    label_filenames = get_dataset_filenames(labels_dir)
    assert len(data_filenames) == len(label_filenames), \
        f'Number of data and label filenames do not match'
    
    logging.info(
        f'Data: {len(data_filenames)}, Label: {len(label_filenames)}')

    # Set main data loader
    main_data_loader = SegmentationDataLoader(
        data_filenames, label_filenames, conf
    )

    # Set multi-GPU training strategy
    with gpu_strategy.scope():
        
        # TODO: add unet maps on the configuration file from the model
        #       add get model option?
        #       add additional model options to work with
        model = unet(
            nclass=conf.n_classes,
            input_size=(
                conf.tile_size, conf.tile_size, len(conf.output_bands)
            ),
            maps=[64, 128, 256, 512, 1024]
        )

        # Compile the model
        model.compile(
            loss=get_loss(conf.loss),
            optimizer=get_optimizer(conf.optimizer)(conf.learning_rate),
            metrics=get_metrics(conf.metrics)
        )
        model.summary()

        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(
                    model_dir, '{epoch:02d}-{val_loss:.2f}.hdf5'),
                monitor='val_acc',
                mode='max',
                save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
            CSVLogger(
                os.path.join(model_dir, f"{conf.experiment_name}.csv")),
            TensorBoard(
                log_dir=os.path.join(model_dir, 'tensorboard_logs')),
            EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=False
            )
        ]

    # Fit the model and start training
    model.fit(
        main_data_loader.train_dataset,
        validation_data=main_data_loader.val_dataset,
        epochs=conf.max_epochs,
        steps_per_epoch=main_data_loader.train_steps,
        validation_steps=main_data_loader.val_steps,
        callbacks=callbacks
    )

    # Close multiprocessing Pools from the background
    atexit.register(gpu_strategy._extended._collective_ops._pool.close)

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
    logging.info('Done with training stage')

    return

# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    main()
