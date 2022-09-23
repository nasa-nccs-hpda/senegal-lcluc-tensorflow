# --------------------------------------------------------------------------
# Preprocessing and dataset creation from vhr data. This assumes you provide
# a configuration file with required parameters and files.
# --------------------------------------------------------------------------
import os
import sys
import logging
import argparse
import omegaconf
from glob import glob
from pathlib import Path

import numpy as np
import cupy as cp
import rioxarray as rxr

from tensorflow_caney.config.cnn_config import Config
from tensorflow_caney.utils.system import seed_everything
from tensorflow_caney.utils.data import gen_random_tiles, \
    gen_random_tiles_from_json, read_dataset_csv
from tensorflow_caney.utils.data import modify_bands, get_dataset_filenames
from tensorflow_caney.utils.data import modify_label_classes
from tensorflow_caney.utils.data import get_mean_std_dataset
from tensorflow_caney.utils.data import normalize_image, rescale_image
from tensorflow_caney.utils.segmentation_tools import SegmentationDataLoader

from tensorflow_caney.utils import indices


CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}

__status__ = "Development"


# ---------------------------------------------------------------------------
# script preprocess.py
# ---------------------------------------------------------------------------
def run(
            args: argparse.Namespace, conf: omegaconf.dictconfig.DictConfig
        ) -> None:
    """
    Run preprocessing steps.

    Possible additions to this process:
        - additional flexibility for golden tiles
        - enable indices calculation
    """
    logging.info('Starting preprocessing stage')

    # Initialize dataframe with data details
    data_df = read_dataset_csv(args.data_csv)
    logging.info(data_df)

    # Set output directories and locations
    images_dir = os.path.join(conf.data_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    labels_dir = os.path.join(conf.data_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    # iterate over each file and generate dataset
    for index_id, (data_filename, label_filename, n_tiles) \
            in enumerate(data_df.values):

        logging.info(f'Processing {Path(data_filename).stem}')

        # Read imagery from disk and process both image and mask
        image = rxr.open_rasterio(data_filename, chunks=CHUNKS).load()
        label = rxr.open_rasterio(label_filename, chunks=CHUNKS).values
        logging.info(f'Image: {image.shape}, Label: {label.shape}')

        # Calculate indices and append to the original raster
        image = indices.add_indices(
            xraster=image, input_bands=conf.input_bands,
            output_bands=conf.output_bands)

        # Lower the number of bands if required
        image = modify_bands(
            xraster=image, input_bands=conf.input_bands,
            output_bands=conf.output_bands)
        logging.info(f'Image: {image.shape}, Label: {label.shape}')

        # Asarray option to force array type
        image = cp.asarray(image.values)
        label = cp.asarray(label)

        # Move from chw to hwc, squeze mask if required
        image = cp.moveaxis(image, 0, -1)
        label = cp.squeeze(label) if len(label.shape) != 2 else label
        logging.info(f'Label classes from image: {cp.unique(label)}')

        # Normalize values within [0, 1] range
        image = normalize_image(image, conf.normalize)

        # Rescale values within [0, 1] range
        image = rescale_image(image, conf.rescale)

        # Modify labels, sometimes we need to merge some training classes
        # Substract values if classes do not start from 0, this is done first
        label = modify_label_classes(
            label, conf.modify_labels, conf.substract_labels)
        logging.info(f'Label classes after modify_labels: {cp.unique(label)}')

        # Making labels int type and grabbing some information
        label = label.astype(np.uint8)
        logging.info(f'Label classes min {label.min()}, max {label.max()}')

        # generate random tiles
        gen_random_tiles(
            image=image,
            label=label,
            expand_dims=conf.expand_dims,
            tile_size=conf.tile_size,
            index_id=index_id,
            num_classes=conf.n_classes,
            max_patches=n_tiles,
            include=conf.include_classes,
            augment=conf.augment,
            output_filename=data_filename,
            out_image_dir=images_dir,
            out_label_dir=labels_dir,
            json_tiles_dir=conf.json_tiles_dir,
            dataset_from_json=conf.dataset_from_json
        )

    # Calculate mean and std values for training
    data_filenames = get_dataset_filenames(images_dir)
    label_filenames = get_dataset_filenames(labels_dir)
    logging.info(f'Mean and std values from {len(data_filenames)} files.')

    # Temporarily disable standardization and augmentation
    conf.standardization = None
    metadata_output_filename = os.path.join(
        conf.model_dir, f'mean-std-{conf.experiment_name}.csv')
    os.makedirs(conf.model_dir, exist_ok=True)

    # Set main data loader
    main_data_loader = SegmentationDataLoader(
        data_filenames, label_filenames, conf, False
    )

    # Get mean and std array
    mean, std = get_mean_std_dataset(
        main_data_loader.train_dataset, metadata_output_filename)
    logging.info(f'Mean: {mean.numpy()}, Std: {std.numpy()}')

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
                        required=True,
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
    logging.info('Done with preprocessing stage')

    return


# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    main()
