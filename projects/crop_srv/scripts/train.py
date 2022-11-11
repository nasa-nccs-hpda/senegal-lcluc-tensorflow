# --------------------------------------------------------------------------
# Training of vhr data. This assumes you provide
# a configuration file with required parameters and files.
# --------------------------------------------------------------------------
import os
import sys
import time
import atexit
import logging
import argparse
import omegaconf
import tensorflow as tf

from packaging.version import parse as parse_version
from tensorflow_caney.config.cnn_config import Config
from tensorflow_caney.utils.system import seed_everything, set_gpu_strategy
from tensorflow_caney.utils.system import set_mixed_precision, set_xla
from tensorflow_caney.utils.data import get_dataset_filenames
from tensorflow_caney.utils.segmentation_tools import SegmentationDataLoader

from tensorflow_caney.utils.losses import get_loss
from tensorflow_caney.utils.optimizers import get_optimizer
from tensorflow_caney.utils.metrics import get_metrics
from tensorflow_caney.utils.callbacks import get_callbacks
from tensorflow_caney.utils.model import get_model, load_model


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
        - plot out of training metrics
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
        'Number of data and label filenames do not match'

    logging.info(
        f'Data: {len(data_filenames)}, Label: {len(label_filenames)}')

    # Set main data loader
    main_data_loader = SegmentationDataLoader(
        data_filenames, label_filenames, conf
    )

    # Set multi-GPU training strategy
    with gpu_strategy.scope():

        transfer_learning = 'fine-tuningxxxx'
        if transfer_learning == 'fine-tuning':
            model = get_model(conf.model)
            model.trainable = False
            model_2 = load_model(
                model_filename=conf.model_filename,
                model_dir=os.path.join(conf.data_dir, 'model')
            )

            model.set_weights(model_2.get_weights())

            # print(dir(model))
            # model = model(model, training=False)
            model.trainable = True
            model.compile(
                loss=get_loss(conf.loss),
                optimizer=get_optimizer(conf.optimizer)(conf.learning_rate),
                metrics=get_metrics(conf.metrics)
            )
        else:
            # Get and compile the model
            model = get_model(conf.model)
            model.compile(
                loss=get_loss(conf.loss),
                optimizer=get_optimizer(conf.optimizer)(conf.learning_rate),
                metrics=get_metrics(conf.metrics)
            )

    model.summary()

    # Fit the model and start training
    model.fit(
        main_data_loader.train_dataset,
        validation_data=main_data_loader.val_dataset,
        epochs=conf.max_epochs,
        steps_per_epoch=main_data_loader.train_steps,
        validation_steps=main_data_loader.val_steps,
        callbacks=get_callbacks(conf.callbacks)
    )

    # Close multiprocessing Pools from the background
    if parse_version(tf.__version__) > parse_version('2.4'):
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
    timer = time.time()
    run(args, conf)
    logging.info(
        f'Done with training, took {(time.time()-timer)/60.0:.2f} min.')

    return


# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    main()
