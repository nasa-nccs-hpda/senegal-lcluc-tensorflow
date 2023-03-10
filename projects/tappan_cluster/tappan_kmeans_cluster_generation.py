
import argparse
import logging
import os
import tqdm

import numpy as np
import omegaconf
from osgeo import gdal
from sklearn import cluster


from core.model.SystemCommand import SystemCommand
from core.model.GeospatialImageFile import GeospatialImageFile
from utils import TqdmLoggingHandler, getBaseCmd


def setup_logging(log_file: str) -> logging.Logger:
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler for the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a stream handler for stdout
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.DEBUG)

    # Define the log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def read_clipped_reshape(clipped_path: str,
                         logger: logging.Logger) -> np.ndarray:
    logger.info(f'Reading in {clipped_path}')
    clipped_dataset = gdal.Open(clipped_path)
    if clipped_dataset is None:
        msg = f'Could not open {clipped_dataset} with GDAL'
        logger.error(msg)
        raise RuntimeError(msg)
    clipped_dataset_array = clipped_dataset.ReadAsArray()
    logger.debug(
        f'Clipped shape before transpose: {clipped_dataset_array.shape}')
    channel_last_array = np.transpose(clipped_dataset_array, (1, 2, 0))
    output_shape = channel_last_array[:, :, 0].shape
    logger.debug(f'Clipped shape after transpose: {channel_last_array.shape}')
    img1d_channels = channel_last_array.reshape(-1,
                                                channel_last_array.shape[-1])
    logger.debug(f'Clipped shape after 1d trf: {img1d_channels.shape}')
    return img1d_channels, output_shape


def agglomerative(conf: dict,
                  clipped_path: str,
                  logger: logging.Logger) -> np.ndarray:
    raise NotImplementedError()


def write_out_clustered_img(cluster_image: np.ndarray,
                            clipped_image_path: str,
                            algorithm: str,
                            output_dir: str,
                            logger: logging.Logger) -> int:
    clipped_image_dataset = GeospatialImageFile(
        clipped_image_path).getDataset()
    clipped_image_name = os.path.basename(clipped_image_path)
    logger.debug(f'Cluster image shape: {cluster_image.shape}')
    post_str = f'{algorithm}.tif'
    output_name = clipped_image_name.replace('.tif', post_str)
    output_path = os.path.join(output_dir, output_name)
    logger.info(f'Writing to {output_path}')
    output_srs = clipped_image_dataset.GetProjection()
    output_trf = clipped_image_dataset.GetGeoTransform()
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(output_path,
                           cluster_image.shape[1],
                           cluster_image.shape[0],
                           1,
                           gdal.GDT_Byte)
    dst_ds.SetProjection(output_srs)
    dst_ds.SetGeoTransform(output_trf)
    dst_band = dst_ds.GetRasterBand(1)
    dst_band.WriteArray(cluster_image.astype(np.uint8))
    dst_band.FlushCache()
    dst_ds.FlushCache()
    return 1


def kmeans(conf: dict,
           clipped_path: str,
           logger: logging.Logger) -> np.ndarray:

    img1d_channels, output_shape = read_clipped_reshape(clipped_path, logger)
    params = {
        'n_clusters': conf['num_clusters'],
        'random_state': conf['random_state'],
        'batch_size': conf['batch_size'],
    }

    cl_kmeans = cluster.MiniBatchKMeans(**params)
    model = cl_kmeans.fit(img1d_channels)

    img_cl = model.labels_
    img_cl = img_cl.reshape(output_shape)

    return img_cl


def clip_geotiff(
        input_path: str,
        output_path: str,
        upper_left_x: float,
        upper_left_y: float,
        window_size_x: int,
        window_size_y: int,
        logger: logging.Logger) -> int:

    dataset = gdal.Open(input_path)
    dataset_geotransform = dataset.GetGeoTransform()

    ulx = upper_left_x
    uly = upper_left_y

    x_scale = dataset_geotransform[1]
    y_scale = dataset_geotransform[5]

    lrx = ulx + window_size_x * x_scale
    lry = uly + window_size_y * y_scale

    cmd = getBaseCmd(dataset)

    cmd += (' -te' +
            ' ' + str(ulx) +
            ' ' + str(lry) +
            ' ' + str(lrx) +
            ' ' + str(uly) +
            ' -te_srs' +
            ' "' + dataset.GetSpatialRef().ExportToProj4() +
            '"')
    cmd += ' ' + input_path + ' ' + output_path
    SystemCommand(cmd, logger, True)
    return 1


def get_output_name(input_path: str,
                    input_identifier: str,
                    square_number: int,
                    output_pre_str: str,
                    output_dir: str) -> str:
    filename = os.path.basename(input_path)
    filename_id = filename.replace(input_identifier, '')
    output_filename = f"{output_pre_str}{square_number}_{filename_id}_data.tif"
    output_filepath = os.path.join(output_dir, output_filename)
    return output_filepath


def run_one_clip(input_path: str, conf: dict, logger: logging.Logger) -> str:
    output_path = get_output_name(
        input_path,
        conf['input_identifier'],
        conf['square_number'],
        conf['output_pre_str'],
        conf['output_dir'])
    if not os.path.exists(output_path):
        logger.info(f'{output_path} does not exist. Making.')
        clip_geotiff(input_path,
                     output_path,
                     conf['upper_left_x'],
                     conf['upper_left_y'],
                     conf['window_size_x'],
                     conf['window_size_y'],
                     logger=logger)
    else:
        logger.info(f'{output_path} already exists')
    return output_path


def open_readlines_file(file_path: str, logger: logging.Logger) -> list:
    if not os.path.exists(file_path):
        msg = f'No file: {file_path}'
        logger.error(msg)
        raise FileNotFoundError(msg)
    with open(file_path, 'r') as file_handler:
        files_to_process = file_handler.readlines()
        files_to_process = [file_to_process.strip()
                            for file_to_process in files_to_process]
        logger.debug(files_to_process)
    return files_to_process


def main(config_file: str) -> int:
    conf = omegaconf.OmegaConf.load(config_file)

    square_number = conf['square_number']
    output_dir = conf['output_dir']

    log_file_name = f'clipCluster{square_number}.log'
    log_file_name = os.path.join(output_dir, log_file_name)

    logger = setup_logging(log_file_name)

    logger.debug(f'Configuration: {conf}')
    input_txt_file = conf['input_txt_file']
    files_to_clip = open_readlines_file(input_txt_file, logger)
    logger.debug(f'Clipping {len(files_to_clip)} files')
    for file_to_clip in tqdm.tqdm(files_to_clip):
        if conf['clip']:
            logger.info('Clipping')
            output_path = run_one_clip(file_to_clip, conf, logger)
        else:
            logger.info('Not clipping')
            output_path = file_to_clip
        if conf['clustering']:
            algorithm = conf['algorithm']
            logger.info(f'Clustering {output_path} using {algorithm}')
            image_clustered = kmeans(conf, output_path, logger)
            write_out_clustered_img(
                image_clustered, output_path, algorithm, output_dir, logger)

    return 1


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        dest='config',
                        type=str,
                        required=True,
                        help='Path to YAML configuration file')
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(args.config)
