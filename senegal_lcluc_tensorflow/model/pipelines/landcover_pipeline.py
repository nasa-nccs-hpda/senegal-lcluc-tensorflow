import os
import re
import csv
import sys
import time
import logging
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import matplotlib.colors as pltc
from rasterstats import zonal_stats
from rioxarray.merge import merge_arrays

from tqdm import tqdm
from glob import glob
from pathlib import Path
from itertools import repeat
from omegaconf import OmegaConf
from multiprocessing import Pool, cpu_count

from tensorflow_caney.utils.system import seed_everything
from tensorflow_caney.model.pipelines.cnn_regression import CNNRegression
from tensorflow_caney.utils.data import gen_random_tiles, \
    get_dataset_filenames, get_mean_std_dataset
from tensorflow_caney.model.dataloaders.regression import RegressionDataLoader
from tensorflow_caney.utils.analysis import confusion_matrix_func
from senegal_lcluc_tensorflow.model.config.landcover_config \
    import LandCoverConfig as Config
from tensorflow_caney.model.pipelines.cnn_segmentation import CNNSegmentation


class LandCoverPipeline(CNNSegmentation):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config_filename, data_csv=None, logger=None):

        # Set logger
        self.logger = logger if logger is not None else self._set_logger()

        # Configuration file intialization
        self.conf = self._read_config(config_filename, Config)

        # Set Data CSV
        self.data_csv = data_csv

        # Set experiment name
        try:
            self.experiment_name = self.conf.experiment_name.name
        except AttributeError:
            self.experiment_name = self.conf.experiment_name

        # output directory to store metadata and artifacts
        # self.metadata_dir = os.path.join(self.conf.data_dir, 'metadata')
        # self.logger.info(f'Metadata dir: {self.metadata_dir}')

        # Set output directories and locations
        # self.intermediate_dir = os.path.join(
        #    self.conf.data_dir, 'intermediate')
        # self.logger.info(f'Intermediate dir: {self.intermediate_dir}')

        self.images_dir = os.path.join(self.conf.data_dir, 'images')
        logging.info(f'Images dir: {self.images_dir}')

        self.labels_dir = os.path.join(self.conf.data_dir, 'labels')
        logging.info(f'Labels dir: {self.labels_dir}')

        self.model_dir = self.conf.model_dir
        logging.info(f'Model dir: {self.labels_dir}')

        # Create output directories
        for out_dir in [
                self.images_dir, self.labels_dir,
                self.model_dir]:
            os.makedirs(out_dir, exist_ok=True)

        # save configuration into the model directory
        OmegaConf.save(self.conf, os.path.join(self.model_dir, 'config.yaml'))

        # Seed everything
        seed_everything(self.conf.seed)

    def test(self, test_truth_regex: str):

        logging.info('Entering test pipeline step.')

        # set colormap
        colormap = pltc.ListedColormap(self.conf.test_colors)

        # get label filenames with truth
        label_filenames = sorted(glob(test_truth_regex))
        logging.info(f'Found {len(label_filenames)} truth filenames.')

        # get model filename for tracking
        if self.conf.model_filename is None or self.conf.model_filename == 'None':
            models_list = glob(os.path.join(self.conf.model_dir, '*.hdf5'))
            model_filename = max(models_list, key=os.path.getctime)
        else:
            model_filename = self.conf.model_filename

        # metrics output filename
        metrics_output_filename = os.path.join(
            self.conf.inference_save_dir,
            f'results_metrics_{os.path.basename(self.conf.data_dir)}' + \
                f'_{Path(model_filename).stem}.csv'
        )
        logging.info(f'Storing CSV metrics under {metrics_output_filename}')

        # metrics csv columns
        metrics_csv_columns = ['filename', 'overall-accuracy', 'accuracy']
        for test_metric in ['recall', 'precision', 'user-accuracy', 'producer-accuracy']:
            for class_name in self.conf.test_classes:
                metrics_csv_columns.append(f'{class_name}-{test_metric}')

        # open csv filename
        with open(metrics_output_filename, 'w') as metrics_filename:

            # write row to filename
            writer = csv.writer(
                metrics_filename, delimiter=',', lineterminator='\n')
            writer.writerow(metrics_csv_columns)

            # iterate over each label filename
            for label_filename in label_filenames:

                # get path to filename to test with
                filename = "_".join(Path(label_filename).stem.split('_')[:-1])

                # select filename to test with
                prediction_filename = glob(
                    os.path.join(
                        self.conf.inference_save_dir, '*', f'{filename}*.tif'))
                if len(prediction_filename) == 0:
                    continue

                # open label and prediction filenames
                label = np.squeeze(
                    rxr.open_rasterio(label_filename).values)
                prediction = np.squeeze(
                    rxr.open_rasterio(prediction_filename[0]).values)                
        
                # group label pixel to 4 classes
                # TODO: use self.conf.modify_test_labels
                label[label == 7] = 3
                label[label == 4] = 3
                label[label == 3] = 0
                label[label == 5] = 3

                # some preprocessing
                # TODO: use self.conf.modify_test_labels
                prediction[prediction == -10001] = 0
                prediction[prediction == 255] = 0

                # metrics report and test values
                metrics_report, cfn_matrix, accuracy, balanced_accuracy = \
                    confusion_matrix_func(
                        y_true=label, y_pred=prediction,
                        nclasses=len(self.conf.test_classes), norm=True
                    )

                # define row data to write into CSV
                row_data = [               
                    filename,
                    accuracy,
                    balanced_accuracy,
                    metrics_report['other-vegetation']['recall'],
                    metrics_report['tree']['recall'],
                    metrics_report['cropland']['recall'],
                    metrics_report['other-vegetation']['precision'],
                    metrics_report['tree']['precision'],
                    metrics_report['cropland']['precision']
                ]

                # two separate for loops because of the columns order
                for n in range(len(self.conf.test_classes)):
                    # user accuracy
                    row_data.append(round(cfn_matrix[n, n] / cfn_matrix[n, :].sum(), 2))
                
                for n in range(len(self.conf.test_classes)):
                    # producer accuracy
                    row_data.append(round(cfn_matrix[n, n] / cfn_matrix[:, n].sum(), 2))

                writer.writerow(row_data)

        logging.info(f'Metrics saved to {metrics_output_filename}')

        # plot_confusion_matrix(cnf_matrix, file_name, search_term_pf, class_names=classes)

        return

    def validate(self, validation_database: str = None):
        """
        Perform validation using georeferenced data
        """

        logging.info('Entering validation pipeline step.')

        # temporary variables
        pred_column = 'majority'
        val_column = 'aggregated_class'

        # read validation database
        gdf = gpd.read_file(validation_database).to_crs(
            self.conf.validation_epsg)
        
        # transform geometry from validation
        gdf['geometry'] = gdf.geometry.buffer(
            self.conf.validation_buffer_kernel,
            cap_style=self.conf.validation_cap_style
        )
        gdf.year = gdf.year.astype(int)

        logging.info(
            f'The database contains {len(gdf["scene_id"].unique())} individual scenes.')

        validation_database = []
        for unique_filename in gdf['short_filename'].unique():

            # get prediction filename
            filename = glob(
                os.path.join(self.conf.inference_save_dir, '*', f'{unique_filename}*.tif'))[0]

            # get zonal stats for the given geometry
            temporary_gdf = gdf[gdf['short_filename'] == unique_filename].copy()
            stats = zonal_stats(temporary_gdf, filename, categorical=True, stats="majority")

            # drive json from zonal stats into dataframe
            zonal_df = pd.DataFrame(stats)
            zonal_df['aggregated_class'] = temporary_gdf['aggregated_class'].to_list()
            zonal_df['short_filename'] = unique_filename

            # append dataframe to validation database
            validation_database.append(zonal_df)

        # convert the entire database in place
        validation_database = pd.concat(validation_database, axis=0).reset_index()

        # prioritize trees in some of these locations
        validation_database['majority'][validation_database[1] > 0] = 1

        # convert values from database into appropiate values
        # TODO: check this to make sure we know what we are doing with the burn class
        validation_database[val_column] = \
            validation_database[val_column].replace([3, 4, 5], [0, 0, 0])
        validation_database = \
             validation_database[validation_database[pred_column] != 3]

        # get model filename for tracking
        if self.conf.model_filename is None or self.conf.model_filename == 'None':
            models_list = glob(os.path.join(self.conf.model_dir, '*.hdf5'))
            model_filename = max(models_list, key=os.path.getctime)
        else:
            model_filename = self.conf.model_filename

        # metrics output filename
        metrics_output_filename = os.path.join(
            self.conf.inference_save_dir,
            f'results_val_{os.path.basename(self.conf.data_dir)}' + \
                f'_{Path(model_filename).stem}.csv'
        )
        logging.info(f'Storing CSV metrics under {metrics_output_filename}')

        # metrics csv columns
        metrics_csv_columns = ['filename', 'overall-accuracy', 'accuracy']
        for test_metric in ['recall', 'precision', 'user-accuracy', 'producer-accuracy']:
            for class_name in self.conf.test_classes:
                metrics_csv_columns.append(f'{class_name}-{test_metric}')
        metrics_csv_columns.append('n_points')

        # open csv filename
        with open(metrics_output_filename, 'w') as metrics_filename:

            # write row to filename
            writer = csv.writer(
                metrics_filename, delimiter=',', lineterminator='\n')
            writer.writerow(metrics_csv_columns)

            # iterate over each unique filename
            for unique_filename in validation_database['short_filename'].unique():

                # get temporary gdf
                temporary_gdf = validation_database[
                    validation_database['short_filename'] == unique_filename].copy()

                # metrics report and test values
                metrics_report, cfn_matrix, accuracy, balanced_accuracy = \
                    confusion_matrix_func(
                        y_true=temporary_gdf[val_column].values,
                        y_pred=temporary_gdf[pred_column].values,
                        nclasses=len(self.conf.test_classes),
                        norm=True,
                        sample_points=False
                    )

                # define row data to write into CSV
                row_data = [               
                    unique_filename,
                    accuracy,
                    balanced_accuracy,
                    metrics_report['other-vegetation']['recall'],
                    metrics_report['tree']['recall'],
                    metrics_report['cropland']['recall'],
                    metrics_report['other-vegetation']['precision'],
                    metrics_report['tree']['precision'],
                    metrics_report['cropland']['precision'],
                ]

                # two separate for loops because of the columns order
                for n in range(len(self.conf.test_classes)):
                    # user accuracy
                    row_data.append(round(cfn_matrix[n, n] / cfn_matrix[n, :].sum(), 2))
                
                for n in range(len(self.conf.test_classes)):
                    # producer accuracy
                    row_data.append(round(cfn_matrix[n, n] / cfn_matrix[:, n].sum(), 2))

                # append number of points
                row_data.append(temporary_gdf.shape[0])

                writer.writerow(row_data)

        logging.info(f'Metrics saved to {metrics_output_filename}')

        # plot_confusion_matrix(cnf_matrix, file_name, search_term_pf, class_names=classes)

        return 
