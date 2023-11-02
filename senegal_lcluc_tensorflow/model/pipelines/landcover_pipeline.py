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

import warnings
import itertools
import tensorflow as tf
import matplotlib.colors as pltc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm
from glob import glob
from pathlib import Path
from itertools import repeat
from omegaconf import OmegaConf
from multiprocessing import Pool, cpu_count
from sklearn.exceptions import UndefinedMetricWarning

from tensorflow_caney.utils.system import seed_everything
from tensorflow_caney.model.pipelines.cnn_regression import CNNRegression
from tensorflow_caney.utils.data import gen_random_tiles, \
    get_dataset_filenames, get_mean_std_dataset
from tensorflow_caney.model.dataloaders.regression import RegressionDataLoader
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
        try:
            OmegaConf.save(
                self.conf, os.path.join(self.model_dir, 'config.yaml'))
        except PermissionError:
            logging.info('No permissions to save config, skipping step.')

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
                    self.confusion_matrix_func(
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

        # assert the validation database filename
        assert os.path.isfile(validation_database), \
            f'{validation_database} does not exist.'
        logging.info(f'Given validation database {validation_database}')

        # read validation database
        gdf = gpd.read_file(validation_database).to_crs(
            self.conf.validation_epsg)
        
        # transform geometry from validation
        gdf['geometry'] = gdf.geometry.buffer(
            self.conf.validation_buffer_kernel,
            cap_style=self.conf.validation_cap_style
        )
        gdf.year = gdf.year.astype(int)

        # get some metadata from the database
        logging.info(f'Validating {gdf.shape[0]} observations')
        logging.info(f'Found {len(gdf["scene_id"].unique())} individual locations.')

        # Iterate over unique filenames for validation
        validation_database = []
        for unique_filename in gdf['short_filename'].unique():
                
            logging.info(f'Processing {unique_filename}')

            # get a filename regex 
            filename_regex = os.path.join(
                self.conf.inference_save_dir, '*',
                f'{unique_filename.replace("_data", "")}*.{self.conf.experiment_type}.tif'
            )
            
            # attempt to match filename
            try:
                # get the first element from the glob, prediction filename
                filename = glob(filename_regex)[0]
            except IndexError:
                continue
                
            # get zonal stats for the given geometry
            temporary_gdf = gdf[gdf['short_filename'] == unique_filename].copy()
            stats = zonal_stats(temporary_gdf, filename, categorical=True, stats="majority")

            # drive json from zonal stats into dataframe
            zonal_df = pd.DataFrame(stats)
            zonal_df[val_column] = temporary_gdf[val_column].to_list()
            zonal_df['short_filename'] = unique_filename

            # append dataframe to validation database
            validation_database.append(zonal_df)

        assert len(validation_database) > 0, \
            "Validation database is empty, check filenames to match."

        # convert the entire database in place
        validation_database = pd.concat(validation_database, axis=0).reset_index()
        logging.info(f'The size of the validation observations {validation_database.shape[0]}')

        # prioritize trees in some of these locations
        validation_database['majority'][validation_database[1] > 0] = 1

        # convert values from database into appropiate values
        # TODO: check this to make sure we know what we are doing with the burn class
        validation_database[val_column] = \
            validation_database[val_column].replace([3, 4, 5], [0, 0, 0])

        # TODO: check this to make sure we know what we are doing with the burn class
        # here we are basically excluding observations from burn areas
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

                # if the validation database is empty, skip this step
                if temporary_gdf.shape[0] == 0:
                    continue

                # metrics report and test values
                metrics_report, cfn_matrix, accuracy, balanced_accuracy = \
                    self.confusion_matrix_func(
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

                # get overall weighted accuracy
                with warnings.catch_warnings():

                    # ignore the UndefinedMetricWarning
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    # two separate for loops because of the columns order
                    for n in range(len(self.conf.test_classes)):
                        # user accuracy
                        try:
                            row_data.append(round(cfn_matrix[n, n] / cfn_matrix[n, :].sum(), 2))
                        except:
                            row_data.append(np.nan)
                
                    for n in range(len(self.conf.test_classes)):
                        # producer accuracy
                        try:
                            row_data.append(round(cfn_matrix[n, n] / cfn_matrix[:, n].sum(), 2))
                        except:
                            row_data.append(np.nan)

                # append number of points
                row_data.append(temporary_gdf.shape[0])

                writer.writerow(row_data)

        logging.info(f'Metrics saved to {metrics_output_filename}')

        # plot_confusion_matrix(cnf_matrix, file_name, search_term_pf, class_names=classes)

        return 

    def confusion_matrix_func(
                self,
                y_true: list = [],
                y_pred: list = [],
                nclasses: int = 3,
                norm: bool = True,
                sample_points: bool = True,
                percent: float = 0.2,
            ):
        """
        Args:
            y_true:   2D numpy array with ground truth
            y_pred:   2D numpy array with predictions (already processed)
            nclasses: number of classes
            Returns:
                numpy array with confusion matrix
        """

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        label_name = [0, 1, 2]
        label_dict = np.unique(y_true, return_counts=True)

        # get overall weighted accuracy
        with warnings.catch_warnings():

            # ignore the UndefinedMetricWarning
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # get subset of pixel
            if sample_points:

                # ind3 = np.random.choice(
                #   np.where(y_true == 3)[0], round(label_dict[1][3] * percent))
                ind2 = np.random.choice(
                    np.where(y_true == 2)[0], round(label_dict[1][2] * percent))
                ind1 = np.random.choice(
                    np.where(y_true == 1)[0], round(label_dict[1][1] * percent))
                ind0 = np.random.choice(
                    np.where(y_true == 0)[0], round(label_dict[1][0] * percent))

                numpix_class0 = round(label_dict[1][0] * percent)
                numpix_class1 = round(label_dict[1][1] * percent)
                numpix_class2 = round(label_dict[1][2] * percent)
                # numpix_class3 = round(label_dict[1][3] * percent)
                # print(numpix_class0)

                y_true = np.concatenate((y_true[ind0], y_true[ind1], y_true[ind2]))
                y_pred = np.concatenate((y_pred[ind0], y_pred[ind1], y_pred[ind2]))

                # change value 3 in prediction (burned area) to value 0 (other vegetation)
                y_pred[y_pred == 3] = 0

            # print('ground truth: ', np.unique(y_true))
            # print('predict: ', np.unique(y_pred))

            # print("TRUUUUE", y_true.min(), y_true.max())
            # print("PREDDD", y_pred.min(), y_pred.max())

            accuracy = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
            # try:
            #    accuracy = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
            # except:
            #    accuracy = np.nan
        
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred, sample_weight=None)
            # try:
            #    balanced_accuracy = balanced_accuracy_score(y_true, y_pred, sample_weight=None)
            # except:
            #    balanced_accuracy = np.nan
        
            print("Accuracy: ", accuracy, "Balanced Accuracy: ", balanced_accuracy)

            # print(classification_report(y_true, y_pred))
            # if len(label_name) < 4:
            #     target_names = ['other-vegetation','tree','cropland']
            # else:
            #     target_names = ['other-vegetation','tree','cropland','burned']

            target_names = ['other-vegetation', 'tree', 'cropland']
            report = classification_report(
                y_true, y_pred, target_names=target_names,
                output_dict=True, labels=label_name)
            cfn_matrix = confusion_matrix(y_true, y_pred)

            # tree_recall = report['tree']['recall']
            # crop_recall = report['cropland']['recall']

            # tree_precision = report['tree']['precision']
            # crop_precision = report['cropland']['precision']

            # get confusion matrix
            con_mat = tf.math.confusion_matrix(
                labels=y_true, predictions=y_pred, num_classes=nclasses
            ).numpy()

            # print('pri')

            # print(con_mat.sum(axis=1)[:, np.newaxis])
            # print(con_mat.sum(axis=1)[:, np.newaxis][0])
            # weights = [con_mat.sum(axis=1)[:, np.newaxis][0][0]/(5000*5000),con_mat.sum(axis=1)[:, np.newaxis][1][0]/(5000*5000),
            # con_mat.sum(axis=1)[:, np.newaxis][2][0]/(5000*5000),con_mat.sum(axis=1)[:, np.newaxis][3][0]/(5000*5000)]

            # print(weights)
            # get overall weighted accuracy
            # accuracy = accuracy_score(y_true, y_pred, normalize=False, sample_weight=weights)
            # print(con_mat)

            if norm:
                con_mat = np.around(
                    con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis],
                    decimals=2
                )
            # print('pra')

            # print(con_mat.sum(axis=1)[:, np.newaxis])
            where_are_NaNs = np.isnan(con_mat)
            con_mat[where_are_NaNs] = 0

            # print("done")

        # tree_recall, crop_recall, tree_precision, crop_precision, cfn_matrix
        return report, cfn_matrix, accuracy, balanced_accuracy
