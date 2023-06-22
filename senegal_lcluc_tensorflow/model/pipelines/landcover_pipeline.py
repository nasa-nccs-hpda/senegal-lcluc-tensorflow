import os
import sys
import time
import logging
import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from rioxarray.merge import merge_arrays

from tqdm import tqdm
from glob import glob
from pathlib import Path
from itertools import repeat
from omegaconf import OmegaConf
from multiprocessing import Pool, cpu_count

from tensorflow_caney.model.config.cnn_config import Config
from tensorflow_caney.utils.system import seed_everything
from tensorflow_caney.model.pipelines.cnn_regression import CNNRegression
from tensorflow_caney.utils.data import gen_random_tiles, \
    get_dataset_filenames, get_mean_std_dataset
from tensorflow_caney.model.dataloaders.regression import RegressionDataLoader
# from vhr_cnn_chm.model.atl08 import ATL08
# from tensorflow_caney.utils.vector.extract import \
#    convert_coords_to_pixel_location, extract_centered_window
# from tensorflow_caney.utils.data import modify_bands, \
#    get_dataset_filenames, get_mean_std_dataset, get_mean_std_metadata
# from tensorflow_caney.utils.system import seed_everything
# from tensorflow_caney.model.pipelines.cnn_regression import CNNRegression
# from tensorflow_caney.model.dataloaders.regression import RegressionDataLoader
# from tensorflow_caney.utils import indices
# from tensorflow_caney.utils.model import load_model
# from tensorflow_caney.inference import regression_inference
# from pygeotools.lib import iolib, warplib

# osgeo.gdal.UseExceptions()

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

    def test(self):
        """
        import os
        import glob
        import warnings
        import itertools
        import xarray as xr
        import numpy as np
        import matplotlib.pyplot as plt
        import tensorflow as tf
        import matplotlib.colors as pltc
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.metrics import classification_report
        import csv
        import re
        ​
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ​
        def confusion_matrix_func(y_true=[], y_pred=[], nclasses=4, norm=True):
            
            Args:
                y_true:   2D numpy array with ground truth
                y_pred:   2D numpy array with predictions (already processed)
                nclasses: number of classes
            Returns:
                numpy array with confusion matrix
            
        ​
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
        ​
            label_name = np.unique(y_pred)
        ​
            # get overall weighted accuracy
            accuracy = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred, sample_weight=None)
        ​
            # print(classification_report(y_true, y_pred))
            if len(label_name) != 4:
                target_names = ['other-vegetation','tree','cropland']
            else:
                target_names = ['other-vegetation','tree','cropland','burned']
            report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        ​
            tree_recall = report['tree']['recall']
            crop_recall = report['cropland']['recall']
        ​
            tree_precision = report['tree']['precision']
            crop_precision = report['cropland']['precision']
        ​
            ## get confusion matrix
            con_mat = tf.math.confusion_matrix(
                labels=y_true, predictions=y_pred, num_classes=nclasses
            ).numpy()
        ​
            # print(con_mat.sum(axis=1)[:, np.newaxis])
            # print(con_mat.sum(axis=1)[:, np.newaxis][0])
            # weights = [con_mat.sum(axis=1)[:, np.newaxis][0][0]/(5000*5000),con_mat.sum(axis=1)[:, np.newaxis][1][0]/(5000*5000),
            # con_mat.sum(axis=1)[:, np.newaxis][2][0]/(5000*5000),con_mat.sum(axis=1)[:, np.newaxis][3][0]/(5000*5000)]
        ​
            # print(weights)
            # get overall weighted accuracy
            # accuracy = accuracy_score(y_true, y_pred, normalize=False, sample_weight=weights)
            # print(con_mat)
        ​
            if norm:
                con_mat = np.around(
                    con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis],
                    decimals=2
                )
        ​
            # print(con_mat)
        ​
            # print(con_mat.sum(axis=1)[:, np.newaxis])
            where_are_NaNs = np.isnan(con_mat)
            con_mat[where_are_NaNs] = 0
            return con_mat, accuracy, balanced_accuracy, tree_recall, crop_recall, tree_precision, crop_precision
        ​
        ​
        def plot_confusion_matrix(cm, label_name, model, class_names=['a', 'b', 'c']):
            
            Returns a matplotlib figure containing the plotted confusion matrix.
            Args:
                cm (array, shape = [n, n]): a confusion matrix of integer classes
                class_names: list with classes for confusion matrix
            Return: confusion matrix figure.
            
            figure = plt.figure(figsize=(8, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            # plt.colorbar()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            # Use white text if squares are dark; otherwise black.
            threshold = 0.55  # cm.max() / 2.
            # print(cm.shape[0], cm.shape[1]) #, threshold[0])
        ​
            print(label_name[:-27])
        ​
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(f'/home/geoint/tri/nasa_senegal/confusion_matrix/cas-tl-wcas/{model}-{label_name[:-27]}_cfn_matrix_4class.png')
            # plt.show()
            plt.close()
        ​
        ​
        # Press the green button in the gutter to run the script.
        if __name__ == '__main__':
        ​
            classes = ['Other', 'Tree/shrub', 'Croplands', 'Burned Area']  # 6-Cloud not present
            colors = ['brown', 'forestgreen', 'orange', 'grey']
            colormap = pltc.ListedColormap(colors)
        ​
            labels = sorted(glob.glob('/home/geoint/tri/allwcasmasks/*.tif'))
        ​
            out_dir = '/home/geoint/tri/nasa_senegal/pred_eval/increase_wCAS_set1/'
        ​
            with open(f'{out_dir}66-0.12-cas15-tl-wcas-8ts-set2-0525_stat_results.csv', 'w') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                writer.writerow(['tappan', 'overall-accuracy', 'accuracy', 'tree-recall', 'crop-recall', 'tree-precision', 'crop-precision'])
        ​
                for lf in labels:
        ​
                    print(os.path.basename(lf))
                    file_name = os.path.basename(lf)
        ​
                    name = file_name[:-9]
        ​
                    search_term_lf = re.search(r'/allwcasmasks/(.*?)_mask.tif', lf).group(1)
                    # search_term_lf = re.search(r'/CAS_West_masks/(.*?)_mask', lf).group(1)
                    print(search_term_lf)
        ​
                    pf = '/home/geoint/tri/nasa_senegal/predictions/66-0.12-cas15-tl-wcas-8ts-set2-0525/images/' + search_term_lf + '_data.landcover.tif'
        ​
                    search_term_pf = re.search(r'predictions.(.*?)/images', pf).group(1)
                    print(search_term_pf)
        ​
                    # open filenames
                    label = np.squeeze(xr.open_rasterio(lf).values)
                    prediction = np.squeeze(xr.open_rasterio(pf).values)
        ​
                    ## group label pixel to 4 classes
                    # label[label == 5] = 3  # merge burned area to other vegetation
                    label[label == 7] = 3  # merge no-data area to shadow/water
                    label[label == 4] = 3
                    label[label == 3] = 0
                    label[label == 5] = 3
        ​
                    # some preprocessing
                    prediction[prediction == -10001] = 0
                    prediction[prediction == 255] = 0
                    # prediction[prediction == 3] = 0
        ​
                    print("Unique pixel values of prediction: ", np.unique(prediction))
                    print("Unique pixel values of label: ", np.unique(label))
        ​
                    cnf_matrix, accuracy, balanced_accuracy, tree_recall, crop_recall, tree_precision, crop_precision = confusion_matrix_func(y_true=label, y_pred=prediction, nclasses=len(classes), norm=True)
        ​
                    writer.writerow([name, accuracy, balanced_accuracy, tree_recall, crop_recall, tree_precision, crop_precision])
        
                    print("Overall Accuracy: ", accuracy)
                    print("Balanced Accuracy: ", balanced_accuracy)
    
                    plot_confusion_matrix(cnf_matrix, file_name, search_term_pf, class_names=classes)
        """
        return

    def validate(self):
        return
