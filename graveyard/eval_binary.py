import os
import sys
import glob
import itertools
import argparse
import omegaconf
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio as rio
import rioxarray as rxr
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from sklearn.metrics import accuracy_score, jaccard_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from tensorflow_caney.utils.data import modify_label_classes
from tensorflow_caney.config.cnn_config import Config


def confusion_matrix_func(y_true=[], y_pred=[], nclasses=4, norm=True):
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

    # get overall weighted accuracy
    accuracy = accuracy_score(
        y_true, y_pred, normalize=True, sample_weight=None)
    balanced_accuracy = balanced_accuracy_score(
        y_true, y_pred, sample_weight=None)

    print(classification_report(y_true, y_pred))

    # get confusion matrix
    con_mat = tf.math.confusion_matrix(
        labels=y_true, predictions=y_pred, num_classes=nclasses
    ).numpy()
    # print(con_mat.sum(axis=1)[:, np.newaxis])
    # print(con_mat.sum(axis=1)[:, np.newaxis][0])
    # weights = [
    # con_mat.sum(axis=1)[:, np.newaxis][0][0]/(5000*5000),
    # con_mat.sum(axis=1)[:, np.newaxis][1][0]/(5000*5000),
    # con_mat.sum(axis=1)[:, np.newaxis][2][0]/(5000*5000),
    # con_mat.sum(axis=1)[:, np.newaxis][3][0]/(5000*5000)]

    # print(weights)
    # get overall weighted accuracy
    # accuracy = accuracy_score(
    # y_true, y_pred, normalize=False, sample_weight=weights)
    # print(con_mat)

    if norm:
        con_mat = np.around(
            con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis],
            decimals=2
        )

    print(con_mat)

    # print(con_mat.sum(axis=1)[:, np.newaxis])
    where_are_NaNs = np.isnan(con_mat)
    con_mat[where_are_NaNs] = 0
    # return pd.DataFrame(
    # con_mat, index=range(nclasses), columns=range(nclasses))
    return con_mat, accuracy, balanced_accuracy


def plot_confusion_matrix(cm, label_name, class_names=['a', 'b', 'c']):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names: list with classes for confusion matrix
    Return: confusion matrix figure.
    """
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    # plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # Use white text if squares are dark; otherwise black.
    threshold = 0.55  # cm.max() / 2.
    # print(cm.shape[0], cm.shape[1]) #, threshold[0])

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{label_name[:-4]}.png')
    plt.show()
    plt.close()


def run(predicted_regex, labels_regex, conf):

    classes = ['Other', 'Croplands']  # 6-Cloud not present
    colors = ['yellow', 'blue']
    colormap = pltc.ListedColormap(colors)

    predicted_list = sorted(glob.glob(predicted_regex))
    labels_list = sorted(glob.glob(labels_regex))

    total_labels = len(labels_list)
    total_accuracy = 0

    for label_filename in labels_list:

        head_tail = os.path.split(label_filename)
        label_array = rxr.open_rasterio(label_filename).values[0, :, :]

        mask = [i for i in predicted_list if head_tail[1][:-27] in i][0]
        mask_array = rxr.open_rasterio(mask).values[0, :, :]

        label_array = modify_label_classes(
            label_array, conf.modify_labels, conf.substract_labels)

        label_array[label_array == 6] = 0
        mask_array[mask_array == 6] = 0
        mask_array[mask_array < 0] = 0

        accuracy = accuracy_score(
            label_array.flatten(), mask_array.flatten())

        total_accuracy += accuracy
        print(head_tail[1][:-27], accuracy)

        cnf_matrix, accuracy, balanced_accuracy = confusion_matrix_func(
            y_true=label_array, y_pred=mask_array,
            nclasses=len(classes), norm=True
        )
        print(cnf_matrix)
        print("Accuracy, Balanced Accuracy: ", accuracy, balanced_accuracy)

        file_name = os.path.basename(label_filename)
        plot_confusion_matrix(cnf_matrix, file_name, class_names=classes)

    print("Total Accuracy: ", total_accuracy / total_labels)

    return


# -----------------------------------------------------------------------------
# main
#
# python fix_labels.py options here
# -----------------------------------------------------------------------------
def main():

    # -------------------------------------------------------------------------
    # Process command-line args.
    # -------------------------------------------------------------------------
    desc = 'Use this application to preprocess label files.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--predicted-regex", type=str, required=True, dest='predicted_regex',
        help="Regex to load predicted image files")

    parser.add_argument(
        "--label-regex", type=str, required=True, dest='labels_regex',
        help="Regex to load label files")

    parser.add_argument(
        "--config-file", type=str, required=True, dest='config_file',
        help="Configuration file")

    args = parser.parse_args()

    # Configuration file intialization
    schema = omegaconf.OmegaConf.structured(Config)
    conf = omegaconf.OmegaConf.load(args.config_file)
    try:
        conf = omegaconf.OmegaConf.merge(schema, conf)
    except BaseException as err:
        sys.exit(f"ERROR: {err}")

    run(args.predicted_regex, args.labels_regex, conf)

    # -------------------------------------------------------------------------
    # Preprocessing of label files
    # -------------------------------------------------------------------------
    # Get list of data and label files
    # images_list = sorted(glob.glob(args.images_regex))
    # labels_list = sorted(glob.glob(args.labels_regex))
    # print(len(images_list))#, len(labels_list))

    # Setup output directory
    # os.makedirs(args.output_dir, exist_ok=True)

    # /Users/jacaraba/Desktop/results_mosaic_v2 - all images
    # /Users/jacaraba/Desktop/holidays_work/senegal/validation/*.gpkg

    """
    all_sets = []
    for image_filename in images_list:

        # open geopackage
        points_df = gpd.read_file(image_filename)
        #print(points_df['mask'], points_df['geometry'])

        # open expected filename
        label_filename = os.path.join(args.labels_regex, f'{Path(image_filename).stem}_clouds.tif')

        if not Path(label_filename).is_file():
            label_filename = os.path.join(args.labels_regex, f'{Path(image_filename).stem}.tif')
        
        #label_filename = os.path.join(args.labels_regex, f'{Path(image_filename).stem}.tif')
        rds = rxr.open_rasterio(label_filename)
        rds = rds[0, :, :]

        val_values = []
        for index in points_df.index:

            val = rds.sel(
                x=points_df['geometry'][index].x, y=points_df['geometry'][index].y, method="nearest"
            )
            #print(val.values, points_df['mask'][index])
            val_values.append(int(val.values))

        points_df['val'] = val_values

        all_sets.append(points_df)
    
    all_sets_concat = pd.concat(all_sets)
    #print(all_sets_concat)
    #print(all_sets_concat['val'].sum() / all_sets_concat['mask'].sum())

    accuracy = accuracy_score(all_sets_concat['mask'], all_sets_concat['val'])
    precision = precision_score(all_sets_concat['mask'], all_sets_concat['val'])
    recall = recall_score(all_sets_concat['mask'], all_sets_concat['val'])
    jaccard = jaccard_score(all_sets_concat['mask'], all_sets_concat['val'])
    confs = confusion_matrix(all_sets_concat['mask'], all_sets_concat['val'])

    print(all_sets_concat.index)
    print('cloud points:', all_sets_concat['mask'].value_counts())
    print(f'acc: {accuracy}')
    print(f'prec: {precision}')
    print(f'rec: {recall}')
    print(f'jacc: {jaccard}')
    print(confs)

    print("Total producer left: ", confs[0, 0], confs[1, 0], confs[0, 0] + confs[1, 0])
    print("Total producer right: ", confs[0, 1], confs[1, 1], confs[0, 1] + confs[1, 1])
    print("Total user up: ", confs[0, 0], confs[0, 1], confs[0, 0] + confs[0, 1])
    print("Total user down: ", confs[1, 0], confs[1, 1], confs[1, 0] + confs[1, 1])

    print("Producer accuracy not cloud", confs[0, 0] / (confs[0, 0] + confs[1, 0]))
    print("Producer accuracy cloud", confs[1, 1] / (confs[0, 1] + confs[1, 1]))

    print("User accuracy not cloud", confs[0, 0] / (confs[0, 0] + confs[0, 1]))
    print("User accuracy cloud", confs[1, 1] / (confs[1, 0] + confs[1, 1]))

    print("overall accuracy", (confs[0, 0] + confs[1, 1]) / (confs[0, 0] + confs[0, 1] + confs[1, 0] + confs[1, 1]))
    """
    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
