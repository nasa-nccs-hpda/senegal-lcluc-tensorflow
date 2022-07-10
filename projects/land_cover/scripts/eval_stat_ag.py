# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=RuntimeWarning)

# classes = ['Tree/shrub', 'Croplands', 'Other', 'Shadow/Water', 'Burned area', 'Cloud', 'No-data'] # 6-Cloud not present
#classes = ['Tree/shrub', 'Croplands', 'Other']  # 6-Cloud not present
# colors = ['forestgreen', 'orange', 'brown', 'blue', 'red', 'white', 'black']
#colors = ['forestgreen', 'orange', 'brown']
#colormap = pltc.ListedColormap(colors)
# labels = sorted(glob.glob('label_sar/*.tif'))
#labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/label_v2/*.tif'))
# predictions = sorted(glob.glob('prediction/trainT0220120218only/*.tif'))
#predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/100_unet_senegal_Adam_128_batch_all_t1_new_v2_4000_15.h5/images/*.tif'))
# predictions = sorted(glob.glob('prediction/drywet_sar_2T02_2012_v2_4000/*.tif'))

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
    accuracy = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred, sample_weight=None)

    print(classification_report(y_true, y_pred))

    ## get confusion matrix
    con_mat = tf.math.confusion_matrix(
        labels=y_true, predictions=y_pred, num_classes=nclasses
    ).numpy()
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

    print(con_mat)

    # print(con_mat.sum(axis=1)[:, np.newaxis])
    where_are_NaNs = np.isnan(con_mat)
    con_mat[where_are_NaNs] = 0
    # return pd.DataFrame(con_mat, index=range(nclasses), columns=range(nclasses))
    return con_mat, accuracy, balanced_accuracy


def plot_confusion_matrix(cm, label_name, class_names=['a', 'b', 'c']):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names: list with classes for confusion matrix
    Return: confusion matrix figure.
    """
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

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('/home/geoint/tri/nasa_senegal/confusion_matrix/{}_chm_int_cfn_matrix.png'.format(label_name[:-4]))
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    classes = ['Other', 'Croplands']  # 6-Cloud not present
    colors = ['yellow', 'blue']
    colormap = pltc.ListedColormap(colors)

    # labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/labels/*.tif'))
    # predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/100_unet_senegal_Adam_128_batch_all_t1_agonly_v2_4000_06.h5/images/*.tif'))

    # labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/labels/*.tif'))
    # predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/66-0.02/images/*.tif'))

    # labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/labels_new/new/*.tif'))
    # predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/46-0.03-allCAS/images/new/*.tif'))

    # labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/labels_new/*.tif'))
    # predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/agonly/46-0.03/images/*.tif'))

    labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/new_masks/*.tif'))
    # predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/48-0.16-allCAS/images/new/*.tif'))

    for lf in labels:
        # for lf, pf in zip(labels, predictions):

        print(os.path.basename(lf))
        file_name = os.path.basename(lf)

        pf = lf[:30] + 'predictions/72-0.01-all/images/' + lf[-71:-26] + 'data.landcover.tif'
        print(os.path.basename(lf))
        file_name = os.path.basename(lf)

        # open filenames
        label = np.squeeze(xr.open_rasterio(lf).values)
        prediction = np.squeeze(xr.open_rasterio(pf).values)

        label[label == 5] = 1  # merge burned area to other vegetation
        label[label == 7] = 1  # merge no-data area to shadow/water
        label[label == 4] = 1
        label[label == 3] = 1
        label = label - 1

        # some preprocessing
        # prediction[prediction == -9999] = 7

        # prediction[prediction == -10001] = 6
        # prediction[prediction == 6] = 3
        # prediction = prediction - 1
        prediction[prediction == -10001] = 0

        prediction[prediction == 6] = 0
        prediction[prediction == 3] = 0

        print("Unique pixel values of prediction: ", np.unique(prediction))
        print("Unique pixel values of label: ", np.unique(label))

        #     # Compute confusion matrix

        cnf_matrix, accuracy, balanced_accuracy = confusion_matrix_func(
            y_true=label, y_pred=prediction, nclasses=len(classes), norm=True
        )

        # let's plot some information here
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
        axes[0].title.set_text("Ground Truth")
        axes[0].imshow(label, cmap=colormap)
        axes[1].title.set_text("Prediction")
        axes[1].imshow(prediction, cmap=colormap)
        fig.tight_layout()
        plt.show()

        print("Overall Accuracy: ", accuracy)
        print("Balanced Accuracy: ", balanced_accuracy)
        # print("Mean Intersection over Union: ", mean_iou)
        plot_confusion_matrix(cnf_matrix, file_name, class_names=classes)

