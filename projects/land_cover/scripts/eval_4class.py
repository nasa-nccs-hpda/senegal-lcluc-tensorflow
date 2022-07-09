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

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    plt.savefig('/home/geoint/tri/nasa_senegal/confusion_matrix/{}_cfn_matrix_4class.png'.format(label_name[:-4]))
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    classes = ['Other', 'Tree/shrub', 'Croplands', 'Burned Area']  # 6-Cloud not present
    colors = ['brown', 'forestgreen', 'orange', 'grey']
    colormap = pltc.ListedColormap(colors)

    #labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/labels/*.tif'))
    #predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/48-0.16/images/*.tif'))
    #predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/new_pred/63-0.13/images/*.tif'))

    #labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/newCAS_label/*.tif'))
    #predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/52-0.12-allCAS/images/newCAS/*.tif'))

    # predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/48-0.16/images/t2/*.tif'))
    # labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/label_t2/*.tif'))

    # labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/label_TS6_7/*.tif'))
    # predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/48-0.16/images/newCAS/*.tif'))

    # labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/labels_new/new/*.tif'))
    # predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/48-0.16-allCAS/images/new/*.tif'))

    # labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/labels_new/*.tif'))
    # predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/48-0.16/images/*.tif'))

    labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/new_masks/*.tif'))
    predictions = sorted(glob.glob('/home/geoint/tri/nasa_senegal/predictions/48-0.16-allCAS/images/new/*.tif'))

    for lf in labels:

    #for lf, pf in zip(labels, predictions):

        print(os.path.basename(lf))
        file_name = os.path.basename(lf)

        pf = lf[:30]+'predictions/37-0.20-all/images/'+lf[-71:-26]+'data.landcover.tif'

        print(pf)

        # open filenames
        label = np.squeeze(xr.open_rasterio(lf).values)
        prediction = np.squeeze(xr.open_rasterio(pf).values)


        label[label == 7] = 3  # merge no-data area to shadow/water
        label[label == 4] = 3
        label[label == 3] = 0
        label[label == 5] = 3

        # some preprocessing
        prediction[prediction == -10001] = 0

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

