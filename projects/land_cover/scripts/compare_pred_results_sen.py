from osgeo import gdal, gdal_array
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import pickle

def read_mask(file_path):
    naip_fn = file_path
    driverTiff = gdal.GetDriverByName('GTiff')
    naip_ds = gdal.Open(naip_fn, 1)
    nbands = naip_ds.RasterCount
    # create an empty array, each column of the empty array will hold one band of data from the image
    # loop through each band in the image nad add to the data array
    data = np.empty((naip_ds.RasterXSize * naip_ds.RasterYSize, nbands))
    for i in range(1, nbands + 1):
        band = naip_ds.GetRasterBand(i).ReadAsArray()
        data[:, i - 1] = band.flatten()

    img_data = np.zeros((naip_ds.RasterYSize, naip_ds.RasterXSize, naip_ds.RasterCount),
                        gdal_array.GDALTypeCodeToNumericTypeCode(naip_ds.GetRasterBand(1).DataType))
    for b in range(img_data.shape[2]):
        img_data[:, :, b] = naip_ds.GetRasterBand(b + 1).ReadAsArray()
    mask = img_data

    return mask

def standardize_image(
            image,
            standardization_type: str,
            mean: list = None,
            std: list = None
        ):
    """
    Standardize image within parameter, simple scaling of values.
    Loca, Global, and Mixed options.
    """
    if standardization_type == 'local':
        for i in range(image.shape[-1]):  # for each channel in the image
            image[:, :, i] = (image[:, :, i] - np.mean(image[:, :, i])) / \
                (np.std(image[:, :, i]) + 1e-8)
    elif standardization_type == 'global':
        for i in range(image.shape[-1]):  # for each channel in the image
            image[:, :, i] = (image[:, :, i] - mean[i]) / (std[i] + 1e-8)
    elif standardization_type == 'mixed':
        raise NotImplementedError
        #    if np.random.random_sample() > 0.75:
        #        for i in range(x.shape[-1]):  # for each channel in the image
        #            x[:, :, i] = (x[:, :, i] - self.conf.mean[i]) / \
        #                (self.conf.std[i] + 1e-8)
        #    else:
        #        for i in range(x.shape[-1]):  # for each channel in the image
        #            x[:, :, i] = (x[:, :, i] - np.mean(x[:, :, i])) / \
        #                (np.std(x[:, :, i]) + 1e-8)
    return image


def get_class_pixels(mask, image):
    labels = [1, 2, 3, 4, 5, 7]
    ## create class dictionary
    band_pixels = {}
    for label in labels:
        band_pixels[label] = {}
        for band in range(image.shape[2]):
            band_pixels[label][band] = []
            itemindex = np.where(mask == label)
            height_ind = itemindex[0]
            width_ind = itemindex[1]

            for index in range(len(height_ind)):
                pixel_value = image[height_ind[index], width_ind[index], band]
                band_pixels[label][band].append(pixel_value)
        band_pixels[label][band] = np.array(band_pixels[label][band])

    return band_pixels

def get_mean_std_metadata(filename):
    """
    Load mean and std from disk.
    Args:
        filename (str): csv filename path to load mean and std from
    Returns:
        np.array mean, np.array std
    """
    assert os.path.isfile(filename), \
        f'{filename} does not exist.'
    metadata = pd.read_csv(filename, header=None)
    # logging.info('Loading mean and std values.')
    return metadata.loc[0].values, metadata.loc[1].values


def get_class_histogram(dict_class_1, dict_class_2={}, dict_class_3={}, name=''):
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(10,10))
    fig_name = '/home/geoint/tri/nasa_senegal/compare_label/{}_.png'.format(name)
    # for i in dict_class_1.keys():
    #     title=str("Band ")+str(i+1)
    #     axes[i].title.set_text(title)
    #     # axes[i].hist(dict_class_1[i], bins=50, range=[0,4000], alpha=0.5, color='red')
    #     # axes[i].hist(dict_class_2[i], bins=50, range=[0,4000], alpha=0.5, color='blue')
    #     # axes[i].hist(dict_class_3[i], bins=50, range=[0,4000], alpha=0.5, color='green')
    #     axes[i].hist(dict_class_1[i], bins=50, range=[0, 4000], alpha=0.5, label='tri', color='red')
    #     axes[i].hist(dict_class_2[i], bins=50, range=[0, 4000], alpha=0.5, label='ulas', color='blue')
    #     axes[i].hist(dict_class_3[i], bins=50, range=[0, 4000], alpha=0.5, label='mark', color='green')
    #     axes[i].legend(loc="upper right")

    class_id = 2
    for i in dict_class_1[class_id].keys():
        axes[i].hist(dict_class_1[class_id][i], bins=50, alpha=0.5, label='20141119_0.90_acc', color='red')
        axes[i].hist(dict_class_2[class_id][i], bins=50, alpha=0.5, label='20130530_0.47_acc', color='blue')
        axes[i].legend(loc="upper right")

    fig.savefig(fig_name)
    plt.show()


if __name__ == '__main__':

    ## get all file path
    # mask_file_tri = '/home/geoint/tri/nasa_senegal/compare_label/reclassified/Tappan02_WV02_20120218_M1BS_103001001077BE00_mask.tif'
    # mask_file_ulas = '/home/geoint/tri/nasa_senegal/compare_label/ulas_reclassified/Tappan02_WV02_20120218_M1BS_103001001077BE00_mask.tif'
    # mask_file_mark = '/home/geoint/tri/nasa_senegal/compare_label/mark_reclassified/Tappan02_WV02_20120218_M1BS_103001001077BE00_mask.tif'
    #
    # multi_spec_file = '/home/geoint/tri/nasa_senegal/cassemance/Tappan02_WV02_20120218_M1BS_103001001077BE00_data.tif'
    #
    # mask_tri = read_mask(mask_file_tri)
    # mask_ulas = read_mask(mask_file_ulas)
    # mask_mark = read_mask(mask_file_mark)
    #
    # multi_spec_im = read_mask(multi_spec_file)
    #
    # band_pixels_tri = get_class_pixels(mask_tri, multi_spec_im)
    # band_pixels_ulas = get_class_pixels(mask_ulas, multi_spec_im)
    # band_pixels_mark = get_class_pixels(mask_mark, multi_spec_im)

    # maskfile_TS18_20160617 = '/home/geoint/tri/nasa_senegal/new_masks/Tappan18_WV03_20160617_M1BS_104001001E3B1600_mask_segs_reclassified.tif'
    # multispec_file_TS18_20160617 = '/home/geoint/tri/nasa_senegal/newCAS/Tappan18_WV03_20160617_M1BS_104001001E3B1600_data.tif'
    # maskfile_TS18_20160307 = '/home/geoint/tri/nasa_senegal/new_masks/Tappan18_WV03_20160307_M1BS_10400100196AFB00_mask_segs_reclassified.tif'
    # multispec_file_TS18_20160307 = '/home/geoint/tri/nasa_senegal/newCAS/Tappan18_WV03_20160307_M1BS_10400100196AFB00_data.tif'

    maskfile_TS18_20141119 = '/home/geoint/tri/nasa_senegal/new_masks/Tappan18_WV03_20141119_M1BS_1040010004CF8900_mask_segs_reclassified.tif'
    multispec_file_TS18_20141119 = '/home/geoint/tri/nasa_senegal/newCAS/Tappan18_WV03_20141119_M1BS_1040010004CF8900_data.tif'
    maskfile_TS18_20130530 = '/home/geoint/tri/nasa_senegal/new_masks/Tappan18_WV02_20130530_M1BS_1030010022925500_mask_segs_reclassified.tif'
    multispec_file_TS18_20130530 = '/home/geoint/tri/nasa_senegal/newCAS/Tappan18_WV02_20130530_M1BS_1030010022925500_data.tif'

    metadata = '/home/geoint/tri/nasa_senegal/stat_file/mean-std-landcover-trees.csv'

    mean, std = get_mean_std_metadata(metadata)

    print(mean[-1])

    print(std[0])

    ######################
    # mask_20160617 = read_mask(maskfile_TS18_20160617)
    #
    # mask_20160307 = read_mask(maskfile_TS18_20160307)
    #
    # multispec_TS18_20160617 = read_mask(multispec_file_TS18_20160617)
    #
    # multispec_TS18_20160307 = read_mask(multispec_file_TS18_20160307)
    #
    # std_TS18_20160617 = standardize_image(
    #         multispec_TS18_20160617/10000, standardization_type='local', mean=None, std=None)
    #
    # std_TS18_20160307 = standardize_image(
    #         multispec_TS18_20160307/10000, standardization_type='local', mean=None, std=None)
    #
    # band_pixels_TS18_20160617 = get_class_pixels(mask_20160617, multispec_TS18_20160617)
    # band_pixels_TS18_20160307 = get_class_pixels(mask_20160307, multispec_TS18_20160307)
    #
    # band_pixels_std_TS18_20160617 = get_class_pixels(mask_20160617, std_TS18_20160617)
    # band_pixels_std_TS18_20160307 = get_class_pixels(mask_20160307, std_TS18_20160307)
    #
    # get_class_histogram(band_pixels_TS18_20160617, band_pixels_TS18_20160307, name="croplands_toa")
    # get_class_histogram(band_pixels_std_TS18_20160617, band_pixels_std_TS18_20160307, name="croplands_std")

    #########################
    mask_20141119 = read_mask(maskfile_TS18_20141119)

    mask_20130530 = read_mask(maskfile_TS18_20130530)

    multispec_TS18_20141119 = read_mask(multispec_file_TS18_20141119)

    multispec_TS18_20130530 = read_mask(multispec_file_TS18_20130530)

    std_TS18_20141119 = standardize_image(
        multispec_TS18_20141119, standardization_type='global', mean=mean, std=std)

    std_TS18_20130530 = standardize_image(
        multispec_TS18_20130530, standardization_type='global', mean=mean, std=std)

    # std_TS18_20141119 = standardize_image(
    #     multispec_TS18_20141119/10000, standardization_type='local', mean=mean, std=std)
    #
    # std_TS18_20130530 = standardize_image(
    #     multispec_TS18_20130530/10000, standardization_type='local', mean=mean, std=std)

    # print(np.min(std_TS18_20141119, axis=(0,1)))
    # print(np.max(std_TS18_20141119, axis=(0,1)))
    # #print(np.mean(std_TS18_20130530, axis=(0,1)))
    #
    print(np.unique(std_TS18_20141119))


    band_pixels_TS18_20141119 = get_class_pixels(mask_20141119, multispec_TS18_20141119)
    band_pixels_TS18_20130530 = get_class_pixels(mask_20130530, multispec_TS18_20130530)

    band_pixels_std_TS18_20141119 = get_class_pixels(mask_20141119, std_TS18_20141119)
    band_pixels_std_TS18_20130530 = get_class_pixels(mask_20130530, std_TS18_20130530)

    get_class_histogram(band_pixels_TS18_20141119, band_pixels_TS18_20130530, name="1_croplands_toa")
    get_class_histogram(band_pixels_std_TS18_20141119, band_pixels_std_TS18_20130530, name="1_croplands_std")


