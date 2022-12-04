import os
import time
import logging
import fiona
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import shapely.speedups
from IPython.display import display
from omegaconf.listconfig import ListConfig
from multiprocessing import Pool, cpu_count

# import cuspatial
import osgeo.gdal
import tensorflow as tf
from pathlib import Path
from glob import glob
from shapely.geometry import box
from skimage.transform import resize
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import mean_squared_error

regex = '/explore/nobackup/projects/3sl/development/cnn_landcover/crop.srv.v1/metadata/*.gpkg'
data_output_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/crop.srv.v1'
tile_size = 256
filenames = sorted(glob(regex), reverse=True)
print(len(filenames), filenames)

import struct, numpy, pylab
import matplotlib.pyplot as plt

def extractWindow(imageDataset, pixelX, pixelY, pixelWidth, pixelHeight):
    # Extract raw data
    if type(imageDataset) is np.ndarray:
        matrix = imageDataset[int(pixelX):int(pixelX+pixelWidth), int(pixelY):int(pixelY+pixelHeight)]
    else:
        matrix = imageDataset.ReadAsArray(pixelX, pixelY, pixelWidth, pixelHeight)
    #print(pixelX, pixelY, pixelWidth, pixelHeight)
    #plt.imshow(matrix[7, :, :] / 10000.0)
    #plt.show()
    #pylab.show()
    # Return
    return matrix

def extractCenteredWindow(imageDataset, pixelX, pixelY, pixelWidth, pixelHeight):
    centeredPixelX = pixelX - pixelWidth / 2
    centeredPixelY = pixelY - pixelHeight / 2
    return extractWindow(imageDataset, centeredPixelX, centeredPixelY, pixelWidth, pixelHeight)

def convertGeoLocationToPixelLocation(geoLocation):
    xGeo, yGeo = geoLocation[0], geoLocation[1]
    if g2 == 0:
        xPixel = (xGeo - g0) / float(g1)
        yPixel = (yGeo - g3 - xPixel*g4) / float(g5)
    else:
        xPixel = (yGeo*g2 - xGeo*g5 + g0*g5 - g2*g3) / float(g2*g4 - g1*g5)
        yPixel = (xGeo - g0 - xPixel*g1) / float(g2)
    return int(round(xPixel)), int(round(yPixel))

def convertGeoDimensionsToPixelDimensions(geoWidth, geoHeight, g1, g5):
    return int(round(abs(float(geoWidth) / g1))), int(round(abs(float(geoHeight) / g5)))

import cv2
from skimage.draw import polygon

data_output_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/crop.srv.v1'

images_output_dir = os.path.join(data_output_dir, 'images')
labels_output_dir = os.path.join(data_output_dir, 'labels')

os.makedirs(images_output_dir, exist_ok=True)
os.makedirs(labels_output_dir, exist_ok=True)

for filename in [filenames[2]]:
    
    # read gpkg
    dataset_gdf = gpd.read_file(filename)
    print(dataset_gdf.shape)
    #display(dataset_gdf)
    
    for index, row in dataset_gdf.iterrows():
        
        try:
            print(row["scene_id"])
            output_data_filename = os.path.join(images_output_dir, f'{Path(row["scene_id"]).stem}_{str(index+1)}.npy')
            output_label_filename = os.path.join(labels_output_dir, f'{Path(row["scene_id"]).stem}_{str(index+1)}.npy')

            #------------------------------------------------------------------------------
            # here we extract the data tile
            #------------------------------------------------------------------------------
            imageDataset = osgeo.gdal.Open(row['scene_id'])
            polygon_centroid = row['geometry'].centroid
            #print("centroid", polygon_centroid.x, polygon_centroid.y, polygon_centroid)

            g0, g1, g2, g3, g4, g5 = imageDataset.GetGeoTransform()
            #print(g0, g1, g2, g3, g4, g5, imageDataset.RasterXSize,imageDataset.RasterYSize)

            windowPixelX, windowPixelY = convertGeoLocationToPixelLocation(
                [row['geometry'].centroid.x, row['geometry'].centroid.y])
            #print(windowPixelX, windowPixelY)

            windowPixelWidth, windowPixelHeight = tile_size, tile_size
            data_matrix = extractCenteredWindow(
                imageDataset, windowPixelX, windowPixelY, windowPixelWidth, windowPixelHeight)

            # if nodata is present, skip
            if data_matrix.min() < 0:
                continue

            #------------------------------------------------------------------------------
            # here we extract the label tile
            #------------------------------------------------------------------------------
            label_mask = numpy.full((imageDataset.RasterXSize,imageDataset.RasterYSize), False)
            #np.ones(shape=(imageDataset.RasterXSize,imageDataset.RasterYSize), dtype="bool")        
            polygon_pixel_coords = np.apply_along_axis(
                convertGeoLocationToPixelLocation, axis=1, arr=np.array(list(row['geometry'].exterior.coords)))
            rr, cc = polygon(
                polygon_pixel_coords[:,0], polygon_pixel_coords[:,1],
                (imageDataset.RasterXSize,imageDataset.RasterYSize)
            )
            label_mask[cc, rr] = True

            # there is a strange represenation of where X and Y are location within the polygon
            # for now just invert the axis and call it a day
            label_matrix = extractCenteredWindow(
                label_mask.astype(int), windowPixelY, windowPixelX, windowPixelWidth, windowPixelHeight)

            #print("occurrence", np.count_nonzero(label_matrix == 1), np.unique(label_matrix))

            if np.count_nonzero(label_matrix == 1) < 15000: #20000:  # 512x512 / 2
                continue

            # TODO: CONDITION THAT TIMES MOST BE 256x256, not smaller not greater
            # TODO: swap axis to make it channels last
            # TODO: be able to select which bands we want for training
            # TODO: expand label axis for binary classification

            np.save(output_data_filename, data_matrix)
            np.save(output_label_filename, label_matrix)

            #plt.figure(figsize = (6,6))
            #plt.imshow(np.moveaxis((data_matrix[6, :, :] / 10000.0), 0, -1))
            #plt.imshow(label_matrix, alpha=0.5)
            #plt.show()
        
        except (AttributeError, IndexError):
            continue
