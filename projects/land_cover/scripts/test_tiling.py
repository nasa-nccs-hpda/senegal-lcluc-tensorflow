import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import rioxarray as rxr
import xarray as xr
import segmentation_models as sm
import matplotlib.colors as pltc
import matplotlib.patches as mpatches

sys.path.append('/adapt/nobackup/people/jacaraba/development/tensorflow-caney')

from tensorflow_caney.inference.tiler.tiler import Tiler
from tensorflow_caney.inference.tiler.merger import Merger

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

#filename = '/adapt/nobackup/people/mcarrol2/LCLUC_Senegal/ForKonrad/Tappan01_WV02_20110430_M1BS_103001000A27E100_data.tif'
filename = '/adapt/nobackup/people/mwooten3/Senegal_LCLUC/VHR/CAS/M1BS/WV02_20181217_M1BS_1030010089CC6D00-toa.tif'
model_filename = '/adapt/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/models/52-0.12.hdf5'
tile_size = 512
tile_channels = 8

# load model
model = tf.keras.models.load_model(
    model_filename, custom_objects={'iou_score': sm.metrics.iou_score}
)

image = rxr.open_rasterio(filename)
image = image.transpose("y", "x", "band")
temporary_tif = xr.where(image > -100, image, 600)
print(image.min(), temporary_tif.min())

tiler_image = Tiler(
    data_shape=temporary_tif.shape,
    tile_shape=(tile_size, tile_size, tile_channels),
    channel_dimension=2,
    overlap=0.5,
    mode='constant',
    constant_value=600
)

# Define the tiler and merger based on the output size of the prediction
tiler_mask = Tiler(
    data_shape=(temporary_tif.shape[0], temporary_tif.shape[1], 4),
    tile_shape=(tile_size, tile_size, 4),
    channel_dimension=2,
    overlap=0.5,
    mode='constant',
    constant_value=600
)

new_shape_image, padding_image = tiler_image.calculate_padding()
new_shape_mask, padding_mask = tiler_mask.calculate_padding()
print(image.shape, new_shape_image, new_shape_mask)

tiler_image.recalculate(data_shape=new_shape_image)
tiler_mask.recalculate(data_shape=new_shape_mask)

merger = Merger(tiler=tiler_mask, window='overlap-tile')
print("WEIGHTS SHAPE", merger.weights_sum.shape)

temporary_tif = temporary_tif.pad(
    y=padding_image[0], x=padding_image[1],
    constant_values=600)
print("After pad", temporary_tif.shape)

cc = ['gray', 'forestgreen', 'orange', 'red']
colors = pltc.ListedColormap(cc)
print(image.values.mean(axis=(0, 1)), image.values.std(axis=(0, 1)))

for tile_id, tile_original in tiler_image.iterate(temporary_tif):
    
    try:
        #print(tile_id)
        #tile = tile.values
        #tile[tile < 0] = 600
        #tile = tile / 10000.0 # 65535.0
        
        tile = tile_original.copy()
        tilea = tile_original.copy()
        print(tile.min().values, tile.max().values, tilea.min().values, tilea.max().values)
        
        for channel in range(tile.shape[-1]):
            #print(tile[:, :, channel].mean(), tile[:, :, channel].std())
            tile[:, :, channel] = (tile[:, :, channel] - tile[:, :, channel].mean()) / \
                (tile[:, :, channel].std())
        #print(tile.mean(), tile.std(), tile[:,:,0].mean(), tile[:,:,0].std(), tile[:,:,0].min(),tile[:,:,0].max())
        
        print(tile.min().values, tile.max().values, tilea.min().values, tilea.max().values)
        
        #prediction = model.predict(np.expand_dims(tile, 0), batch_size=1)
        
        #prediction = np.squeeze(np.argmax(prediction, axis=-1))
        #prediction = np.expand_dims(prediction, -1)
        #merger.add(tile_id, prediction)
        #print(np.unique(prediction))


        #plt.imshow(tilea[:, :, 6])
        #plt.show()

        #plt.imshow(prediction, cmap=colors)
        #plt.show()
    except IndexError:
        pass