import rasterio
#import cupy as cp
import numpy as np
import pandas as pd
import rioxarray as rxr
import geopandas as gpd
from glob import glob
from pathlib import Path

"""
# get epoch metadata
epoch_1_regex = '/explore/nobackup/people/mwooten3/Senegal_LCLUC/Compositing/Byte/v3/CAS.M1BS.*.*.*.2010.*.*.tif'
epoch_2_regex = '/explore/nobackup/people/mwooten3/Senegal_LCLUC/Compositing/Byte/v3/CAS.M1BS.*.*.*.2016.*.*.tif'

#epoch_filenames = glob(epoch_1_regex)
epoch_filenames = glob(epoch_2_regex)
print(len(epoch_filenames))

classes_count = {
    '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '10': 0}

for filename in epoch_filenames:

    raster = cp.asarray(rxr.open_rasterio(filename).values).squeeze()
    occurrences = cp.unique(raster, return_counts=True)
    for key, value in zip(occurrences[0], occurrences[1]):
        #print(type(key.item()), type(value.item()))
        #classes_count[str(key)] = classes_count[str(key)] + value.value
        classes_count[str(key.item())] += value.item()
print(classes_count)
"""

"""
# calculate class percentages

epoch_1_values = {
    'other': 3004476776, 'tree': 1975298010, 'tree_other': 750475800, 'crop': 942049571,
    'crop_other': 330977933, 'tree_crop': 96830217, 'tree_crop_other': 88849989, 'nodata': 425404089
}

epoch_2_values = {
    'other': 3489304691, 'tree': 1833412118, 'tree_other': 517485730, 'crop': 1471906653,
    'crop_other': 301242247, 'tree_crop': 41661046, 'tree_crop_other': 56409565, 'nodata': 520225116
}

epoch1_df = pd.DataFrame(epoch_1_values, index=['epoch1_count'])
epoch2_df = pd.DataFrame(epoch_2_values, index=['epoch2_count'])

epochs_df = pd.concat([epoch1_df, epoch2_df])
epochs_df = epochs_df.T
epochs_df = epochs_df.drop(['nodata'], axis=0)

print(epochs_df)

# get total number of pixels without nodata
epochs_df['epoch1_percentage'] = epochs_df.epoch1_count / epochs_df.epoch1_count.sum()
epochs_df['epoch2_percentage'] = epochs_df.epoch2_count / epochs_df.epoch2_count.sum()

epochs_df['epoch1_percentage_rounded'] = (epochs_df.epoch1_percentage * 100).round(2)
epochs_df['epoch2_percentage_rounded'] = (epochs_df.epoch2_percentage * 100).round(2)

epochs_df.to_csv('composite_class_percentages_v3.csv')
"""

"""
# extracting validation

epoch_1_regex = '/explore/nobackup/people/mwooten3/Senegal_LCLUC/Compositing/Byte/v3/CAS.M1BS.*.*.*.2010.*.*.tif'
epoch_2_regex = '/explore/nobackup/people/mwooten3/Senegal_LCLUC/Compositing/Byte/v3/CAS.M1BS.*.*.*.2016.*.*.tif'

epoch_filenames = glob(epoch_1_regex)
print("Filenames", len(epoch_filenames))

database_filename = '/explore/nobackup/projects/3sl/data/Validation/3sl-validation-database-20230209-filtered.gpkg'
gdf = gpd.read_file(database_filename)
gdf.year = gdf.year.astype(int)
print("Database", gdf.year.unique())

epoch_df = gdf[gdf['year'] >= 2016].reset_index()
print("Database After year filter", epoch_df.year.unique(), epoch_df.shape)
#print(epoch_df.crs)

epoch_df = epoch_df.to_crs("EPSG:32628")
print("Database Bounds", epoch_df.total_bounds)

epoch_df['composite'] = 10
epoch_df['composite_filename'] = 'filename'

for filename in epoch_filenames:

    src = rasterio.open(filename)
    extent = [src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3]]
    #print("Raster Extent", extent, src.crs)

    coord_list = [
        (x, y) for x, y in zip(epoch_df['geometry'].x , epoch_df['geometry'].y)]
    values_composite = [x.item() for x in src.sample(coord_list)]

    for index, row in epoch_df.iterrows():
        # epoch_df['composite'][0] = 
        # x = 0
        if values_composite[index] != 10:
            print(Path(filename).stem, values_composite[index], src.shape)
            epoch_df.at[index, 'composite'] = values_composite[index]
            epoch_df.at[index, 'composite_filename'] = Path(filename).stem

    #for x in src.sample(coord_list):
    #    print(x.item())
    #for _, row in epoch_df.iterrows():
    #    print(src.sample([(row['geometry'].x , row['geometry'].y)])[0].item())
    #    #for x in src.sample((epoch_df['geometry'].x , epoch_df['geometry'].y)):
    #    #    print(x.item())
    #    #[print(x.item() for x in src.sample(coord_list)]

#print(epoch_df.head(5))
#epoch_df = epoch_df.drop(['geometry'], axis=1)
#epoch_df = epoch_df.loc[:, epoch_df.nunique() != 1]
#epoch_df = epoch_df.replace(10, np.nan)
#print(epoch_df.head(5))

print(epoch_df.composite.unique())

epoch_df.to_file('epoch2_composite.gpkg', format='GPKG')

"""

#tappan_regex = '/explore/nobackup/projects/3sl/products/landcover/otcb_CAS_v3/Tappan/*.tif'
#tappan_regex = '/explore/nobackup/projects/3sl/products/landcover/otcb_CAS_v3/Tappan_Binary_Tree/*.tif'
#tappan_regex = '/explore/nobackup/projects/3sl/labels/landcover/2m/*/*.tif'
#tappan_regex = '/explore/nobackup/projects/3sl/development/cnn_landcover/normalization/otcb.v5/Tappan_Binary_Tree/*.tif'
#tappan_regex = '/explore/nobackup/projects/3sl/development/cnn_landcover/normalization/otcb.v5/*.tif'
#tappan_regex = '/explore/nobackup/projects/3sl/development/cnn_landcover/GMU_Experiments/land_cover_otcb_cas-wcas_global-std_50TS_4band-v2/models/predictions/ForKonrad/*.tif'
#tappan_regex = '/explore/nobackup/projects/3sl/development/cnn_landcover/GMU_Experiments/land_cover_otcb_cas-wcas_60TS-set1/data/model/predictions/ForKonrad/*.tif'
#tappan_regex = '/explore/nobackup/projects/3sl/development/cnn_landcover/GMU_Experiments/land_cover_otcb_cas-wcas_60TS-set1/data/model/predictions/ForKonrad/Tappan_Binary_Tree/*.tif'
#tappan_regex = '/explore/nobackup/projects/3sl/development/cnn_landcover/GMU_Experiments/land_cover_otcb_cas-wcas_global-std_50TS_4band-v2/models/predictions/ForKonrad/*.tif'
#tappan_regex = '/explore/nobackup/projects/3sl/development/cnn_landcover/GMU_Experiments/land_cover_otcb_cas-wcas_60TS-set1/data/model/predictions/ForKonrad/Tappan_Binary_Crop_Tree/*.tif'
#tappan_regex = '/explore/nobackup/projects/3sl/development/cnn_landcover/GMU_Experiments/land_cover_otcb_cas-wcas_60TS-set1/data/model/predictions/ForKonrad/Tappan_Binary_Crop_Tree_8bit_Ensemble/*.tif'

#tappan_regex = '/explore/nobackup/projects/3sl/development/cnn_landcover/accuracy-increase/landcover-otcb-senegal-quality-tiles-augmentation/results/ForKonrad/*.tif'
#tappan_regex = '/explore/nobackup/projects/3sl/development/cnn_landcover/accuracy-increase/landcover-otcb-senegal-quality-tiles-tversky-augmentation/results/ForKonrad/*.tif'

#tappan_regex = '/explore/nobackup/projects/3sl/development/cnn_landcover/accuracy-increase/quality-scale-unet/results/ForKonrad/*.tif'
#output_database = '/home/jacaraba/quality-scale-unet.gpkg'

tappan_regex = '/explore/nobackup/projects/3sl/development/cnn_landcover/accuracy-increase/quality-scale-attention/results/ForKonrad/*.tif'
output_database = '/home/jacaraba/quality-scale-attention.gpkg'

version = 'composite'#'training' #'composite'
epoch_filenames = glob(tappan_regex)
print("Filenames", len(epoch_filenames))

#database_filename = '/explore/nobackup/projects/3sl/data/Validation/3sl-validation-database-20230209-filtered.gpkg'
database_filename = '/explore/nobackup/projects/3sl/data/Validation/3sl-validation-database-20230412-all-three-agreed.gpkg'

gdf = gpd.read_file(database_filename)
gdf.year = gdf.year.astype(int)
print("Database", gdf.year.unique())

#epoch_df = gdf.to_crs("EPSG:32628")
epoch_df = gdf.to_crs(rasterio.open(epoch_filenames[0]).crs)
print("Database Bounds", epoch_df.total_bounds)

if version == 'training':
    epoch_df['Tappan_Training_Data'] = 10
    epoch_df['Tappan_Training_Data_filename'] = 'filename'
else:
    epoch_df['composite'] = 10
    epoch_df['composite_filename'] = 'filename'

for filename in epoch_filenames:

    print(Path(filename).stem)

    try:

        src = rasterio.open(filename)
        epoch_df = epoch_df.to_crs(src.crs)
        #src = src.reproject("EPSG:32628")
        #print(src.crs)
        extent = [src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3]]

        coord_list = [
            (x, y) for x, y in zip(epoch_df['geometry'].x , epoch_df['geometry'].y)]
        values_composite = [x.item() for x in src.sample(coord_list)]

        for index, row in epoch_df.iterrows():
            if version == 'training':
                training_shortname = '_'.join((Path(Path(filename).stem).stem).split('_')[:5])
                val_shortname = '_'.join(row['short_filename'].split('_')[:5])
                if values_composite[index] < 5 and training_shortname == val_shortname:
                    print(Path(filename).stem, values_composite[index], src.shape)
                    epoch_df.at[index, 'Tappan_Training_Data'] = values_composite[index]
                    epoch_df.at[index, 'Tappan_Training_Data_filename'] = Path(filename).stem
            else:
                if values_composite[index] < 5 and Path(Path(filename).stem).stem == row['short_filename']:
                    print(Path(filename).stem, values_composite[index], src.shape)
                    epoch_df.at[index, 'composite'] = values_composite[index]
                    epoch_df.at[index, 'composite_filename'] = Path(filename).stem

    except:
        continue

#print(epoch_df.composite.unique())
epoch_df.to_file(output_database, format='GPKG')

print(epoch_df.shape)