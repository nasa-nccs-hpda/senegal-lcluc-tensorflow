import os
import re
import random
import pandas as pd
import geopandas as gpd
import cupy as cp
import numpy as np
import rasterio as rio
import rioxarray as rxr
import matplotlib.pyplot as plt
from itertools import repeat
from multiprocessing import Pool, cpu_count

from glob import glob
from pathlib import Path
from shapely.geometry import box
from IPython.display import display

"""
cas_regex = '/explore/nobackup/projects/3sl/data/VHR/CAS/M1BS/*-toa.tif'
etz_regex = '/explore/nobackup/projects/3sl/data/VHR/ETZ/M1BS/*-toa.tif'
srv_regex = '/explore/nobackup/projects/3sl/data/VHR/SRV/M1BS/*-toa.tif'

tappan_cas_regex = '/explore/nobackup/projects/3sl/labels/landcover/2m/CAS/*.tif'
tappan_etz_regex = '/explore/nobackup/projects/3sl/labels/landcover/2m/ETZ/*.tif'

tappan_cas_filenames = glob(tappan_cas_regex)
tappan_etz_filenames = glob(tappan_etz_regex)
print(f"CAS training tiles: {len(tappan_cas_filenames)}")
print(f"ETZ training tiles: {len(tappan_etz_filenames)}")


def set_dataframe(filenames, training=False):

    # define variables to store the output of the searches
    filename_list, scene_list, satellite_list, year_list, month_list,\
        day_list, study_area_list, bounds_list = [], [], [], [], [], [], [], []
    
    # iterate over filenames
    for filename in filenames:

        # mine some information from the string
        regex_match = re.split('_+', Path(filename).stem)

        # append some metadata
        filename_list.append(filename)

        if training:
            year_match = re.search(r'(\d{4})(\d{2})(\d{2})', regex_match[2])
            scene_list.append(regex_match[0])
            satellite_list.append(regex_match[1])
        else:
            year_match = re.search(r'(\d{4})(\d{2})(\d{2})', regex_match[1])
            satellite_list.append(regex_match[0])

        crs = rio.open(filename).crs
        bounds_list.append(box(*rio.open(filename).bounds))

        # adding site name
        if os.path.basename(os.path.dirname(filename)) == 'M1BS':
            study_area_list.append(os.path.basename(
                os.path.dirname(os.path.dirname(filename))))
        else:
            study_area_list.append(os.path.basename(
                os.path.dirname(filename)))

        # date metadata
        year_list.append(int(year_match.group(1)))
        month_list.append(int(year_match.group(2)))
        day_list.append(int(year_match.group(3)))

    if training:
        df_metadata = {
            'study_area': study_area_list,
            'filename': filename_list,
            'scene_id': scene_list,
            'satellite': satellite_list,
            'year': year_list,
            'month': month_list,
            'day': day_list,
            'geometry': bounds_list
        }
    else:
        df_metadata = {
            'study_area': study_area_list,
            'filename': filename_list,
            'satellite': satellite_list,
            'year': year_list,
            'month': month_list,
            'day': day_list,
            'geometry': bounds_list
        }
    return gpd.GeoDataFrame(df_metadata, crs=crs)

tappan_cas_gdf = set_dataframe(tappan_cas_filenames, training=True)
tappan_etz_gdf = set_dataframe(tappan_etz_filenames, training=True)
print(tappan_cas_gdf.head())

tappan_cas_gdf = set_dataframe(tappan_cas_filenames, training=True)
tappan_etz_gdf = set_dataframe(tappan_etz_filenames, training=True)
print(tappan_cas_gdf.head())

cas_filenames = glob(cas_regex)
etz_filenames = glob(etz_regex)
srv_filenames = glob(srv_regex)

print(f"CAS scenes: {len(cas_filenames)}")
print(f"ETZ scenes: {len(etz_filenames)}")
print(f"SRV scenes: {len(srv_filenames)}")

cas_gdf = set_dataframe(cas_filenames)
etz_gdf = set_dataframe(etz_filenames)
srv_gdf = set_dataframe(srv_filenames)
all_scenes_gdf = pd.concat([cas_gdf, etz_gdf, srv_gdf])
print(all_scenes_gdf.head())


def extract_raster(df, tiles_per_scene, tile_size, output_dir):
    _, row = df

    data = rxr.open_rasterio(row['filename'])
    print(data.shape)

    generated_tiles = 0  # counter for generated tiles
    metadata = dict()
    while generated_tiles < tiles_per_scene:

        # Generate random integers from image
        y = random.randint(0, data.shape[1] - tile_size)
        x = random.randint(0, data.shape[2] - tile_size)

        # Generate img and mask patches
        image_tile = data[:, y:(y + tile_size), x:(x + tile_size)].values

        # if image[x: (x + tile_size), y: (y + tile_size)].min() < -100:
        # second condition, if want zero nodata values, skip
        if np.any(image_tile > 10000) or np.any(image_tile < 0):
            continue

        generated_tiles += 1
        filename = f'{Path(row["filename"]).stem}_{generated_tiles}.npy'
        np.save(os.path.join(output_dir, filename), image_tile)
    return

def extract() -> None:

    #logging.info(f"Iterating over {len(wv_evhr_gdf)} rasters")
    #logging.info(f"Iterating over {atl08_gdf.shape[0]} points")

    # Generate tiles per month
    study_area = 'CAS'
    total_tiles = 25000
    tile_size = 256
    out_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/normalization'
    for m in range(1, 13):

        month_dataframe = all_scenes_gdf.loc[
            (all_scenes_gdf['study_area'] == study_area) & (all_scenes_gdf['month'] == m)]
        print(month_dataframe.shape)
        tiles_per_scene = int(total_tiles / month_dataframe.shape[0])
        print(tiles_per_scene)

        output_dir = os.path.join(out_dir, str(m))
        os.makedirs(output_dir, exist_ok=True)

        # multiprocessing pool
        p = Pool(processes=40)
        p.starmap(
            extract_raster,
            zip(
                month_dataframe.iterrows(),
                repeat(tiles_per_scene),
                repeat(tile_size),
                repeat(output_dir)
            )
        )
        p.close()
        p.join()
    return
"""

# extract()

out_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/normalization'
study_area = 'CAS'
new_array = []
for m in range(1, 13):
    filenames = glob(os.path.join(out_dir, str(m), '*.npy'))
    for filename in filenames:
        new_array.append(np.load(filename))
    new_array = np.array(new_array)
    print(new_array.shape)
    stats =  {
        'mean': np.mean(new_array, axis=(0, 2, 3)),
        'median': np.median(new_array, axis=(0, 2, 3)),
        'perc_1': np.percentile(new_array, q=1, axis=(0, 2, 3)),
        'perc_5': np.percentile(new_array, q=5, axis=(0, 2, 3)),
        'perc_95': np.percentile(new_array, q=95, axis=(0, 2, 3)),
        'perc_99': np.percentile(new_array, q=99, axis=(0, 2, 3)),
        'std': np.std(new_array, axis=(0, 2, 3)),
        'minimum': np.min(new_array, axis=(0, 2, 3)),
        'maximum': np.max(new_array, axis=(0, 2, 3)),
        'timestamp': [m] * new_array.shape[1],
    }
    #print(stats['mean'].shape)

    stats = pd.DataFrame.from_dict(stats)
    print(stats)
    stats.to_csv(os.path.join(out_dir, f'{study_area}_{str(m)}.csv'))
    new_array = []
