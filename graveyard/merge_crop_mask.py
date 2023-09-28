from glob import glob
import rioxarray as rxr
import numpy as np
import os
import xarray as xr
from pathlib import Path

tree_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/standardization/landcover-local-standardization-512-tree/Tappan'

# too much crop
# crop_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/accuracy-increase/landcover-global-standardization-256-crop-4band-local/results/Tappan'

crop_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/accuracy-increase/8bit-scale-256-crop-short2-local/results/Tappan8bit'
tree_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/standardization/landcover-local-standardization-512-tree/Tappan'
otcb = '/explore/nobackup/projects/3sl/development/cnn_landcover/GMU_Experiments/land_cover_otcb_cas-wcas_60TS-set1/data/model/predictions/ForKonrad/*.tif'
output_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/GMU_Experiments/land_cover_otcb_cas-wcas_60TS-set1/data/model/predictions/ForKonrad/Tappan_Binary_Crop_Tree_8bit_Ensemble'


# otcb_filenames = sorted(glob('/explore/nobackup/projects/3sl/products/landcover/otcb_CAS_v3/Tappan/*.tif'))
# output_dir = '/explore/nobackup/projects/3sl/products/landcover/otcb_CAS_v3/Tappan_Binary_Tree'

#output_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/GMU_Experiments/land_cover_otcb_cas-wcas_60TS-set1/data/model/predictions/ForKonrad/Tappan_Binary_Tree'

#crop_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/accuracy-increase/landcover-global-standardization-256-crop-4band-short2-local/results/Tappan'
#otcb = '/explore/nobackup/projects/3sl/development/cnn_landcover/GMU_Experiments/land_cover_otcb_cas-wcas_60TS-set1/data/model/predictions/ForKonrad/*.tif'
#output_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/GMU_Experiments/land_cover_otcb_cas-wcas_60TS-set1/data/model/predictions/ForKonrad/Tappan_Binary_Crop_Tree_8bit'


# get otcb filenames
otcb_filenames = sorted(glob(otcb))


print(len(otcb_filenames))

os.makedirs(output_dir, exist_ok=True)

for otcb_filename in otcb_filenames:

    short_path = Path(otcb_filename).stem.split('.')[0]

    tree_filename = glob(os.path.join(tree_dir, f'{short_path}*.tif'))
    crop_filename = glob(os.path.join(crop_dir, f'{short_path}*.tif'))
    if len(tree_filename) > 0 and len(crop_filename) > 0:
        tree_filename = tree_filename[0]
        crop_filename = crop_filename[0]
        print(tree_filename, crop_filename)
    else:
        continue

    otcb = rxr.open_rasterio(otcb_filename)
    tree = rxr.open_rasterio(tree_filename)
    crop = rxr.open_rasterio(crop_filename)

    otcb_array = np.squeeze(otcb.values)
    tree_array = np.squeeze(tree.values)
    crop_array = np.squeeze(crop.values)

    print('pre unique tree count ', np.count_nonzero(otcb_array == 1))
    print('pre unique crop count ', np.count_nonzero(otcb_array == 2))

    otcb_array[crop_array == 1] = 2
    otcb_array[tree_array == 1] = 1

    print('post unique tree count', np.count_nonzero(otcb_array == 1))
    print('post unique crop count', np.count_nonzero(otcb_array == 2))

    # save output dir
    output_filename = os.path.join(output_dir, f'{Path(otcb_filename).stem}.tif')
    print(output_filename)

    # Drop image band to allow for a merge of mask
    otcb = otcb.drop(
        dim="band",
        labels=otcb.coords["band"].values[1:],
    )

    # Get metadata to save raster
    otcb_array = xr.DataArray(
        np.expand_dims(otcb_array, axis=0),
        name='otcb-crop-tree-binary',
        coords=otcb.coords,
        dims=otcb.dims,
        attrs=otcb.attrs
    )

    # Add metadata to raster attributes
    otcb_array.attrs['long_name'] = ('otcb-crop-tree-binary')
    otcb_array.attrs['model_name'] = ('landcover-local-standardization-512-tree')

    # Set nodata values on mask
    nodata = otcb_array.rio.nodata
    otcb_array = otcb_array.where(otcb != nodata)
    otcb_array.rio.write_nodata(
        nodata, encoded=True, inplace=True)

    # Save output raster file to disk
    otcb_array.rio.to_raster(
        output_filename,
        BIGTIFF="IF_SAFER",
        compress='LZW',
        driver='GTiff',
        dtype='uint8'
    )
