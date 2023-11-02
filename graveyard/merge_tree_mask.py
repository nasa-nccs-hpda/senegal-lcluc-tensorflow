from glob import glob
import rioxarray as rxr
import numpy as np
import os
import xarray as xr
from pathlib import Path

tree_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/standardization/landcover-local-standardization-512-tree/Tappan'


# otcb_filenames = sorted(glob('/explore/nobackup/projects/3sl/products/landcover/otcb_CAS_v3/Tappan/*.tif'))
# output_dir = '/explore/nobackup/projects/3sl/products/landcover/otcb_CAS_v3/Tappan_Binary_Tree'

otcb = '/explore/nobackup/projects/3sl/development/cnn_landcover/GMU_Experiments/land_cover_otcb_cas-wcas_60TS-set1/data/model/predictions/ForKonrad/*.tif'

otcb_filenames = sorted(glob(otcb))
output_dir = '/explore/nobackup/projects/3sl/development/cnn_landcover/GMU_Experiments/land_cover_otcb_cas-wcas_60TS-set1/data/model/predictions/ForKonrad/Tappan_Binary_Tree'
print(len(otcb_filenames))

os.makedirs(output_dir, exist_ok=True)

for otcb_filename in otcb_filenames:

    #tree_filename = os.path.join(tree_dir, f'{Path(otcb_filename).stem}.tif')
    tree_filename = os.path.join(tree_dir, f'{Path(otcb_filename).stem[:-10]}.otcb.tif')

    print(tree_filename)

    if os.path.isfile(tree_filename):

        otcb = rxr.open_rasterio(otcb_filename)
        tree = rxr.open_rasterio(tree_filename)
        print(Path(otcb_filename).stem, otcb.shape, Path(tree_filename).stem, tree.shape)

        otcb_array = np.squeeze(otcb.values)
        tree_array = np.squeeze(tree.values)

        print('pre unique tree count ', np.count_nonzero(otcb_array == 1))

        otcb_array[tree_array == 1] = 1

        print('post unique tree count', np.count_nonzero(otcb_array == 1))

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
            name='otcb-tree-binary',
            coords=otcb.coords,
            dims=otcb.dims,
            attrs=otcb.attrs
        )

        # Add metadata to raster attributes
        otcb_array.attrs['long_name'] = ('otcb-tree-binary')
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
