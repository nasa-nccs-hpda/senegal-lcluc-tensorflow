import ee
import os
import geedim as gd
from glob import glob
from pathlib import Path
import rioxarray as rxr
from shapely.geometry import box

# input to download data for
gee_account = 'id-sl-senegal-service-account@ee-3sl-senegal.iam.gserviceaccount.com'
gee_key = '/home/jacaraba/gee/ee-3sl-senegal-8fa70fe1c565.json'
#filename = '/explore/nobackup/projects/3sl/data/Tappan/Tappan21_WV02_20190119_M1BS_103001008B4CF000_data.tif'
output_dir = '/explore/nobackup/projects/3sl/labels/landuse/radar_data'

# get credentials
credentials = ee.ServiceAccountCredentials(gee_account, gee_key)
ee.Initialize(credentials)  # gd initialize does not take service account
print("Initialized")

filenames = glob('/explore/nobackup/people/mcarrol2/LCLUC_Senegal/ForKonrad/*_data.tif')
for filename in filenames:

    print(filename)

    tmp_output_dir = os.path.join(output_dir, Path(filename).stem.split('_')[0])

    if os.path.exists(tmp_output_dir):
        print(tmp_output_dir)
        print("skipping")
        continue

    os.makedirs(tmp_output_dir, exist_ok=True)

    # get boundary
    raster = rxr.open_rasterio(filename).rio.reproject("EPSG:4326")
    xx, yy = box(*raster.rio.bounds()).exterior.coords.xy
    image_bounds = [[x, y] for x, y in zip(xx, yy)]
    region = ee.Geometry.Polygon(image_bounds)

    # iterate over the years we need
    for year in range(2017, 2023, 1):

        for polarization in ['VV', 'VH']:

            output_filename = os.path.join(tmp_output_dir, f"{Path(filename).stem.split('_')[0]}-{polarization}-{year}.tif")

            def compute_gee_power(image: ee.image.Image) -> ee.element.Element:
                return ee.Image(10).pow(image.divide(10)).rename(polarization).copyProperties(image, ['system:time_start'])

            # get collection
            gee_image_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .filterBounds(region) \
                .select(polarization) \
                .filterDate(f'{year}-01-01', f'{year}-12-31')
            print('S1 ImageCollection: ' +f'{len(gee_image_collection.toBands().bandNames().getInfo())}')

            # compute power to normalize
            gee_image_collection = gee_image_collection.map(compute_gee_power)
            print('Processed logarithmic power for ' + f'{len(gee_image_collection.toBands().bandNames().getInfo())} bands')

            # get the median month sequence
            months = ee.List.sequence(1, 12)
            gee_image_collection = ee.ImageCollection.fromImages(
                months.map(
                    lambda month: gee_image_collection.filter(
                        ee.Filter.calendarRange(month, month, 'month')
                    ).median().set('month', month)
                )
            )

            # clip the imagery again
            gee_image = gee_image_collection.toBands().clip(region)
            print('Processed monthly mean, resulted in ' + f'{len(gee_image.bandNames().getInfo())} bands')

            im = gd.mask.MaskedImage(gee_image)
            im.download(output_filename, region=region, crs='EPSG:32628', scale=10)
