import sys
import pandas as pd
import geopandas as gpd
from glob import glob
from multiprocessing import Pool, cpu_count
import os
import argparse
import h5py
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio as rio
import rioxarray as rxr
import matplotlib.pyplot as plt
from itertools import repeat


from rasterio.plot import show, show_hist
from glob import glob
from pathlib import Path
from scipy import ndimage
from scipy.stats import entropy
from skimage.transform import rescale, resize
from shapely.geometry import Point
from shapely.geometry import box
from sklearn import metrics

import warnings
from shapely.errors import ShapelyDeprecationWarning

def read_gedi_path(gedi_path, toa_path):
    """
    Read filenames from gedi path and generate geopackage.
    """
    output_filename = os.path.join(gedi_path, Path(os.path.basename(gedi_path)).with_suffix('.gpkg'))
    if os.path.exists(output_filename):
        return
    
    gedi_gdfs = []
    for gedi_filename in glob(os.path.join(gedi_path, '*.h5')):

        # get boundaries from toa filename
        toa_filename = os.path.join(toa_path, Path(os.path.basename(gedi_path)).with_suffix('.tif'))
        raster = rio.open(toa_filename)
        toa_gdf = gpd.GeoDataFrame({"id":1,"geometry":[box(*raster.bounds)]}, crs=raster.crs)
        
        # read gedi filename and create dataframe
        gediL2A = h5py.File(gedi_filename, 'r')
        gediL2A_objs = []
        gediL2A.visit(gediL2A_objs.append)
        gediSDS = [o for o in gediL2A_objs if isinstance(gediL2A[o], h5py.Dataset)]  # Search for relevant SDS inside data file

        # get full power beam name
        beamNames = [g for g in gediL2A.keys() if g.startswith('BEAM')]
        print("All beams:  ", beamNames)
        beamNames = [b for b in beamNames if gediL2A[b].attrs['description'] == "Full power beam"]
        print("Full beams: ", beamNames)
        
        # set general lists to store data
        shotNum, dem, zElevation, zHigh, zLat, zLon, rh25, rh90, rh98, rh100, quality, degrade, sensitivity, beamI = ([] for i in range(14))  

        # iterate over each beam
        for b in beamNames:
            [shotNum.append(h) for h in gediL2A[[g for g in gediSDS if g.endswith('/shot_number') and b in g][0]][()]]
            [dem.append(h) for h in gediL2A[[g for g in gediSDS if g.endswith('/digital_elevation_model') and b in g][0]][()]]
            [zElevation.append(h) for h in gediL2A[[g for g in gediSDS if g.endswith('/elev_lowestmode') and b in g][0]][()]]  
            [zHigh.append(h) for h in gediL2A[[g for g in gediSDS if g.endswith('/elev_highestreturn') and b in g][0]][()]]  
            [zLat.append(h) for h in gediL2A[[g for g in gediSDS if g.endswith('/lat_lowestmode') and b in g][0]][()]]  
            [zLon.append(h) for h in gediL2A[[g for g in gediSDS if g.endswith('/lon_lowestmode') and b in g][0]][()]]  
            [rh25.append(h[25]) for h in gediL2A[[g for g in gediSDS if g.endswith('/rh') and b in g][0]][()]]  
            [rh90.append(h[90]) for h in gediL2A[[g for g in gediSDS if g.endswith('/rh') and b in g][0]][()]]  
            [rh98.append(h[98]) for h in gediL2A[[g for g in gediSDS if g.endswith('/rh') and b in g][0]][()]]
            [rh100.append(h[100]) for h in gediL2A[[g for g in gediSDS if g.endswith('/rh') and b in g][0]][()]]  
            [quality.append(h) for h in gediL2A[[g for g in gediSDS if g.endswith('/quality_flag') and b in g][0]][()]]  
            [degrade.append(h) for h in gediL2A[[g for g in gediSDS if g.endswith('/degrade_flag') and b in g][0]][()]]  
            [sensitivity.append(h) for h in gediL2A[[g for g in gediSDS if g.endswith('/sensitivity') and b in g][0]][()]]  
            [beamI.append(h) for h in [b] * len(gediL2A[[g for g in gediSDS if g.endswith('/shot_number') and b in g][0]][()])]  

        # Convert lists to Pandas dataframe
        allDF = pd.DataFrame({
            'Shot Number': shotNum, 'Beam': beamI, 'Latitude': zLat, 'Longitude': zLon,
            'Tandem-X DEM': dem, 'Elevation (m)': zElevation, 'Canopy Elevation (m)': zHigh,
            'Canopy Height (rh100)': rh100, 'RH 98': rh98, 'RH 90': rh90, 'RH 25': rh25, 'Quality Flag': quality,
            'Degrade Flag': degrade, 'Sensitivity': sensitivity
        })
        
        # delete variables data
        del beamI, degrade, dem, gediSDS, rh100, rh98, rh90, rh25, quality, sensitivity, zElevation, zHigh, zLat, zLon, shotNum

        # convert to geodataframe
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
            allDF['geometry'] = allDF.apply(lambda row: Point(row.Longitude, row.Latitude), axis=1)
        allDF = gpd.GeoDataFrame(allDF, crs='EPSG:4326')
        allDF = allDF.to_crs('EPSG:32628')
        print(allDF.shape)
        
        allDF = allDF.clip(toa_gdf)
        print(allDF.shape)
        
        if allDF.shape[0] != 0:
            gedi_gdfs.append(allDF)
            #allDF.to_file(os.path.join(gedi_path, Path(gedi_filename).with_suffix('.gpkg')))

    try:
        allGDF = pd.concat(gedi_gdfs, axis=0).reset_index(drop=True)
        print(allGDF.shape, type(allGDF))
        allGDF.to_file(output_filename)
    except:
        return

    return

def main():

    # ETZ
    gedi_path_regex = '/explore/nobackup/projects/ilab/projects/AIML_CHM/CNN_CHM/data/GEDI/ETZ/*'
    toa_path = '/explore/nobackup/projects/3sl/data/VHR/ETZ/M1BS'
    gedi_path = '/explore/nobackup/projects/ilab/projects/AIML_CHM/CNN_CHM/data/GEDI/ETZ'

    #gedi_files = []
    #selections = pd.read_csv('/home/jacaraba/selection.csv', header=None)
    #for _, row in selections.iterrows():
    #    gedi_files.append(os.path.join(gedi_path, f'{row.values[0]}-toa'))
    gedi_files = glob(gedi_path_regex)

    # Distribute a file per processor
    p = Pool(processes=cpu_count())
    p.starmap(
        read_gedi_path,
        zip(
            gedi_files,
            repeat(toa_path)
        )
    )
    p.close()
    p.join()


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()