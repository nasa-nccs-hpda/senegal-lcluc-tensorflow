# Senegal LCLUC Experiments

Here we want to document the experiments proposed by the GMU team. The idea
is to test seasonality effects in the classification of data.

## 1. Data Locations

The following table describes the data and data locations used in this project.

| Data Description   | Data Path                                                            |
| :---------------:  | :------------------------------------------------------------------: |
| Multispectral      | /att/nobackup/mwooten3/Senegal_LCLUC/VHR/priority-tiles              |
| Tappan Squares     | /att/nobackup/mcarrol2/LCLUC_Senegal/ForKonrad                       |
| Tappan Labels      | /adapt/nobackup/projects/ilab/projects/Senegal/Casamance_training_v2 |
| Cloud Masks        | TBD                                                                  |
| SAR                | /att/nobackup/nmthoma1/LCLUC_Senegal/CAS/CAS_S1_norm                 |
| CHM                | /att/nobackup/pmontesa/chm_work/hrsi_chm_senegal                     |

The classes included are:

- 1: trees/shrub
- 2: crop
- 3: other vegetation
- 4: water/shadow
- 5: burn
- 6: clouds
- 7: nodata

## 3. Overall Experiments

The following table describes the experiments behind this. The following definition covers dry vs wet season.

"The climate of Senegal, a West African country between the Tropic and the Equator, is tropical, with a long dry season (which runs roughly from mid-October to mid-June in the north, and from early November to mid-May in the south) and a rainy season (approximately, from late June to early October in the north, and from late May to late October in the south) due to the African monsoon, which in summer arrives from the south."

Also, they want to segregate the time in T1 (2011) and T2 (2016).

| Experiment                                    | Data Sources           | Training        |
| :-----------------------------------------:   | :--------------------: | :-------------: |
| Dry season only data to predict all other     | multi-spectral         | dry season only |
| Wet season only data to predict all other     | multi-spectral         | wet season only |
| Dry+Wet season data to predict all other      | multi-spectral         | dry+wet         |
| Dry+Wet+TL season data to predict all other   | multi-spectral,tf      | dry+wet+tf      |
| Dry+Wet+SAR data to predict all other         | multi-spectral,sar     | dry+wet+sar     |
| Dry+Wet+CHM data to predict all other         | multi-spectral,chm     | dry+wet+chm     |
| Dry+Wet+SAR+CHM data to predict all other     | multi-spectral,sar,chm | dry+wet+sar+chm |

** TF: transfer learning

In this case:

| Data Description   | Data Path                                                |
| :---------------:  | :------------------------------------------------------: |
| Dry                | Tappan02_WV02_20120218_M1BS_103001001077BE00_mask_v2.tif |
| Dry                | Tappan05_WV02_20110207_M1BS_1030010008B55200_mask_v2.tif |
| Dry                | Tappan02_WV03_20160123_M1BS_1040010018A59100_mask_v2.tif |
| Dry                | Tappan05_WV02_20110430_M1BS_103001000A27E100_mask_v2.tif |
| Dry                | Tappan04_WV02_20120218_M1BS_103001001077BE00_mask_v2.tif |
| Dry                | Tappan05_WV02_20181217_M1BS_1030010089CC6D00_mask_v2.tif |
| Dry                | Tappan04_WV03_20160123_M1BS_1040010018A59100_mask_v2.tif |

Also, experiments going forward from Konrad:

- Use dry season images only – 70-80% coverage
- Where dry season images are missing, use wet season images, but with dry season labels, if possible (unlikely)
- Need to include more TS ‘s that are more widely distributed. (Neighboring Tappan squares (TS) could have the exact same WV images, so different TS are not necessarily different areas or images to which models are transferred). Maggie process more widely distributed TS’s in Cassemanse.
- Most TS’s in Cassemanse only have 1 or 2 dry season images per epoch (2011-2013 vs. 2016-present) – this simplifies experiments and make it more manageble.
- Need to produce a T1 and T2 map and start looking at change error propagation.

### 3.1 Dry season only data to predict all other

In this experiment we explore using dry season data to predict all of the data.
