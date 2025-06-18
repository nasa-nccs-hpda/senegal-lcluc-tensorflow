# Senegal LCLUC TensorFlow

[![DOI](https://zenodo.org/badge/474016543.svg)](https://zenodo.org/badge/latestdoi/474016543)
![CI Workflow](https://github.com/nasa-nccs-hpda/senegal-lcluc-tensorflow/actions/workflows/ci.yml/badge.svg)
![CI to DockerHub ](https://github.com/nasa-nccs-hpda/senegal-lcluc-tensorflow/actions/workflows/dockerhub.yml/badge.svg)
![Code style: PEP8](https://github.com/nasa-nccs-hpda/senegal-lcluc-tensorflow/actions/workflows/lint.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/nasa-nccs-hpda/senegal-lcluc-tensorflow/badge.svg?branch=main)](https://coveralls.io/github/nasa-nccs-hpda/senegal-lcluc-tensorflow?branch=main)

Python library to process and classify remote sensing imagery by means of GPUs and CPU parallelization for high performance and commodity base environments. This repository focuses in using CNNs for the inference of very
high-resolution remote sensing imagery in Senegal.

![study-area](docs/senegal-study-area.png)

Figure 1. Study area of this repository.

We are currently working on tutorials and documentations. Feel to follow this repository for documentation
updates and upcoming tutorials.

## Science Questions

- Can 2m VHR imagery enhance our understanding of changes in the extent, intensity and land use of agriculture and forestry in Senegal?
- How can we better take advantage of NASA’s HEC resources to apply Deep learning for land cover change monitoring?
- Can we scale-up Unet CNNs to compensate for the diversity of landscapes and images to map regional land cover?

![example-study](docs/senegal-example.png)

Figure 2. Wet and dry seasonality for cluster of typical fields in Senegal, Photos from Collaborator Gray Tappan
## Getting Started

1) Log on to the NASA ADAPT server

```bash
ssh user@adaptlogin.nasa.gov
```
2) Connect to PRISM GPU Cluster
```bash
ssh gpulogin1
```
3) Request computational resources with Slurm (see [NCCS guide to Slurm on ADAPT](https://www.nccs.nasa.gov/nccs-users/instructional/adapt-instructional/slurm)). For example, to request a quick interactive Slurm job:
```bash
salloc -G1 -J composite-text -c 10
```
which requests an interactive job (salloc), with 1 GPU (-G), a job name (-J) of "composite-text", and 10 CPU cores (-c). 

4) Create personal directories to work within:
- Confirm that the $USER environment variable is set to your username
 ```bash
echo $USER
```
which should return your username.
* Create a "development" directory
 ```bash
mkdir -p /explore/nobackup/people/$USER/development
```

5) Clone the GitHub
 ```bash
cd /explore/nobackup/people/$USER/development # Navigate to development folder
git clone https://github.com/nasa-nccs-hpda/senegal-lcluc-tensorflow
```

6) Load necessary module(s).
 ```bash
module load singularity
```

### Bonus Tips

 - If you are using a PIV card to connect, on your local system, save your login keychain to avoid repeatedly entering your pin by entering the following and answering the subsequent prompt:
 ```bash
   ssh-add -s /usr/lib/ssh-keychain.dylib
```
 - If you plan to open any graphical applications while logged on to the ADAPT server (e.g., QGIS), make sure you have an X11 server installed on your local system (such as [XQuartz](https://www.xquartz.org/) for Mac or [MobaXterm](https://mobaxterm.mobatek.net/) for Windows) and make sure you enable trusted X11 forwarding when logging on to the server:
 ```bash
   ssh -Y user@adaptlogin.nasa.gov
   ssh -Y gpulogin1
```

## Downloading the Container

The container for this work can be downloaded from DockerHub. The container is deployed on a weekly basis
to take care of potential OS vulnerabilities. All CPU and GPU dependencies are baked into the container image
for end-to-end processing.

```bash
singularity build --sandbox /lscratch/$USER/container/tensorflow-caney docker://nasanccs/tensorflow-caney:latest
```

## Data Preprocessing

The data used in this work is provided by NGA through the NextView agreement. As long as you are part of this
agreement, you can get access to the WorldView imagery we have processed. We also work with Planet imagery
as part of this project, which is free of access through the NICFI program.

### Generate Tappan Squares

To generate training and small samples of data we generated the so called Tappan squares in honor of our
colleague Gray Tappan. These Tappan squares are 5000x5000 pixel squares, for an area of 10000x10000 m^2.
Generating Tappan squares can be achieved with the tapann_pipeline_cli.py script. 
1. First create a folder to save output data:
```bash
mkdir -p /explore/nobackup/people/$USER/development/sample_data
```
2. Edit the "config" file to save data to the newly created output directory with the editor of your choice. For example, to edit the config file with Vim:
```bash
vim /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/tappan_generation/configs-srlite/tappan_06.yaml
```
Within the config file, set the output directory to: ```/explore/nobackup/people/$USER/development/sample_data/tappan_${square_number}```

4. Execute the Tappan Squares pipeline:

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow:/explore/nobackup/people/$USER/development/tensorflow-caney:/explore/nobackup/people/jacaraba/development" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/explore/nobackup/people /lscratch/$USER/container/tensorflow-caney python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/tappan_pipeline_cli.py -c /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/tappan_generation/configs-srlite/tappan_06.yaml
```

## Land Cover CNN Workflow

In this workflow we perform land cover segmentation using CNNs. The main steps from this workflow
are the preprocess, train, predict, and validate steps. The CLI binary is used to manage the execution
of these steps in the pipeline.

### Full Pipeline

Here we run the full pipeline for preprocess, train, and predict.

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow:/explore/nobackup/people/$USER/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/tensorflow-caney python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-AccuracyIncrease/global_standardization_256_crop_4band_short.yaml -d /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-AccuracyIncrease/land_cover_512_otcb_50TS_cas-wcas-short.csv -s preprocess train predict
```

Another example:

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow:/explore/nobackup/people/$USER/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/tensorflow-caney python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-AccuracyIncrease/8bit_scale_256_crop_4band_short.yaml -d /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-AccuracyIncrease/land_cover_512_otcb_50TS_cas-wcas-8bit-short.csv -s preprocess train predict
```

### Test

Generate metrics and statistics using test data.

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow:/explore/nobackup/people/$USER/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/explore/nobackup/people /explore/nobackup/projects/ilab/containers/tensorflow-caney-2023.05 python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-GMU-V2/batch1/eCAS-wCAS-otcb-40/eCAS-wCAS-otcb-40.yaml -t '/explore/nobackup/projects/3sl/labels/landcover/2m_all_fixed/*.tif' -s test
```

### Validate

Generate metrics and statistics using validation data.

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow:/explore/nobackup/people/jacaraba/development/tensorflow-caney",PROJ_LIB='/usr/share/proj' --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -v '/explore/nobackup/projects/3sl/data/Validation/3sl-validation-database-2023-10-05-all-row-based_all-agreed.gpkg' -c /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-GMU-V2/batch2/eCAS-wCAS-ETZ-otcb-80_repeat3/eCAS-wCAS-ETZ-otcb-80_repeat3.yaml -s validate
```

In some cases we need to perform larger scale validations (for example, N number of subdirectories). The following script allows you to perform the
validation using the subdirectory where the configuration file is available. All we ned to do is to point to the main directory
that houses all other configuration subdirectories.

This is an example for general TOA imagery using the GMU experiments for the paper:

```bash
bash validation_gmu.sh \
    "/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-GMU-V2/batch1/*/*.yaml" \
    "/explore/nobackup/projects/3sl/data/Validation/3sl-validation-database-2023-10-05-all-row-based_all-agreed.gpkg"
```

An an example for the Surface Reflectance experiments:

```bash
bash validation_srlite.sh \
    "/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/NASA/2023-surface_reflectance/ard_srlite_toa/*/*.yaml" \
    "/explore/nobackup/projects/3sl/data/Validation/3sl-validation-database-2023-10-05-all-row-based_all-agreed.gpkg"
```

## Compositing

After we generate predictions for the entire study area, we need to proceed to create composites. Below you will find the documentation to perform the compositing steps. This pipeline has 3 main steps:

1. Build footprints
2. Extract metadata
3. Build composite

Below you will find examples on how to run each one of these.

### Build footprints

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/vhr-composite:/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/tensorflow-caney-2023.05 python /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_composite_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/composite/configs/composite_cas.yaml -t /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/composite/configs/tile_lists/test_tile_0.txt -s build_footprints
```

### Extract metadata

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/vhr-composite:/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/tensorflow-caney-2023.05 python /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_composite_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/composite/configs/composite_cas.yaml -t /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/composite/configs/tile_lists/test_tile_0.txt -s extract_metadata
```

### Build composite

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/vhr-composite:/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/tensorflow-caney-2023.05 python /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_composite_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/composite/configs/composite_cas.yaml -t /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/composite/configs/tile_lists/tile_0.txt -s composite
```

## Land Use 1D CNN Workflow

In this workflow we perform land use object segmentation using CNNs.

### Setup

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow:/explore/nobackup/people/$USER/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/tensorflow-caney python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landuse_cnn_pipeline_cli.py -c /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_use/configs/landuse.yaml --gee-account 'id-sl-senegal-service-account@ee-3sl-senegal.iam.gserviceaccount.com' --gee-key '/home/$USER/gee/ee-3sl-senegal-8fa70fe1c565.json' -s setup
```

## CLoud Masking Imagery

In this workflow we use the [vhr-cloudmask]() software to generate cloud masks of our WorldView imagery.
The command to run cloud mask across the imagery from the PRISM GPU cluster is as follows, where:
- '-o' is the output directory
- '-r' is a list of regex where data files live
- '-s' is the pipeline step to perform which in this case is 'predict'

```bash
for i in {0..64}; do sbatch --mem-per-cpu=10240 -G1 -c10 -t05-00:00:00 -J clouds -q ilab --wrap="singularity exec --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/vhr-cloudmask.sif vhr-cloudmask-cli -o '/explore/nobackup/projects/3sl/products/cloudmask/v2' -r '/explore/nobackup/projects/3sl/data/Tappan/new_dates/Tappan*.data.tif' '/explore/nobackup/projects/3sl/data/VHR/new_evhr/5-toas/*toa.tif' -s predict"; done
```

## TOA vs ARD vs SRLite

The following documentation pertains to the TOA vs ARD vs SRLite experiments. The process requires several pieces and parts,
but here is a summary of the process for reproducibility. The overall process looks like this:

1. Identify TOA scenes to process.
2. Run EVHR.
3. Run SRLite.
4. Generate Tappan squares.
5. Run the different CNN experiments (training, inference, validation).
6. Visualize and summarize.

Below you will find the different commands to run each step from the pipeline.

### Identify TOA scenes to process

In this step we generate a CSV file with the respective TOA scenes available in ADAPT/Explore.
For this we use a notebook provided by mwooten that queries the NGA database.

### Run EVHR

For this step, we need the list of NTF files together with the output directory where we want to store
the output data. The output directory is to /explore/nobackup/projects/3sl/data/EVHR/<study_area>, where
study area is one of CAS, ETZ, SRV.

ILAB Nodes:

```bash
cd /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr
./run_evhr_exec /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/subset_2010/CAS_M1BS_scenes_noDups_2010_1.csv /explore/nobackup/projects/3sl/data/EVHR/CAS /explore/nobackup/people/iluser/ilab_containers/evhr_4.0.0.sif
```

PRISM Nodes:

```bash
salloc --ntasks 1 --cpus-per-task 40 -G1 -t 05-00:00:00 -J lightning
cd /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr
./run_evhr_exec /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/subset_2010/CAS_M1BS_scenes_noDups_2010_1.csv /explore/nobackup/projects/3sl/data/EVHR/CAS /lscratch/jacaraba/container/evhr-container
```

The following are the scenes we need to process via EVHR.

```bash
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups_2010.csv
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups_2011.csv
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups_2012.csv
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups_2013.csv
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups_2014.csv
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups_2015.csv
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups_2016.csv
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups_2017.csv
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups_2018.csv
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups_2019.csv
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups_2020.csv
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups_2021.csv
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups_2022.csv
/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/scripts/evhr/input_files_all_senegal/CAS_M1BS_scenes/CAS_M1BS_scenes_noDups.csv
```

## Authors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Margaret Wooten, margaret.wooten@nasa.gov
- Caleb S. Spradlin, caleb.s.spradlin@nasa.gov
- Minh Tri Le, mle35@gmu.edu 

## Installation

See the build [guide](requirements/README.md).

## Contributing

Please see our [guide for contributing to terragpu](CONTRIBUTING.md).

## References

[1] Raschka, S., Patterson, J., & Nolet, C. (2020). Machine learning in python: Main developments and technology trends in data science, machine learning, and artificial intelligence. Information, 11(4), 193.

[2] Paszke, Adam; Gross, Sam; Chintala, Soumith; Chanan, Gregory; et all, PyTorch, (2016), GitHub repository, <https://github.com/pytorch/pytorch>. Accessed 13 February 2020.

[3] Caraballo-Vega, J., Carroll, M., Li, J., & Duffy, D. (2021, December). Towards Scalable & GPU Accelerated Earth Science Imagery Processing: An AI/ML Case Study. In AGU Fall Meeting 2021. AGU.

