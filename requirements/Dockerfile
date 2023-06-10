FROM nasanccs/tensorflow-caney:latest
LABEL maintainer="jordan.a.caraballo-vega@nasa.gov"

ENV TFC_VERSION=0.1.1
ENV SENEGAL_LCLUC_TF_VERSION=main

RUN pip install git+https://github.com/dshean/pygeotools.git@master

#RUN pip install pygsl pyod cvxopt gsutil feather-format openpyxl SALib \
#    alphashape xlsxwriter black flake8 pylint pytest pytest-html coverage \
#    pytest-cov pysptools pymcr imbalanced-learn

ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

WORKDIR /opt

RUN apt-get update && \
    apt-get -y install libgsl-dev libmuparser-dev build-essential g++ autotools-dev \
    libicu-dev libbz2-dev libboost-all-dev libhdf5-serial-dev libhdf5-dev

RUN mkdir -p /opt/rsgislib_bld && \
    cd /opt/rsgislib_bld && \
    git clone https://github.com/remotesensinginfo/rsgislib.git && \
    cd rsgislib && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_SKIP_RPATH=OFF \
        -DGDAL_INCLUDE_DIR=/usr/include -DGDAL_LIB_PATH=/usr/lib \
        -DGSL_INCLUDE_DIR=/usr/include -DGSL_LIB_PATH=/usr/lib \
        -DBOOST_INCLUDE_DIR=/usr/include -DBOOST_LIB_PATH=/usr/lib \
        -DMUPARSER_INCLUDE_DIR=/usr/include -DMUPARSER_LIB_PATH=/usr/lib \
        -DHDF5_INCLUDE_DIR=/usr/include/hdf5/serial -DHDF5_LIB_PATH=/usr/lib/x86_64-linux-gnu/hdf5/serial .. && \
    make -j 4 && \
    make install && \
    rm -rf /opt/rsgislib_bld

# RUN mamba install --yes -c au-eoed -c conda-forge python=3.10 
# gdal - done
# geos - done
# gsl - maybe pygsl
# kealib 
# xerces-c 
# muparser 
# boost-cpp 
# rios 
# scikit-learn 
# scikit-image 
# scikit-optimize 
# imbalanced-learn - done
# scikit-plot 
# scikit-fuzzy 
# bayesian-optimization 
# matplotlib 
# pandas 
# geopandas
# statsmodels
# h5py
# scipy
# rasterio
# shapely
# networkx
# sqlalchemy
# pycurl
# xgboost 
# catboost
# lightgbm 
# tpot 
# seaborn 
# ml_tooling 
# cartopy 
# numba 
# pip 
# sphinx 
# elevation 
# rtree 
# tqdm 
# jinja2 
# keras 
# keras-preprocessing 
# pytables 
# parallel 
# bokeh 
# pygal 
# jupyterlab 
# psutil 
# pysal 
# libpysal 
# esda 
# pyyaml 
# netcdf4 
# xarray 
# rasterstats 
# fiona 
# plotly 
# python-kaleido; sync \

# if all the dependencies are within the same path (e.g., installed with conda)
# then use a variable to specify the path.
#export DEPEND_ENV_PATH=/home/user/miniconda/envs/rsgislib_v5_dev

# Set the install location and path to dependencies
#cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
#-D BOOST_INCLUDE_DIR=$DEPEND_ENV_PATH/include \
#-D BOOST_LIB_PATH=$DEPEND_ENV_PATH/lib \
#-D GDAL_INCLUDE_DIR=$DEPEND_ENV_PATH/include \
#-D GDAL_LIB_PATH=$DEPEND_ENV_PATH/lib \
#-D HDF5_INCLUDE_DIR=$DEPEND_ENV_PATH/include \
#-D HDF5_LIB_PATH=$DEPEND_ENV_PATH/lib \
#-D GSL_INCLUDE_DIR=$DEPEND_ENV_PATH/include \
#-D GSL_LIB_PATH=$DEPEND_ENV_PATH/lib \
#-D MUPARSER_INCLUDE_DIR=$DEPEND_ENV_PATH/include \
#-D MUPARSER_LIB_PATH=$DEPEND_ENV_PATH/lib \
#-D KEA_INCLUDE_DIR=$DEPEND_ENV_PATH/include \
#-D KEA_LIB_PATH=$DEPEND_ENV_PATH/lib \
#-D Python_EXECUTABLE=$DEPEND_ENV_PATH/bin/python \
#-D CMAKE_VERBOSE_MAKEFILE=ON \
#-D CMAKE_BUILD_TYPE=Debug \
#-D CMAKE_SKIP_RPATH=ON \
#..
  
# build rsgislib -j N is the number of cores to use for build 
#make -j 2
  
# Install the build into the install prefix.
#make install

HEALTHCHECK NONE
ENTRYPOINT [""]
