##
# osgeo/gdal:ubuntu-full

# This file is available at the option of the licensee under:
# Public domain
# or licensed under MIT (LICENSE.TXT) Copyright 2019 Even Rouault <even.rouault@spatialys.com>
ARG TF_VER=tf2
ARG VERSION_DATE=23.05
ARG FROM_IMAGE=nvcr.io/nvidia/tensorflow

#ARG BASE_IMAGE=ubuntu:20.04
#ARG TARGET_BASE_IMAGE=ubuntu:20.04

ARG BASE_IMAGE=${FROM_IMAGE}:${VERSION_DATE}-${TF_VER}-py3
ARG TARGET_BASE_IMAGE=${FROM_IMAGE}:${VERSION_DATE}-${TF_VER}-py3

FROM $BASE_IMAGE as builder

# Derived from osgeo/proj by Howard Butler <howard@hobu.co>
LABEL maintainer="Even Rouault <even.rouault@spatialys.com>"

ARG TARGET_ARCH=
RUN echo ${TARGET_ARCH}
COPY ./requirements/bh-set-envvars.sh /buildscripts/bh-set-envvars.sh

RUN . /buildscripts/bh-set-envvars.sh \
    && if test "${TARGET_ARCH}" != ""; then \
    rm -f /etc/apt/sources.list \
    && echo "deb [arch=amd64] http://us.archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb [arch=amd64] http://us.archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb [arch=amd64] http://us.archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb [arch=amd64] http://security.ubuntu.com/ubuntu jammy-security main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb [arch=${TARGET_ARCH}] http://ports.ubuntu.com/ubuntu-ports/ jammy main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb [arch=${TARGET_ARCH}] http://ports.ubuntu.com/ubuntu-ports/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb [arch=${TARGET_ARCH}] http://ports.ubuntu.com/ubuntu-ports/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list \
    && dpkg --add-architecture ${TARGET_ARCH} \
    && apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y g++-11-${GCC_ARCH}-linux-gnu \
    && ln -s ${GCC_ARCH}-linux-gnu-gcc-11 /usr/bin/${GCC_ARCH}-linux-gnu-gcc \
    && ln -s ${GCC_ARCH}-linux-gnu-g++-11 /usr/bin/${GCC_ARCH}-linux-gnu-g++ \
    && rm -rf /var/lib/apt/lists/*; \
    fi

# Setup build env for PROJ
USER root
RUN . /buildscripts/bh-set-envvars.sh \
    && apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing --no-install-recommends \
            build-essential ca-certificates \
            git make cmake wget unzip libtool automake \
            zlib1g-dev${APT_ARCH_SUFFIX} libsqlite3-dev${APT_ARCH_SUFFIX} pkg-config sqlite3 libcurl4-openssl-dev${APT_ARCH_SUFFIX} \
            libtiff5-dev${APT_ARCH_SUFFIX} \
    && rm -rf /var/lib/apt/lists/*

ARG JAVA_VERSION=17
# Setup build env for GDAL
RUN . /buildscripts/bh-set-envvars.sh \
    && apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing --no-install-recommends \
       libopenjp2-7-dev${APT_ARCH_SUFFIX} libcairo2-dev${APT_ARCH_SUFFIX} \
       python3-dev${APT_ARCH_SUFFIX} python3-numpy${APT_ARCH_SUFFIX} python3-setuptools${APT_ARCH_SUFFIX} \
       libpng-dev${APT_ARCH_SUFFIX} libjpeg-dev${APT_ARCH_SUFFIX} libgif-dev${APT_ARCH_SUFFIX} liblzma-dev${APT_ARCH_SUFFIX} libgeos-dev${APT_ARCH_SUFFIX} \
       curl libxml2-dev${APT_ARCH_SUFFIX} libexpat-dev${APT_ARCH_SUFFIX} libxerces-c-dev${APT_ARCH_SUFFIX} \
       libnetcdf-dev${APT_ARCH_SUFFIX} libpoppler-dev${APT_ARCH_SUFFIX} libpoppler-private-dev${APT_ARCH_SUFFIX} \
       libspatialite-dev${APT_ARCH_SUFFIX} librasterlite2-dev${APT_ARCH_SUFFIX} swig ant libhdf4-alt-dev${APT_ARCH_SUFFIX} libhdf5-serial-dev${APT_ARCH_SUFFIX} \
       libfreexl-dev${APT_ARCH_SUFFIX} unixodbc-dev${APT_ARCH_SUFFIX}  mdbtools-dev${APT_ARCH_SUFFIX} libwebp-dev${APT_ARCH_SUFFIX} \
       liblcms2-2 libpcre3-dev${APT_ARCH_SUFFIX} libcrypto++-dev${APT_ARCH_SUFFIX} libfyba-dev${APT_ARCH_SUFFIX} \
       libkml-dev${APT_ARCH_SUFFIX} libmysqlclient-dev${APT_ARCH_SUFFIX} libogdi-dev${APT_ARCH_SUFFIX} \
       libcfitsio-dev${APT_ARCH_SUFFIX} openjdk-"$JAVA_VERSION"-jdk${APT_ARCH_SUFFIX} libzstd-dev${APT_ARCH_SUFFIX} \
       libpq-dev${APT_ARCH_SUFFIX} libssl-dev${APT_ARCH_SUFFIX} libboost-dev${APT_ARCH_SUFFIX} \
       autoconf automake bash-completion libarmadillo-dev${APT_ARCH_SUFFIX} \
       libopenexr-dev${APT_ARCH_SUFFIX} libheif-dev${APT_ARCH_SUFFIX} \
       libdeflate-dev${APT_ARCH_SUFFIX} libblosc-dev${APT_ARCH_SUFFIX} liblz4-dev${APT_ARCH_SUFFIX} libbz2-dev${APT_ARCH_SUFFIX} \
       libbrotli-dev${APT_ARCH_SUFFIX} \
       libarchive-dev${APT_ARCH_SUFFIX} \
    && rm -rf /var/lib/apt/lists/*
