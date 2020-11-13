FROM nvcr.io/nvidia/l4t-base:r32.4.3
ARG TF_VERSION

# install tensorflow
RUN apt-get update && apt-get install -y python3-pip python3-dev build-essential pkg-config \
        libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev \
        gfortran libopenblas-dev liblapack-dev

RUN pip3 install -U pip cython
RUN pip3 install -U --no-binary=h5py h5py==2.10.0
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==${TF_VERSION}+nv20.8

# add fancy UpStride bash welcome screen 
COPY dockerfiles/bash.bashrc /root/.bash_aliases

# install the python package in the docker
COPY build/upstride-*.whl /opt/
RUN export IMPORTLIBPATH=`python3 -c "import importlib; print(importlib.__path__[0])"` && \
    pip install /opt/upstride-*.whl && \
    export UPSTRIDELIBPATH=`python3 -c "import upstride; print(upstride.__path__._path[0])"` && \
    ls $UPSTRIDELIBPATH/libupstride.so && \
    ls $UPSTRIDELIBPATH/libdnnl.so.1 && \
    ln -s $UPSTRIDELIBPATH/libupstride.so $IMPORTLIBPATH/libupstride.so && \
    ln -s $UPSTRIDELIBPATH/libdnnl.so.1 $IMPORTLIBPATH/libdnnl.so.1 && \
    rm /opt/upstride-*.whl && \
    unset IMPORTLIBPATH && \
    unset UPSTRIDELIBPATH
RUN python3 -m pip install packaging

# enabling libgomp preload to avoid an allocation issue
# https://github.com/keras-team/keras-tuner/issues/317
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
