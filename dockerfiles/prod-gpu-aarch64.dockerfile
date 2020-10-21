FROM nvcr.io/nvidia/l4t-base:r32.4.3
ARG TF_VERSION

# install tensorflow
RUN apt-get update && apt-get install -y python3-pip python3-dev build-essential pkg-config \
        libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev \
        gfortran libopenblas-dev liblapack-dev

RUN pip3 install -U pip cython
RUN pip3 install -U --no-binary=h5py h5py 
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==${TF_VERSION}+nv20.8

# add fancy UpStride bash welcome screen 
COPY dockerfiles/bash.bashrc /root/.bash_aliases

# install the python package in the docker
COPY build/ /opt/upstride
RUN export IMPORTLIBPATH=`python3 -c "import importlib; print(importlib.__path__[0])"`
RUN cd /opt/upstride && \
    ls libs/libupstride.so && \
    ls core/thirdparty/onednn/src/libdnnl.so.1 && \
    ln -s /opt/upstride/libs/libupstride.so $IMPORTLIBPATH/libupstride.so && \
    ln -s /opt/upstride/core/thirdparty/onednn/src/libdnnl.so.1 $IMPORTLIBPATH/libdnnl.so.1 && \
    ls $IMPORTLIBPATH/libupstride.so && \
    ls $IMPORTLIBPATH/libdnnl.so.1 && \
    pip install upstride-*.whl && \
    cd / && rm -r /opt/upstride
RUN unset IMPORTLIBPATH

RUN python3 -m pip install packaging

# enabling libgomp preload to avoid an allocation issue
# https://github.com/keras-team/keras-tuner/issues/317
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
