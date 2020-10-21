FROM nvcr.io/nvidia/l4t-base:r32.4.3
ARG TF_VERSION

# install tensorflow
RUN apt-get update && apt-get install -y python3-pip python3-dev build-essential pkg-config \
        libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev \
        gfortran libopenblas-dev liblapack-dev

RUN pip3 install -U pip cython
RUN pip3 install -U --no-binary=h5py h5py 
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==${TF_VERSION}+nv20.8

# wget and compile cmake 3.18
RUN apt update && apt install -y wget libssl1.0-dev
RUN wget --tries=10 --retry-connrefused -q https://github.com/Kitware/CMake/releases/download/v3.18.3/cmake-3.18.3.tar.gz &&\
    tar -xzf cmake-3.18.3.tar.gz
RUN cd cmake-3.18.3 &&\
    chmod +x bootstrap &&\
    ./bootstrap &&\
    make -j$(nproc) &&\
    make install &&\
    rm -rf cmake-3.18.3*

# install additional tools and python packages
RUN apt install -y git nano
RUN python3 -m pip install nvgpu packaging cython

# install toolchain
COPY core/toolchain/gcc-8.4_8.4-1_aarch64.deb .
RUN dpkg -i gcc-8.4_8.4-1_aarch64.deb &&\
    rm gcc-8.4_8.4-1_aarch64.deb &&\
    update-alternatives --install /usr/bin/gcc gcc /usr/local/bin/gcc  100 &&\
    update-alternatives --install /usr/bin/g++ g++ /usr/local/bin/g++  100