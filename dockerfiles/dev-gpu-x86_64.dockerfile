# Engine compilation environment with CUDA support for x64

ARG TF_VERSION
FROM tensorflow/tensorflow:${TF_VERSION}-gpu

# install cmake 3.18
RUN apt update && apt install -y apt-transport-https ca-certificates gnupg software-properties-common wget
RUN wget --tries=10 --retry-connrefused -q -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null &&\
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' &&\
    apt update && apt install -y cmake=3.18.3-0kitware1 cmake-data=3.18.3-0kitware1

# install additional tools and python packages
RUN apt install -y git nano
RUN python3 -m pip install nvgpu packaging cython

# install cuDNN and cuBLAS
RUN apt install -y \
        libcudnn7=7.6.5.32-1+cuda10.1  \
        libcudnn7-dev=7.6.5.32-1+cuda10.1  \
        libcublas-dev=10.1.0.105-1 \
    && rm -rf /var/lib/apt/lists/*

# install toolchain
COPY core/toolchain/gcc-8.4_8.4-1_x86_64.deb .
RUN dpkg -i gcc-8.4_8.4-1_x86_64.deb && \
    rm gcc-8.4_8.4-1_x86_64.deb && \
    update-alternatives --install /usr/bin/gcc gcc /usr/local/bin/gcc8.4  100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/local/bin/g++8.4  100
