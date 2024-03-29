# tensorflow/tensorflow:2.3.0-gpu already contains nvprof, gprof, tensorboad
FROM tensorflow/tensorflow:2.3.0-gpu

RUN apt update && apt install -y \
        cmake \
        git \
        libcudnn7-dev=7.6.4.38-1+cuda10.1 \
        libcublas10 \
    && rm -rf /var/lib/apt/lists/* \ 
    && python3 -m pip install nvgpu cython

COPY core/toolchain/gcc-8.4_8.4-1_amd64.deb .
RUN dpkg -i gcc-8.4_8.4-1_amd64.deb && \
    rm gcc-8.4_8.4-1_amd64.deb && \
    update-alternatives --install /usr/bin/gcc gcc /usr/local/bin/gcc8.4  100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/local/bin/g++8.4  100
