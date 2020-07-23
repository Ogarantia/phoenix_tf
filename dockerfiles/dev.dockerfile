FROM tensorflow/tensorflow:2.2.0

RUN apt update && apt install -y cmake \
    && rm -rf /var/lib/apt/lists/*

COPY custom_gcc/gcc-8.4_8.4-1_amd64.deb .
RUN dpkg -i gcc-8.4_8.4-1_amd64.deb && \
    rm gcc-8.4_8.4-1_amd64.deb && \
    update-alternatives --install /usr/bin/gcc gcc /usr/local/bin/gcc8.4  100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/local/bin/g++8.4  100
