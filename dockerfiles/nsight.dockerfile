# from https://developer.nvidia.com/blog/nvidia-nsight-systems-containers-cloud/
FROM nvidia/cudagl:10.1-base-ubuntu18.04
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        dbus \
        fontconfig \
        gnupg \
        libfreetype6 \
        libglib2.0-0 \
        libsqlite3-0 \
        libx11-xcb1 \
        libxcb-glx0 \
        libxcb-xkb1 \
        libxcomposite1 \
        libxi6 \
        libxml2 \
        libxrender1 \
        openssh-client \
        wget \
        xcb \
        xkb-data \
        libasound2 \
        libnss3 \
        libxcursor1 \
        libxrandr2 \
        libxtst6 \
        libxkbcommon-x11-0 \
        nsight-compute-2019.5.0 \
        nsight-systems-2019.5.2

RUN echo "export PATH=/opt/nvidia/nsight-compute/2019.5.0/:$PATH" >> /root/.bashrc 
RUN echo "export LD_LIBRARY_PATH=/opt/nvidia/nsight-compute/2019.5.0/host/linux-desktop-glibc_2_11_3-x64/:$LD_LIBRARY_PATH" >> /root/.bashrc 
WORKDIR /opt/src
