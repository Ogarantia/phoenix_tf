ARG TF_VERSION
FROM tensorflow/tensorflow:${TF_VERSION}-gpu

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