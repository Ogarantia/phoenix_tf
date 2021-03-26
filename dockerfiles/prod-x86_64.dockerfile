ARG TF_VERSION
FROM tensorflow/tensorflow:${TF_VERSION}

# add fancy UpStride bash welcome screen 
COPY dockerfiles/bash.bashrc /root/.bash_aliases

# install the python package in the docker
COPY build/upstride-*.whl /opt/
RUN export IMPORTLIBPATH=`python3 -c "import importlib; print(importlib.__path__[0])"` && \
    pip install /opt/upstride-*.whl && \
    export UPSTRIDELIBPATH=`python3 -c "import upstride; print(upstride.__path__._path[0])"` && \
    ls $UPSTRIDELIBPATH/libupstride.so && \
    ls $UPSTRIDELIBPATH/libdnnl.so && \
    ln -s $UPSTRIDELIBPATH/libupstride.so $IMPORTLIBPATH/libupstride.so && \
    ln -s $UPSTRIDELIBPATH/libdnnl.so $IMPORTLIBPATH/libdnnl.so && \
    rm /opt/upstride-*.whl && \
    unset IMPORTLIBPATH && \
    unset UPSTRIDELIBPATH

RUN python3 -m pip install packaging