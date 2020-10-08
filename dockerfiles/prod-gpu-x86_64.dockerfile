ARG TF_VERSION
FROM tensorflow/tensorflow:${TF_VERSION}-gpu

# add fancy UpStride bash welcome screen 
COPY dockerfiles/bash.bashrc /root/.bash_aliases

# install the python package in the docker
COPY src/python/ /opt/upstride
RUN cd /opt/upstride && \
    ls upstride/type_generic/_upstride.so && \
    pip install . && \
    cd / && \
    rm -r /opt/upstride
