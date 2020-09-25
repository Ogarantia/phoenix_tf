FROM tensorflow/tensorflow:2.3.0-gpu

# Upstride bash welcome screen 
COPY dockerfiles/bash.bashrc /etc/bash.bashrc

# install the python package in the docker
COPY src/python/ /opt/upstride
RUN cd /opt/upstride && \
    ls upstride/type_generic/_upstride.so && \
    pip install . && \
    cd / && \
    rm -r /opt/upstride
