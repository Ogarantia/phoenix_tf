#!/bin/bash

# Usage 1 (Run development docker):
# $ ./docker.sh dev

# Usage 2 (Run production docker):
# $ ./docker.sh 

docker run --gpus all -it --rm \
	-v $(pwd):/opt/upstride \
	upstride:$(cat VERSION)-px${1}-$(cat TF_VERSION)-gpu-x86_64
