# The Phoenix Project !!! (tf version)

## Compile in docker
Docker is life. Thanks to docker we can create a compilation environment with a simple command.

2 dockerfiles are provided for compilation, `dockerfiles/dev.dockerfile` for the cpu version and `dockerfiles/dev-gpu.dockerfile` for the gpu version.
Both need the folder `custom_gcc` to exist with the file `gcc-8.4_8.4-1_amd64.deb` inside.

- To create the docker image (for gpu in this example) run :
```bash
docker build -t phoenix_tf:dev-gpu -f dockerfiles/dev-gpu.dockerfile .
```

- Now you can create a container :
```bash
docker run -it --rm --gpus all \
	-v $(pwd):/opt/src \
	phoenix:dev
```

this command mount as volume the source code in dir /opt/src. Now you can compile the project :
```bash
cd /opt/src
export WITH_CUDNN=yes
make 
```
