
# pycoral with docker

To train and infer fashion mnist just run `docker compose up`
Be sure to attach a USB coral accelerator to the host device

# Run example with fMNIST

`docker run -v /dev:/dev -v $PWD/data:/home/data --privileged coral`

Or with [custom script](data/run.sh)

`docker run -v /dev:/dev -v $PWD/data:/home/data --privileged coral /home/data/run.sh`


# Run converter

`docker run -v $PWD/data:/home/data coral /home/convert.sh`
