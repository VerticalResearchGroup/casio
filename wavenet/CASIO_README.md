Wavenet Casio Readme

docker and data
---------------

docker:  nvcr.io/nvidia/tensorflow:22.08-tf1-py3 
necessary dataset: /nobackup.1/karu/datasets/wavenet-tiny
(218MB reduced dataset, that has 416 files, so at batch-size 1 we can do 416 steps of fwd/bwd)
full dataset: /nobackup.1/karu/datasets/wavenet

Cmds to put in Dockerfile
-------------------------

pip install librosa
apt-get update
apt-get install libsndfile1-dev


Executing program
-----------------
assume following bind-mounts are provided to docker: -v casio-root:/code -v /nobackup.1/karu/datasets/wavenet-tiny:/data
for example, if your casio git repo is cloned at: /p/vertical/afs-huge3/casio
docker run --gpus all -it  --ulimit stack=67108864 --shm-size=4.53gb --rm -v $PWD:/data -v /p/vertical/afs-huge3/casio:/code nvcr.io/nvidia/tensorflow:22.08-tf1-py3

$ cd /code/wavenet
$ python train.py --data_dir=/data/wavenet-tiny --silence_threshold 0

