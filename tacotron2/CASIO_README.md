Tacotron Casio Readme

docker and data
---------------

docker:  nvcr.io/nvidia/tensorflow:22.08-tf1-py3 
necessary dataset: /nobackup.1/karu/datasets/tacotron2/LJSpeech-1.1

Cmds to put in Dockerfile
-------------------------

pip install torch
pip install librosa
apt-get update
apt-get install libsndfile1-dev
pip install unidecode
pip install inflect
cd /work/tacotron2/apex
pip install -v --disable-pip-version-check --no-cache-dir ./apt-get install libsndfile1-dev

Executing program
-----------------
assume following bind-mounts are provided to docker: -v casio-root:/code -v /nobackup.1/karu:/data

filelists/*.txt expect this following path to exist and the wav files there: 
/work/datasets/tacotron2/LJSpeech-1.1/wavs/.*.wav

for example, if your casio git repo is cloned at: /p/vertical/afs-huge3/casio
docker run --gpus all -it  --ulimit stack=67108864 --shm-size=4.53gb --rm -v /nobackup.1./karu:/data -v /p/vertical/afs-huge3/casio:/code nvcr.io/nvidia/tensorflow:22.08-tf1-py3

$ cd /code/tacotron2
$ python train.py --output_directory=outputs --log_directory=logs

