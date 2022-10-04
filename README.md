How to run these Apps
=====================

## Environments
### Torch
Setup: Make sure you have a conda installation. Create a new environment for
casio by following these instructions:

**MAKE SURE TO FOLLOW THESE IN ORDER**

```bash
$ conda create -n casio-torch python=3.9
# Press enter to accept

$ conda activate casio-torch

# Install requirements
$ pip install -r requirements.txt
```

**NOTE: mmcv-full will take a WHILE to install. This is a one-time thing.**


### TensorFlow
For tensorflow, use the utils/tf1-docker.sh script to launch a docker container.

**NOTE: YOU WILL END UP SOURCE'ING env.sh TWICE!**

## Swin Transformer (Torch)
```bash
$ conda activate casio-torch
$ source env.sh
==============================================
REMEMBER: RUN THIS INSIDE THE DOCKER CONTAINER
FOR TENSORFLOW v1 APPLICATIONS!

MIKE WILL NOT ANSWER THIS QUESTION!
==============================================
What platform is this? (cpu, p100, v100, a100): <type of gpu>
What gpu should we use? (cuda:0, cuda:1, ...): cuda:<N>
Path to CASIO: /nobackup/medavies/casio
Platform: <type of gpu>
Device: cuda:<N>

$ cd Swin-Transformer
$ ./runall.sh
```

## QD Track (Torch)
**DOWNLOAD DATA AND RUN SETUP FIRST**
```bash
$ conda activate casio-torch
$ cd qdtrack/
$ wget cs.wisc.edu/~davies/qdtrack-data.tar.xz
$ tar xvf qdtrack-data.tar.xz
$ python setup.py develop
$ cd ..
```

**Running qdtrack:**
```bash
$ conda activate casio-torch
$ source env.sh
...
$ cd qdtrack
$ ./runall.sh
```

## PINN (Tensorflow)
```
$ source env.sh
$ ./utils/tf1-docker.sh
$ cd /work
$ source env.sh
...
$ pip install pydoe
$ cd PINNs
$ ./runall.sh
```

## tabnet (Tensorflow)
```
$ source env.sh
$ ./utils/tf1-docker.sh
$ cd /work
$ source env.sh
...
$ cd tabnet
$ ./runall.sh
```

## meshgraphnets (Tensorflow)
**DOWNLOAD DATA AND RUN SETUP FIRST**
```bash
$ cd meshgraphnets
$ wget cs.wisc.edu/~davies/mgn-datasets.tar.xz
$ tar xvf mgn-datasets.tar.xz
$ python setup.py develop
$ cd ..
```

**Running meshgraphnets:**
```
$ source env.sh
$ ./utils/tf1-docker.sh
$ cd /work
$ source env.sh
...
$ cd meshgraphnets
$ pip install -r requirements
$ cd /work
$ ./meshgraphnets/runall.sh
```
