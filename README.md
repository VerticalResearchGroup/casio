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

## Swin Transformer
```bash
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