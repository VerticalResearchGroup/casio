#To run this network follow the following instructions. The nsys/ncu profiling is done on the ROI during the 20th iteration. The training loop breaks after that.
#You can enable printing on DNN forward/backward pass times whenever needed by uncommenting the DNN print statements. 
# The network's core DNN is in FP16 while the surrounding logic is in FP32 which is image rendering/batch normalization.

#Perform all the below directions within the nerf directory after untaring.

#Command to enable docker service.
systemctl --user start docker.service

#Command to start the docker service.
docker run --gpus all -it --rm -v $PWD:/work nvcr.io/nvidia/tensorflow:22.02-tf1-py3 bash

#Command to install the needed packages
pip install imageio imageio-ffmpeg configargparse

#Run this to download the dataset
bash download_example_data.sh

#Use this command to run the training loop
python run_nerf.py --config config_fern.txt

#The bkp_fp32 contains the pure fp32 implementation, bkp_fp16 contains the pure fp16 implementation, bkp_mlpcore_fp16 contains the implementation where the core DNN is alone in fp16
#Use appropriate nsys/ncu commands to do do things as per your requirement.
