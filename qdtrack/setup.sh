#!/bin/bash

python -m pip install -r requirements.txt
python -m pip install git+git://github.com/bdd100k/bdd100k.git
python -m pip install -e .

pip install "opencv-python-headless<4.3"
pip install mmcv-full mmdet
pip install seaborn
