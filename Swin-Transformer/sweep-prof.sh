#!/bin/bash

MODE=prof BS=1 NI=30 python main.py --cfg configs/swinv2/swinv2_base_patch4_window16_256.yaml
MODE=prof BS=1 NI=30 python main.py --cfg configs/swinv2/swinv2_base_patch4_window12_192_22k.yaml
MODE=prof BS=1 NI=30 python main.py --cfg configs/swinv2/swinv2_large_patch4_window12_192_22k.yaml
MODE=prof BS=1 NI=30 python main.py --cfg configs/swinv2/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.yaml

MODE=prof BS=64 NI=30 python main.py --cfg configs/swinv2/swinv2_base_patch4_window16_256.yaml
MODE=prof BS=32 NI=30 python main.py --cfg configs/swinv2/swinv2_base_patch4_window12_192_22k.yaml
MODE=prof BS=16 NI=30 python main.py --cfg configs/swinv2/swinv2_large_patch4_window12_192_22k.yaml
MODE=prof BS=2 NI=30 python main.py --cfg configs/swinv2/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.yaml



