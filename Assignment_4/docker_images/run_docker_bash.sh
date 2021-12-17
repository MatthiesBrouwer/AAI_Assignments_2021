#!/bin/bash
xhost +
sudo docker run -v ~/Documents/Studie/AAI/AAI_Assignments_2021/Assignment_4/Workspace:/mnt -it  --rm --gpus=all --net=host --volume="$HOME/.Xauthority:/root/Xauthority:rw"  --env="DISPLAY" --runtime=nvidia me/tensorflowplot bin/bash
