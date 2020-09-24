# Project-Template for IW276 Autonome Systeme Labor

Short introduction to project assigment.

<p align="center">
  Screenshot / GIF <br />
  Link to Demo Video
</p>

> This work was done by Xiahong C., Nadine V. and Melanie W. during the IW276 Autonome Systeme Labor at the Karlsruhe University of Applied Sciences (Hochschule Karlsruhe - Technik und Wirtschaft) in WS 2020 / 2021. 

## Table of Contents

* [Requirements](#requirements)
* [Prerequisites](#prerequisites)
* [Pre-trained model](#pre-trained-model)
* [Running](#running)
* [Docker](#Docker)
* [Acknowledgments](#acknowledgments)

## Requirements
* Python 3.6 (or above)
* OpenCV 4.1 (or above)
* Jetson Nano
* Jetpack 4.4
* trt_pose
* torch2trt

## Prerequisites
You can either install the repository directly or install it via Docker (see point '[Docker](#Docker)').

1. Install dependencies:
    ```
    sudo apt-get update
    sudo apt-get install -y build-essential libssl-dev libffi-dev python-dev git python3-matplotlib libopencv-python
    ```
2. Install trt_pose:
    ```
    git clone https://github.com/NVIDIA-AI-IOT/trt_pose
    cd trt_pose && python3 setup.py install
    ```
3. Install torch2trt:
    ```
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    cd torch2trt && python3 setup.py install
    ```
4. Clone IW276WS20-P11 project:
    ```
    git clone https://github.com/IW276/IW276WS20-P11.git
    cd IW276WS20-P11 && pip3 install -r requirements.txt
    ``` 
5. Make sure you have some .mp4 video files in the ~/Videos folder on your system
  > The demo works best with square videos.

    
## Pre-trained models

Pre-trained models are available at ```pretrained-models```.
* ``resnet18_crowdpose_224x224_epoch_129.pth`` was trained using the CrowdPose dataset and is based on the resnet model.
> If you want to continue training the model, see point '[Training](#Training)'.

* ``resnet18_baseline_att_224x224_A_epoch_249.pth`` and ``densenet121_baseline_att_256x256_B_epoch_160.pth`` were pre-trained on the MSCOCO dataset (source: trt_pose).

## Running

To run the demo, switch to the src folder and pass a video file name after --video and the absolute path of the directory in which the video can be found after --path. The processed video will also be saved here:
```
python3 demo.py --video <video.mp4> --path </videos/>
```
## Docker

To run the demo in a Docker container, follow these steps:
> We assume you have already installed Docker on your system.

1. Build the docker container via ```docker_build.sh```. The container will be called ```p11_image```.
2. To start the demo directly in the container, use this command:
    ```
    sudo docker run --runtime nvidia -v ~/Videos/:/videos/ p11_image /bin/bash -c 'cd IW276WS20-P11/src && python3 demo.py --path /videos/ --video <video.mp4>'
    ```
   > Please replace the video file name in the command.                                                                                                                                                                                                                                            
   > Add -d to the command to run the container in detached mode.                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                     
3. To start the Docker container with an interactive terminal, follow these steps:
    * Start the Docker container via
        ```
        sudo docker run --runtime nvidia -d -v ~/Videos/:/videos/ p11_image sleep infinity
        ```
   * To find out the container ID, use
        ```
        sudo docker ps 
        ```
   * To interact with the terminal into the Docker container, use
        ```
        sudo docker exec -it <CONTAINER_ID> /bin/bash 
        ```
     > You can switch to the host system via 
        ```
        exit
        ```                                                                                                                                                                                                                                                                                          
   * To stop the Docker container, use
        ```
        sudo docker stop <CONTAINER_ID>
        ```

## Training

1. Execute the preprocess_coco_person.py in the train folder to adjust the keypoint IDs and links:
    ```
    python3 preprocess_coco_person.py
    ```
2. Check your dataset has exactly the following 14 keypoints:
    * Left/Right Shoulder 
    * Left/Right Elbow
    * Left/Right Wrist
    * Left/Right Hip
    * Left/Right Knee
    * Left/Right Ankle
    * Head
    * Neck

## Acknowledgments

This repo is based on
* [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose)
* [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
* [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)

Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.
