# Project-Template for IW276 Autonome Systeme Labor

Short introduction to project assigment.

<p align="center">
  Screenshot / GIF <br />
  Link to Demo Video
</p>

> This work was done by Xiahong C., Nadine V. and Melanie W. during the IW276 Autonome Systeme Labor at the Karlsruhe University of Applied Sciences (Hochschule Karlruhe - Technik und Wirtschaft) in WS 2020 / 2021. 

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

## Prerequisites
You can either install the repository directly or install it via Docker (see point '[Docker](#Docker)').
1. Install requirements:
    ```
    pip install -r requirements.txt
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
3. The demo can be found in ```src```.

## Pre-trained models

Pre-trained models are available at ```pretrained-models```.
* ``epoch129.pth`` was trained using the CrowdPose dataset and is based on the resnet model.
* ``resnet18_baseline_att_224x224_A_epoch_249.pth`` and ``densenet121_baseline_att_256x256_B_epoch_160.pth`` were pre-trained on the MSCOCO dataset (source: trt_pose).

## Running
To run the demo, pass a video file name and the path in which the video can be found. The processed video will also be saved here:
```
python3 demo.py --video video.mp4 --path /videos/
```
> The demo works best with square videos.

### Docker
1. Build the docker container via ```docker_build.sh```. The container will be called ```P11_image```.
2. Start the container, either via ```docker_run.sh``` or ```docker_run_interactive.sh```
    * ```docker_run.sh``` will use a video called 'video.mp4' from your 'Videos' folder and run the demo automatically.
    * ```docker_run_interactive.sh``` can be used to test your own videos.

## Acknowledgments

This repo is based on
* [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose)
* [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)

Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.
