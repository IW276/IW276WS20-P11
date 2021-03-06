# IW276WS20P11: 2D Multi Person Pose Estimation (Jetson Nano)

In the context of Smart Cities, it is important to be able to estimate and recognize situations in urban space.
Therefore the video-based action recognition in real time is used in Person Pose Estimation. This project was developed for the Jetson Nano.

Based on the results of [IW276SS20P1](https://github.com/IW276/IW276SS20-P1) we were able to optimize the existing pipeline to process videos in real time (> 15 fps). We used the [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) data set to train a model in [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) which can estimate the 2D poses of several people from a video stream.

**Result image using the pipeline from [IW276SS20P1](https://github.com/IW276/IW276SS20-P1)** <br />
![Result image of old pipeline](/result-images/dance_demo_old.jpg)<br />

**Result image using the optimized pipeline**<br />
![Result image of optimized pipeline](/result-images/dance_demo_optimized.jpg)<br />
  

[Link to Demo Video]()


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
  > You can download example videos [here](https://drive.google.com/drive/folders/1V--ryc-o-DVLaBRe7ET7AeKvQKV8SVm8).     

    
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
3. Adjust the config file resnet18CrowdPose.json with your own location of the dataset.
4. Start your training with the skript train.py.
## Acknowledgments

This repo is based on
* [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose)
* [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
* [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)
* [IW276SS20-P1](https://github.com/IW276/IW276SS20-P1)

The example videos are made from
* [https://www.youtube.com/watch?v=4Ch3MWQG3CE](https://www.youtube.com/watch?v=4Ch3MWQG3CE)
* [https://www.youtube.com/watch?v=SvldnZ6qMGU&t=54s](https://www.youtube.com/watch?v=SvldnZ6qMGU&t=54s)
* [https://www.youtube.com/watch?v=EErq1Km6HZE](https://www.youtube.com/watch?v=EErq1Km6HZE)
* [https://www.pexels.com/video/a-busy-street-on-a-sunny-day-1625973/](https://www.pexels.com/video/a-busy-street-on-a-sunny-day-1625973/)
* [https://www.pexels.com/video/people-giving-high-fives-1149521/](https://www.pexels.com/video/people-giving-high-fives-1149521/)
* [https://www.pexels.com/video/a-crowd-of-people-gathered-in-the-city-street-carrying-placard-and-flags-in-protest-3105293/](https://www.pexels.com/video/a-crowd-of-people-gathered-in-the-city-street-carrying-placard-and-flags-in-protest-3105293/)
* [https://www.pexels.com/video/showing-the-new-workplace-4435249/](https://www.pexels.com/video/showing-the-new-workplace-4435249/)
* [https://www.pexels.com/video/two-women-crossing-the-street-4873842/](https://www.pexels.com/video/two-women-crossing-the-street-4873842/)
* [https://pixabay.com/videos/people-alley-street-ukraine-bike-39836/](https://pixabay.com/videos/people-alley-street-ukraine-bike-39836/)
* [https://pixabay.com/videos/people-commerce-shop-busy-mall-6387/](https://pixabay.com/videos/people-commerce-shop-busy-mall-6387/)


Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.
