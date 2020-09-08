FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3
RUN echo "Build a Docker container for the trt_pose project based on L4T Pytorch"
RUN nvcc --version

# needed for accessing jetpack.  This is for 4.4
COPY  nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
COPY  jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

#Install all dependencies of the project
RUN apt-get update && \
        apt-get install -y git && \
        pip3 install -U \
                pip \
                setuptools \
                wheel \
                tqdm \
                cython \
                pycocotools
#       apt-get install python3-matplotlib &&
#RUN rm -rf ~/.cache/pip

WORKDIR /Autonome_Systeme_Labor

RUN git clone https://github.com/NVIDIA-AI-IOT/trt_pose
RUN cd trt_pose && python3 setup.py install && cd ..
RUN git clone https://github.com/IW276/IW276WS20-P11.git


#WORKDIR tasks/human_pose

#COPY densenet121_baseline_att_256x256_B_epoch_160.pth .
#COPY resnet18_baseline_att_224x224_A_epoch_249.pth .