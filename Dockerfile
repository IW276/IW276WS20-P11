FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3
RUN echo "Build a Docker container for the IW276WS20-P11 project based on L4T Pytorch"
RUN nvcc --version

# needed for accessing jetpack.  This is for 4.4
COPY  nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
COPY  jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc
COPY requirements.txt requirements.txt

#Install all dependencies of the project
RUN apt-get update && \
        apt-get install -y \
                build-essential \
                libssl-dev \
                libffi-dev \
                python-dev \
                git \
                python3-matplotlib \
                libopencv-python

RUN pip3 install -r requirements.txt
WORKDIR /Autonome_Systeme_Labor

RUN git clone https://github.com/NVIDIA-AI-IOT/trt_pose
RUN cd trt_pose && python3 setup.py install

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
RUN cd torch2trt && python3 setup.py install

RUN git clone https://github.com/IW276/IW276WS20-P11.git
RUN cd IW276WS20-P11/src && git pull origin master