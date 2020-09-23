cp /etc/apt/sources.list.d/nvidia-l4t-apt-source.list .
cp /etc/apt/trusted.gpg.d/jetson-ota-public.asc .

sudo docker build . -t P11_image

rm nvidia-l4t-apt-source.list jetson-ota-public.asc