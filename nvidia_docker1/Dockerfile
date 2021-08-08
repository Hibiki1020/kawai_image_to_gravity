FROM nvidia/cuda:10.0-devel-ubuntu18.04

########## nvidia-docker1 hooks ##########
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
########## basis ##########
RUN apt-get update && apt-get install -y \
	vim \
	wget \
	unzip \
	git \
	build-essential
######### Python ##########
RUN apt-get update &&\
	apt-get install -y \
		python3.6 \
		python3-pip &&\
	pip3 install \
		tqdm \
		matplotlib \
		tensorboardX \
		torch==1.0.1 torchvision==0.2.2
		# torch==1.2.0 torchvision==0.4.0
######### NO cache ##########
ARG CACHEBUST=1
######### My package ##########
RUN cd /home &&\
	git clone https://github.com/ozakiryota/image_to_gravity
######### initial position ##########
WORKDIR /home/image_to_gravity
