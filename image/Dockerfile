# Original base image is producing the following error:
# RuntimeError: cuda runtime error (11) : invalid argument at /pytorch/aten/src/THC/THCGeneral.cpp:405
#FROM nvidia/cuda:10.1-base

FROM nvidia/cuda:9.2-base

ADD pytorch-CycleGAN-and-pix2pix /workspace/pytorch-CycleGAN-and-pix2pix
RUN apt update && apt install -y wget unzip curl bzip2 git vim ffmpeg libsm6 libxext6
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install -y pip pytorch torchvision -c pytorch 

RUN cd /workspace/pytorch-CycleGAN-and-pix2pix && pip install -r requirements.txt

WORKDIR /workspace/pytorch-CycleGAN-and-pix2pix
