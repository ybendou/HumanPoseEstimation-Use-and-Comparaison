#!/bin/sh
# Install and compile Caffe on NVIDIA Jetson TK1 Development Kit
sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install libprotobuf-dev protobuf-compiler gfortran \
libboost-dev cmake libleveldb-dev libsnappy-dev \
libboost-thread-dev libboost-system-dev \
libatlas-base-dev libhdf5-serial-dev libgflags-dev \
libgoogle-glog-dev liblmdb-dev -y

sudo usermod -a -G video $USER

# Git clone Caffe
sudo apt-get install -y git
git clone https://github.com/BVLC/caffe.git
cd caffe && git checkout dev
cp Makefile.config.example Makefile.config

make -j 4 all

make -j 4 runtest

build/tools/caffe time --model=models/bvlc_alexnet/deploy.prototxt --gpu=0