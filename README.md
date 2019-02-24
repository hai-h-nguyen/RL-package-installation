# Install Gym, Mujoco, Mujoco-py, CUDA, CUdNN in Ubuntu 16.04, Python v3.5.2

My system: Alienware Aurora 7, Ubuntu 16.04, Python v3.5.2, GTX 1080 TI, CUDA 9.0, Cudnn 7.0.5, TensorflowGPU 1.5.0. Mujoco, Gym, Mujoco-py latest version

1. Install CUDA
- Prepare
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake git unzip pkg-config
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install libhdf5-serial-dev graphviz
sudo apt-get install libopenblas-dev libatlas-base-dev gfortran
sudo apt-get install python-tk python3-tk python-imaging-tk
sudo apt-get install python2.7-dev python3-dev

sudo apt-get install linux-image-generic linux-image-extra-virtual
sudo apt-get install linux-source linux-headers-generic
```
- Go to Software Update / Additional Drivers and change to use NVIDIA binary instead of X Server
- Go to CUDA website, download the appropriate version and install 
```
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```
- After finish installing, update .bashrc file, replace cuda-8.0 by your correct version [combination that works: CUDA 8.0 + Tensorflow GPU 1.4.1; CUDA 9.0 + Tensorflow GPU 1.5.0]

#NVIDIA CUDA Toolkit
```
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/
```

- Check installation (have to restart or log-out first), you should see Pass result
```
source ~/.bashrc
cd /usr/local/cuda-8.0/samples/1_Utilities/deviceQuery
sudo make
/deviceQuery
deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 8.0, CUDA Runtime Version = 8.0, NumDevs = 1, Device0 = Tesla K80
Result = PASS
```
- Download cuDNN
Go to cuDNN website and download the .tar file
```
cd ~
tar -zxf cudnn-8.0-linux-x64-v6.0.tgz
cd cuda
sudo cp -P lib64/* /usr/local/cuda/lib64/
sudo cp -P include/* /usr/local/cuda/include/
cd ~
```
2. Install Keras
```
pip3 install numpy
pip3 install scipy matplotlib pillow
pip3 install imutils h5py requests progressbar2
pip3 install scikit-learn scikit-image
pip3 install keras

pip3 install tensorflow-gpu (Maybe need to specify correct version to your system here, mine is 1.5.0 to work with CUDA 9.0; 1.4.1 for CUDA 8.0)
```
- Check for importing OK in Python for tensorflow and keras
```
$ python3
>>> import tensorflow
>>>
```
```
pip3 install keras
$ python3
>>> import keras
>>>
```
3. Install Mujoco: Can ask for a trial license or use your uni email to get 1-year license
```
#download mjpro150.zip
wget -O mjpro150.zip https://www.roboti.us/download/mjpro150_linux.zip
mkdir ~/.mujoco
unzip mjpro150.zip -d ~/.mujoco/mjpro150
#put the license file into ~/.mujoco & ~/.mujoco/mjpro150/bin
cp mjkey.txt ~/.mujoco
cp mjkey.txt ~/.mujoco/mjpro150/bin
sudo apt install libglew-dev
```

- Modify bashrc file and source (important)
```
#CUDA
export PATH=/usr/local/cuda-8.0/bin:~/.mujoco/mjpro150/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64

# MUJOCO PRO
export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin:/usr/lib/nvidia-384:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so
```

- Validate installation
```
cd ~/.mujoco/mjpro150/bin
./simulate ../model/humanoid.xml
```

4. Install Mujoco-py
```
#install dependencies
sudo apt-get update -q DEBIAN_FRONTEND=noninteractive 
sudo apt-get install -y \
curl \
git \
libgl1-mesa-dev \
libgl1-mesa-glx \
libosmesa6-dev \
python3-pip \
python3-numpy \
python3-scipy \
unzip \
vim \
wget \
xpra \
xserver-xorg-dev
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
#install patchelf
sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf
sudo chmod +x /usr/local/bin/patchelf
#download mujoco-py
git clone https://github.com/openai/mujoco-py.git
#install mujoco-py
cd mujoco-py
sudo python3 setup.py install
```
- Test installation
```
python3
import mujoco_py
from os.path import dirname
model = mujoco_py.load_model_from_path(dirname(dirname(mujoco_py.__file__))  +"/xmls/claw.xml")
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
xxxx

sim.step()
print(sim.data.qpos)
xxxx
```
- Create alias in bashrc file
```
alias sudo='sudo env PATH=$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH LD_PRELOAD=$LD_PRELOAD'
```

- Test for visualization
```
cd mujoco-py/examples
python3 body_interaction.py
```

5. Install Gym
```
#download source code
git clone https://github.com/openai/gym.git
cd gym
#install requirements
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
#install gym (full-version)
sudo pip3 install -e .
```

References:
- https://www.pyimagesearch.com/2017/09/27/setting-up-ubuntu-16-04-cuda-gpu-for-deep-learning-with-python/
- https://zhuanlan.zhihu.com/p/34195184
