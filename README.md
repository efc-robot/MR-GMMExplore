# MR-GMMExplore
<!-- 该项目是多机 GMM Submap 探索的工作 -->
This is the open-source project **MR-GMMExplore**, a communication efficient **M**ulti-**R**obot **GMM** submap-based **Exploration** system. It is an extension to our previous work **MR-GMMapping**.

The related paper "MR-GMMExplore: Multi-Robot Exploration System in Unknown
Environments based on Gaussian Mixture Model" is published in the 2022 IEEE
International Conference on Robotics and Biomimetics (ROBIO).

## Platform
- Multi-robots with NVIDIA Jetson TX2, Intel RealSense T265, and depth camera D435i
- ubuntu 18.04 (bash)
- ROS melodic
- python2
 
## Dependency

<!-- 项目基于 ROS 和 Python2
需要安装的库主要有 -->
### Pytorch for Place Recognition
```
pip install torch torchvision
```

Pre-trained model is available [here](https://cloud.tsinghua.edu.cn/d/b37383f4c5e145c2b92a/)
Then change the regarding path of `model13.ckpt` in `MapBuilderNode.py`.

### Python-PCL

Useful reference [here](https://python-pcl-fork.readthedocs.io/en/rc_patches4/install.html#install-python-pcl).

### GTSAM

```
git clone https://bitbucket.org/gtborg/gtsam.git
mkdir build
cd build
cmake .. -DGTSAM_INSTALL_CYTHON_TOOLBOX=on
make check
make install
cd cython
sudo python setup.py install
```

### Other dependency 
```
sudo apt install ros-melodic-tf2-ros
pip install autolab_core
pip install sklearn
pip install future
pip install transforms3d
pip install pygraph
```

For ROS libraries, you can use ```apt install ros-melodic-(missing library)``` to install the missing libraries.

## Usage Example

### Installation
download the repo and then put it in your ros package
```
catkin_make
source <repo_path>/devel/setup.bash
```

### Multi-robot Exploration
```
roslaunch gmm_map_python visualgmm_realsence_2robot.launch
roslaunch gmm_map_python exploration_2robot.launch
```