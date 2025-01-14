# DACB-PolarMix
We propose a modified version of PolarMix's instance-level rotation and pasting method that dynamically adjusts the number of rotations and pastes based on the proportion of each instance's point cloud count relative to the total. This adaptive class-balancing approach ensures a more balanced distribution of instances across the entire dataset. We term our new algorithm Dynamic Adaptive Class-Balanced PolarMix (DACB-PolarMix). 
### Installation
#### System requirements
System requirements are as follows: CUDA 10.2, Python 3.8, and PyTorch 1.6.
```bash
conda create -n minknet python=3.8
conda activate minknet 
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

#### Install torchsparse 1.4.0.
```bash
sudo apt-get update
sudo apt-get install libsparsehash-dev
mkdir torchsparse
cd ./torchsparse
wget https://github.com/mit-han-lab/torchsparse/archive/refs/tags/v1.4.0.zip
unzip v1.4.0.zip
cd torchsparse-1.4.0
pip install -e
```
#### Install other packages.
```bash
pip install tqdm
pip install torchpack
pip install numba
pip install opencv-python
```


### Data Preparation

#### SemanticKITTI  
- Download the SemanticKITTI dataset from [here](http://www.semantic-kitti.org)
- Change the data root path in configs/semantic_kitti/default.yaml
#### SemanticPOSS
- Download the SemanticPOSS dataset from [here](http://www.poss.pku.edu.cn)
- Change the data root path in configs/semantic_kitti/default.yaml

### Training

#### SemanticKITTI

We release the training code for SPVCNN and MinkowskiNet with DACB-PolarMix. You may run the following code to train the model from scratch. 

SPVCNN:
```bash
python train.py configs/semantic_kitti/spvcnn/cr0p5.yaml --run-dir runs/semantickitti/spvcnn_polarmix --distributed False
```
MinkowskiNet:
```bash
python train.py configs/semantic_kitti/minkunet/cr0p5.yaml --run-dir run/semantickitti/minkunet_polarmix --distributed False
```

- Note If the training dataset is SemanticPOSS, you only need to Change the data root path in configs/semantic_kitti/default.yaml. Additionally, adjust the sequence values in core/datasets/semantic_kitti.py and core/datasets/semantic_kitti_polarmix.py.

### Testing Models

You can run the following command to test the performance of SPVCNN/MinkUNet models with PolarMix.

```bash
torchpack dist-run -np 1 python test.py --name ./runs/semantickitti/spvcnn_polarmix
torchpack dist-run -np 1 python test.py --name ./runs/semantickitti/minkunet_polarmix
```


## Thanks
Our modifications to the code are primarily in three files: core/datasets/semantic_kitti.py, core/datasets/semantic_kitti_polarmix.py, and core/datasets/utils.py.
We thank the original PolarMix works.
```
@article{xiao2022polarmix,
  title={PolarMix: A General Data Augmentation Technique for LiDAR Point Clouds},
  author={Xiao, Aoran and Huang, Jiaxing and Guan, Dayan and Cui, Kaiwen and Lu, Shijian and Shao, Ling},
  journal={arXiv preprint arXiv:2208.00223},
  year={2022}
}
