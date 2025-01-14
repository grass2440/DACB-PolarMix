# DACB-PolarMix

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
- Please follow the instructions from [here](http://www.semantic-kitti.org) to download the SemanticKITTI dataset (both KITTI Odometry dataset and SemanticKITTI labels) and extract all the files in the `sequences` folder to `/dataset/semantic-kitti`. You shall see 22 folders 00, 01, …, 21; each with subfolders named `velodyne` and `labels`.  
- Change the data root path in configs/semantic_kitti/default.yaml


### Training

#### SemanticKITTI

We release the training code for SPVCNN and MinkowskiNet with PolarMix. You may run the following code to train the model from scratch. 

SPVCNN:
```bash
python train.py configs/semantic_kitti/spvcnn/cr0p5.yaml --run-dir runs/semantickitti/spvcnn_polarmix --distributed False
```
MinkowskiNet:
```bash
python train.py configs/semantic_kitti/minkunet/cr0p5.yaml --run-dir run/semantickitti/minkunet_polarmix --distributed False
```

- Note we only used one 2080Ti for training and testing. Training from scratch takes around 1.5 days. You may try larger batch size or distributed learning for faster training.

### Testing Models

You can run the following command to test the performance of SPVCNN/MinkUNet models with PolarMix.

```bash
torchpack dist-run -np 1 python test.py --name ./runs/semantickitti/spvcnn_polarmix
torchpack dist-run -np 1 python test.py --name ./runs/semantickitti/minkunet_polarmix
```


## Thanks
We thank the opensource project [TorchSparse](https://github.com/mit-han-lab/torchsparse) and [SPVNAS](https://github.com/mit-han-lab/spvnas).
```
@article{xiao2022polarmix,
  title={PolarMix: A General Data Augmentation Technique for LiDAR Point Clouds},
  author={Xiao, Aoran and Huang, Jiaxing and Guan, Dayan and Cui, Kaiwen and Lu, Shijian and Shao, Ling},
  journal={arXiv preprint arXiv:2208.00223},
  year={2022}
}
