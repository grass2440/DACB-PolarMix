import os
import os.path
import random

import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from torchpack.utils.logging import logger

from core.datasets.utils import polarmix

__all__ = ['SemanticKITTI_PolarMix']

LABEL_DICT = {
    0: "unlabeled",
    4: "1 person",
    5: "2+ person",
    6: "rider",
    7: "car",
    8: "trunk",
    9: "plants",
    10: "traffic sign 1", # standing sign
    11: "traffic sign 2", # hanging sign
    12: "traffic sign 3", # high/big hanging sign
    13: "pole",
    14: "trashcan",
    15: "building",
    16: "cone/stone",
    17: "fence",
    21: "bike",
    22: "ground"} # class definition



label_map_dict = {
    0: 23, #"unlabeled"
    4: 0, #"1 person"
    5: 0, #"2+ person"
    6: 1, #"rider"
    7: 2, #"car"
    8: 3, #"trunk"
    9: 4, #"plants"
    10: 5, #"traffic sign 1" # standing sign
    11: 5, #"traffic sign 2" # hanging sign
    12: 5, #"traffic sign 3" # high/big hanging sign
    13: 6, #"pole"
    14: 7, #"trashcan"
    15: 8, #"building"
    16: 9, #"cone/stone"
    17: 10, #"fence"
    21: 11, #"bike"
    22: 12, #"ground"
} # class definition

#self.label_map = [0,1,1,2,3,4,5,6,6,6,7,8,9,10,11,12,13]
'''
kept_labels = [
    'unlabeled', 'person', 'rider', 'car', 'trunk',
    'plants', 'traffic sign', 'pole', 'trashcan',
    'building', 'cone/stone', 'fence', 'bike', 'ground'
]
'''


#instance_classes0 = [0, 5, 6, 7, 9]
instance_classes0 = [0, 1, 2, 5, 6, 7, 9, 11]

#Omega0 = np.random.random(8) * 2 * np.pi
Omega0 = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]

class SemanticKITTI_PolarMix(dict):

    def __init__(self, root, voxel_size, num_points, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)

        #logger.info("SemanticKITTI with PolarMix\n")
        logger.info("SemanticPOSS with PolarMix\n")

        if submit_to_server:
            super().__init__({
                'train':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='train',
                                          submit=True),
                'test':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='test')
            })
        else:
            super().__init__({
                'train':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          polarcutmix=True,
                                          sample_stride=1,
                                          split='train',
                                          google_mode=google_mode),
                'test':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=sample_stride,
                                          split='val')
            })


class SemanticKITTIInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 split,
                 polarcutmix=False,
                 sample_stride=1,
                 submit=False,
                 google_mode=True):
        if submit:
            trainval = True
        else:
            trainval = False
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.polarcutmix = polarcutmix
        self.sample_stride = sample_stride
        self.google_mode = google_mode
        self.seqs = []
        if split == 'train':
            #self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
            self.seqs = ['00', '01', '02', '04', '05']
            if self.google_mode or trainval:
                self.seqs.append('03')
        elif self.split == 'val':
            self.seqs = ['03']
        #elif self.split == 'test':
            #self.seqs = [
               # '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
            #]

        self.files = []
        for seq in self.seqs:
            seq_files = sorted(
                os.listdir(os.path.join(self.root, seq, 'velodyne')))
            seq_files = [
                os.path.join(self.root, seq, 'velodyne', x) for x in seq_files
            ]
            self.files.extend(seq_files)

        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]
            
        
        self.label_map = np.zeros(23)
        for label_id, label_name in label_map_dict.items():
            self.label_map[label_id] = label_name
        
        self.angle = 0.0
        
    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

    def read_lidar_scan(self, index):
        with open(self.files[index], 'rb') as b:
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        label_file = self.files[index].replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(block_.shape[0]).astype(np.int32)
        labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)
        return block_, labels_

    def __getitem__(self, index):
        block_, labels_ = self.read_lidar_scan(index)

        if self.split == 'train':
            # read another lidar scan
            index_2 = np.random.randint(len(self.files))
            pts2, labels2 = self.read_lidar_scan(index_2)
           
           # polarmix
            alpha = (np.random.random() - 1) * np.pi
            beta = alpha + np.pi
            block_, labels_ = polarmix(block_, labels_, pts2, labels2,
                                      alpha=alpha, beta=beta,
                                      instance_classes0 =instance_classes0,
                                      Omega0=Omega0
                                      )
            
        block = np.zeros_like(block_)

        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])
            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)

        block[:, 3] = block_[:, 3]
        pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        feat_ = block

        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)

        if 'train' in self.split:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.files[index]
        }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
