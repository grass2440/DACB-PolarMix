# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Written by Aoran Xiao, 09:43 2022/03/05
# Wish for world peace!

import numpy as np


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

instance_classes0 = [0, 1, 2, 5, 6, 7, 9, 11]

def swap(pt1, pt2, start_angle, end_angle, label1, label2):
    # 计算每个点的水平角度
    yaw1 = -np.arctan2(pt1[:, 1], pt1[:, 0])
    yaw2 = -np.arctan2(pt2[:, 1], pt2[:, 0])

    # 选择扇区内的点
    idx1 = np.where((yaw1 > start_angle) & (yaw1 < end_angle))
    idx2 = np.where((yaw2 > start_angle) & (yaw2 < end_angle))

    # 交换
    pt1_out = np.delete(pt1, idx1, axis=0)
    pt1_out = np.concatenate((pt1_out, pt2[idx2]))
    pt2_out = np.delete(pt2, idx2, axis=0)
    pt2_out = np.concatenate((pt2_out, pt1[idx1]))

    label1_out = np.delete(label1, idx1)
    label1_out = np.concatenate((label1_out, label2[idx2]))
    label2_out = np.delete(label2, idx2)
    label2_out = np.concatenate((label2_out, label1[idx1]))
    assert pt1_out.shape[0] == label1_out.shape[0]
    assert pt2_out.shape[0] == label2_out.shape[0]

    return pt1_out, pt2_out, label1_out, label2_out

def rotate_copy(pts, labels, instance_classes, Omega):
    # 提取实例点
    pts_inst, labels_inst = [], []
    for s_class in instance_classes:
        pt_idx = np.where((labels == s_class))
        pts_inst.append(pts[pt_idx])
        labels_inst.append(labels[pt_idx])
    pts_inst = np.concatenate(pts_inst, axis=0)
    labels_inst = np.concatenate(labels_inst, axis=0)

    # 旋转复制
    pts_copy = [pts_inst]
    labels_copy = [labels_inst]
    for omega_j in Omega:
        rot_mat = np.array([[np.cos(omega_j), np.sin(omega_j), 0],
                            [-np.sin(omega_j), np.cos(omega_j), 0], [0, 0, 1]])
        new_pt = np.zeros_like(pts_inst)
        new_pt[:, :3] = np.dot(pts_inst[:, :3], rot_mat)
        new_pt[:, 3] = pts_inst[:, 3]
        pts_copy.append(new_pt)
        labels_copy.append(labels_inst)
    pts_copy = np.concatenate(pts_copy, axis=0)
    labels_copy = np.concatenate(labels_copy, axis=0)
    return pts_copy, labels_copy

def calculate_counts(labels, instance_classes):
    total_points = len(labels)
    counts = np.zeros(len(instance_classes))

    for label in labels:
        if label in label_map_dict:
            mapped_label = label_map_dict[label]
            if mapped_label in instance_classes:
                counts[instance_classes.index(mapped_label)] += 1

    proportions = counts / total_points
    #softmax_proportions = np.exp(proportions) / np.sum(np.exp(proportions))
    #return softmax_proportions
    normalized_proportions = proportions / proportions.sum()
    return normalized_proportions

def polarmix(pts1, labels1, pts2, labels2, alpha, beta, instance_classes0, Omega0):
    pts_out, labels_out = pts1, labels1
    # swapping
    if np.random.random() < 0.5:
        pts_out, _, labels_out, _ = swap(pts1, pts2, start_angle=alpha, end_angle=beta, label1=labels1, label2=labels2)

    normalized_proportions = calculate_counts(labels_out, instance_classes0)
    print("Proportions:", normalized_proportions)
    
    # 根据 softmax 值计算旋转粘贴次数
    max_count = 3
    min_count = 1
    
    #rotation_counts = max_count - (normalized_proportions * (max_count - min_count))
    rotation_counts = max_count - (normalized_proportions * 10)
    #random_offsets = np.random.randint(-1, 2, size=len(instance_classes0))
    rotation_counts = np.clip(rotation_counts, min_count, max_count).astype(int)
    #rotation_counts = np.ceil((1 - softmax_proportions) * (max_times - min_times) + min_times).astype(int)
    #rotation_counts = np.ceil((1 - proportions) * (max_times - min_times) + min_times).astype(int)
    print("rotation_counts:", rotation_counts)
    # 旋转和粘贴
    Omega0 = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]  # 生成随机旋转角度
    for i, s_class in enumerate(instance_classes0):
        count = rotation_counts[i]
        for _ in range(count):
            pts_copy, labels_copy = rotate_copy(pts2, labels2, [s_class], Omega0)
            pts_out = np.concatenate((pts_out, pts_copy), axis=0)
            labels_out = np.concatenate((labels_out, labels_copy), axis=0)

    return pts_out, labels_out
