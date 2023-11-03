# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

"""please config ROOT_dir and user when u first using"""
C.abs_dir = osp.dirname(osp.realpath(__file__))
C.parent_dir_of_curpath=osp.abspath(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.project_dir=os.path.abspath(osp.dirname(os.path.abspath(C.abs_dir)))
C.checkpoint=os.path.join(C.project_dir,'checkpoint')

C.motion = edict()
C.motion.input_length =50
C.motion.input_length_dct = 50
C.motion.target_length_train = 10
C.motion.target_length_eval = 25
C.motion.dim = 66

C.data_aug = True
C.deriv_input = True
C.deriv_output = True
C.use_relative_loss = True
C.use_identity_loss = True

""" Model Config"""
## Network
C.pre_dct = False
C.post_dct = False
## Motion Network mlp
dim_ = 66
C.motion_mlp = edict()
C.motion_mlp.hidden_dim = dim_
C.motion_mlp.seq_len = C.motion.input_length_dct
C.motion_mlp.num_layers = 48
C.motion_mlp.with_normalization = True
C.motion_mlp.spatial_fc_only = False
C.motion_mlp.norm_axis = 'spatial'

"""Train Config"""
C.batch_size =256
C.num_workers = 0

C.cos_lr_max=1e-5
C.cos_lr_min=5e-8
C.cos_lr_total_iters=40000

C.weight_decay = 1e-4
C.model_pth = None

"""Eval Config"""
C.shift_step = 1

"""Display Config"""
C.print_every = 100
C.save_every = 100



"adjacent matrix"
C.h36m = edict()
C.h36m.node_n=22
C.h36m.joint_level_edge = [(0, 1), (1, 2), (2, 3), (0, 8), (8, 4), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (9, 17),
                    (17, 18),(18, 19), (19, 20), (19, 21), (9, 12), (12, 13), (13, 14), (14, 15), (14, 16)]
C.h36m.part_level_edge = [(0, 1), (1, 2, 3), (0, 8), (8, 4), (4, 5), (5, 6, 7), (8, 9), (9, 10, 11), (9, 17), (17, 18),
                   (18, 19), (19, 20, 21), (9, 12), (12, 13), (13, 14), (14, 15, 16)]
C.h36m.pose_level_edge = [(0, 1, 2, 3), (0, 8), (8, 4), (4, 5, 6, 7), (8, 9, 10, 11), (9, 17), (17, 18, 19, 20, 21),
                   (9, 12), (12, 13, 14, 15, 16)]


C.CMU_Mocap = edict()
C.CMU_Mocap.node_n = 25
C.CMU_Mocap.joint_level_edge = [(0, 1), (1, 2), (2, 3), (0, 8), (4, 5), (5, 6), (6, 7), (4, 8), (8, 9), (9, 10),(10, 11), (11, 12),  (9, 13),
                                (13, 14),(14, 15), (15, 18), (15, 16), (16, 17), (9, 19),(19, 20), (20, 21), (21, 24),(21, 22), (22, 23)]
C.CMU_Mocap.part_level_edge = [(0, 1), (1, 2, 3), (0, 8), (4, 5), (5, 6, 7), (4, 8), (8, 9), (9, 10), (10, 11),
                               (11, 12), (9, 13), (13, 14), (14, 15), (15, 18, 16, 17), (9, 19), (19, 20), (20, 21), (21, 24, 22, 23)]
C.CMU_Mocap.pose_level_edge = [(0, 1, 2, 3), (0, 8), (4, 5, 6, 7), (4, 8), (8, 9, 10, 11, 12), (9, 13),
                               (13, 14, 15, 16, 17, 18), (9, 19), (19, 20, 21, 24, 22, 23)]


C._3DPW = edict()
C._3DPW.node_n = 23
C._3DPW.joint_level_edge = [(0, 3), (3, 6), (6, 9), (0, 2), (1, 4), (4, 7), (7, 10), (1, 2), (2, 5), (5, 8),(8, 11), (11, 14),
                            (8, 13), (13, 16), (16, 18), (18, 20), (20, 22), (8, 12), (12, 15), (15, 17), (17, 19), (19, 21)]
C._3DPW.part_level_edge = [(0, 3), (3, 6, 9), (0, 2), (1, 4), (4, 7, 10), (1, 2), (2, 5), (5, 8), (8, 11, 14),(8, 13),
                           (13, 16), (16, 18), (18, 20), (20, 22), (8, 12), (12, 15), (15, 17),(17, 19), (19, 21)]
C._3DPW.pose_level_edge = [(0, 3, 6, 9), (0, 2), (1, 4, 7, 10), (1, 2), (2, 5), (5, 8), (8, 11, 14),
                           (8, 13), (13, 16), (16, 18, 20, 22), (8, 12), (12, 15), (15, 17, 19, 21)]


C._curUsed = edict()
C._curUsed.node_n = 0
C._curUsed.joint_level_edge = []
C._curUsed.part_level_edge = []
C._curUsed.pose_level_edge = []

#这个part_level_edge pose_level_edeg这儿有问题！！！


if __name__ == '__main__':
    print(config.motion_mlp)
