from torch.utils.data import Dataset
import pickle as pkl
import numpy as np
from os import walk
import torch


class Datasets(Dataset):

    def __init__(self, config, split=0, data_aug=False):

        # path_to_data = opt.data_dir
        self.in_n = config.motion.input_length
        self.out_n = config.motion.target_length
        self.data_aug=data_aug
        if split == 1:
            their_input_n = 50
        else:
            their_input_n =   self.in_n

        self.out_n= config.motion.target_length
        seq_len = their_input_n + self.out_n
        path_to_data = "./datasets/3dpw"
        if split == 0:
            self.data_path = path_to_data + '/train/'
        elif split == 1:
            self.data_path = path_to_data + '/validation/'
        elif split == 2:
            self.data_path = path_to_data + '/test/'
        all_seqs = []
        files = []

        # load data
        for (dirpath, dirnames, filenames) in walk(self.data_path):
            files.extend(filenames)
        for f in files:
            with open(self.data_path + f, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                joint_pos = data['jointPositions']
                for i in range(len(joint_pos)):
                    seqs = joint_pos[i]
                    seqs = seqs - seqs[:, 0:3].repeat(24, axis=0).reshape(-1, 72)
                    n_frames = seqs.shape[0]
                    fs = np.arange(0, n_frames - seq_len + 1)
                    fs_sel = fs
                    for j in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + j + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = seqs[fs_sel, :]
                    if len(all_seqs) == 0:
                        all_seqs = seq_sel
                    else:
                        all_seqs = np.concatenate((all_seqs, seq_sel), axis=0)

        # self.all_seqs = all_seqs[:, (their_input_n - input_n):, :]

        self.dim_used = np.array(range(3, all_seqs.shape[2]))
        #all_seqs = all_seqs[:, (their_input_n - input_n):, 3:]
        all_seqs = all_seqs[:, (their_input_n -  self.in_n ):, :]
        self.all_seqs = all_seqs * 1000

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        motion = self.all_seqs[item]
        motion=motion[:,self.dim_used]
        # 随机反转 一半一半的几率
        if self.data_aug:
            if torch.rand(1)[0] > .5:
                idx = [i for i in range(motion.shape[0] - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                motion = motion[idx]

        human_motion_input = motion[:self.in_n] / 1000  # meter
        human_motion_target = motion[self.in_n:] / 1000  # meter
        return human_motion_input,human_motion_target