from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils import data_utils
from matplotlib import pyplot as plt
import torch


class Datasets(Dataset):

    def __init__(self, config, actions=None, split=0,data_aug=False):
        '''
        :param config:
        :param split:
        :param data_aug:
        '''
        self.path_to_data = "./datasets/h36m/"
        self.split = split
        self.in_n = config.motion.input_length
        self.out_n = config.motion.target_length
        self.sample_rate = 2
        self.skip_rate=1
        self.data_aug = data_aug
        self.p3d = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n
        # subs = np.array([[1, 6, 7, 8, 9], [11], [5]])
        subs = [[1, 6, 7, 8, 9], [11], [5]]
        self.dim_used = np.array(
            [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]).astype(np.int64)
        # acts = data_utils.define_actions(actions)
        if actions is None:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
        else:
            acts = actions
        joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg",
                      "LeftFoot",
                      "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                      "LeftForeArm",
                      "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm",
                      "RightForeArm",
                      "RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]

        subs = subs[split]
        key = 0
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                if self.split <= 1:
                    for subact in [1, 2]:  # subactions
                        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                        the_sequence = data_utils.readCSVasFloat(filename)
                        n, d = the_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        num_frames = len(even_list)
                        the_sequence = np.array(the_sequence[even_list, :])
                        the_sequence = torch.from_numpy(the_sequence).float().cuda()
                        # remove global rotation and translation
                        the_sequence[:, 0:6] = 0
                        p3d = data_utils.expmap2xyz_torch(the_sequence)
                        # self.p3d[(subj, action, subact)] = p3d.view(num_frames, -1).cpu().data.numpy()
                        self.p3d[key] = p3d.view(num_frames, -1).cpu().data.numpy()

                        valid_frames = np.arange(0, num_frames - seq_len + 1, self.skip_rate)

                        # tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        key += 1
                else:
                    # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 1)
                    the_sequence1 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence1.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames1 = len(even_list)
                    the_sequence1 = np.array(the_sequence1[even_list, :])
                    the_seq1 = torch.from_numpy(the_sequence1).float().cuda()
                    the_seq1[:, 0:6] = 0
                    p3d1 = data_utils.expmap2xyz_torch(the_seq1)
                    # self.p3d[(subj, action, 1)] = p3d1.view(num_frames1, -1).cpu().data.numpy()
                    self.p3d[key] = p3d1.view(num_frames1, -1).cpu().data.numpy()

                    # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 2)
                    the_sequence2 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence2.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames2 = len(even_list)
                    the_sequence2 = np.array(the_sequence2[even_list, :])
                    the_seq2 = torch.from_numpy(the_sequence2).float().cuda()
                    the_seq2[:, 0:6] = 0
                    p3d2 = data_utils.expmap2xyz_torch(the_seq2)

                    # self.p3d[(subj, action, 2)] = p3d2.view(num_frames2, -1).cpu().data.numpy()
                    self.p3d[key + 1] = p3d2.view(num_frames2, -1).cpu().data.numpy()

                    # print("action:{}".format(action))
                    # print("subact1:{}".format(num_frames1))
                    # print("subact2:{}".format(num_frames2))

                    #adopted form MSRGCN
                    # if test_manner == "all":
                    #     # # 全部数据用来测试
                    #     fs_sel1 = [np.arange(i, i + seq_len) for i in range(num_frames1 - 100)]
                    #     fs_sel2 = [np.arange(i, i + seq_len) for i in range(num_frames2 - 100)]
                    # elif test_manner == "8":
                    #     # 随机取 8 个
                    #     fs_sel1, fs_sel2 = find_indices_srnn(num_frames1, num_frames2, seq_len)


                    # test on 256
                    # fs_sel1, fs_sel2 = data_utils.find_indices_256(num_frames1, num_frames2, seq_len,
                    #                                                input_n=self.in_n)
                    # valid_frames_1 = fs_sel1[:, 0]

                    fs_sel1 = np.arange(0, num_frames1 - 100)
                    valid_frames_1=fs_sel1

                    tmp_data_idx_1 = [key] * len(valid_frames_1)
                    tmp_data_idx_2 = list(valid_frames_1)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                    fs_sel2 = np.arange(0, num_frames2 - 100)
                    valid_frames_2 = fs_sel2
                    # valid_frames_2 = fs_sel2[:, 0] # test on 256
                    tmp_data_idx_1 = [key + 1] * len(valid_frames_2)
                    tmp_data_idx_2 = list(valid_frames_2)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    key += 2

        # ignore constant joints and joints at same position with other joints
        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        self.dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        motion = self.p3d[key][fs]
        length,dim=motion.shape
        motion=motion.reshape(length,-1,3)
        motion = motion[:, self.dim_used]
        motion=motion.reshape(length,-1)

        # 随机反转 一半一半的几率
        if self.data_aug:
            if torch.rand(1)[0] > .5:
                idx = [i for i in range(motion.shape[0] - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                motion = motion[idx]

        human_motion_input = motion[:self.in_n] / 1000  # meter
        human_motion_target = motion[self.in_n:] / 1000  # meter

        # human_motion_input = human_motion_input.float()
        # human_motion_target = human_motion_target.float()

        return human_motion_input, human_motion_target
