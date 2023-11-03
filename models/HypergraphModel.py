import copy
import torch
from torch import nn
from einops.layers.torch import Rearrange
from torch.nn.parameter import Parameter
import math
import numpy as np

class HyperGraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, cur_used_dataset, bias=True):
        super(HyperGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.data_config=config.h36m
        self.node_n=cur_used_dataset.node_n*3
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.joint_att = Parameter(torch.FloatTensor(self.node_n,self. node_n))
        self.part_att = Parameter(torch.FloatTensor(self.node_n, self.node_n))
        self.pose_att = Parameter(torch.FloatTensor(self.node_n, self.node_n))
        self.global_att = Parameter(torch.FloatTensor(self.node_n, self.node_n))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        joint_level_adj, part_level_adj, pose_level_adj= self.obtain_hypergraph_mask(cur_used_dataset)
        joint_levl_mask, part_level_mask, pose_level_mask=self.obtain_hypergraph_mask_3d(joint_level_adj, part_level_adj, pose_level_adj,cur_used_dataset)
        self.joint_Adj = torch.from_numpy(joint_levl_mask).float().cuda()
        self.part_Adj =torch.from_numpy(part_level_mask).float().cuda()
        self.pose_adj =torch.from_numpy(pose_level_mask).float().cuda()
        self.reset_parameters()

    def obtain_hypergraph_mask(self,cur_used_dataset):
        joint_level_adj = np.zeros((cur_used_dataset.node_n, cur_used_dataset.node_n))
        for item in cur_used_dataset.joint_level_edge:
            for i in item:
                for j in item:
                    joint_level_adj[j, i] = 1
                    joint_level_adj[i, j] = 1

        part_level_adj = np.zeros((cur_used_dataset.node_n, cur_used_dataset.node_n))
        for item in cur_used_dataset.part_level_edge:
            for i in item:
                for j in item:
                    part_level_adj[j, i] = 1
                    part_level_adj[i, j] = 1

        pose_level_adj = np.zeros((cur_used_dataset.node_n, cur_used_dataset.node_n))
        for item in cur_used_dataset.pose_level_edge:
            for i in item:
                for j in item:
                    pose_level_adj[j, i] = 1
                    pose_level_adj[i, j] = 1
        return joint_level_adj, part_level_adj, pose_level_adj


    def obtain_hypergraph_mask_3d(self,joint_level_adj,part_level_adj,pose_level_adj,cur_used_dataset):
        node_n=cur_used_dataset.node_n*3
        joint_Adj_Mask = np.zeros((node_n, node_n))
        for i in range(0,len(joint_level_adj)):
            for j in range(0, len(joint_level_adj[0])):
                if joint_level_adj[i][j]==1:
                    for m in range(0,3):
                        for n in range(0,3):
                            joint_Adj_Mask[i*3+m,j*3+n]=1

        part_Adj_Mask = np.zeros((node_n, node_n))
        for i in range(0, len(part_level_adj)):
            for j in range(0, len(part_level_adj[0])):
                if part_level_adj[i][j] == 1:
                    for m in range(0, 3):
                        for n in range(0, 3):
                            part_Adj_Mask[i * 3 + m, j * 3 + n] = 1

        pose_Adj_Mask = np.zeros((node_n, node_n))
        for i in range(0, len(pose_level_adj)):
            for j in range(0, len(pose_level_adj[0])):
                if pose_level_adj[i][j] == 1:
                    for m in range(0, 3):
                        for n in range(0, 3):
                            pose_Adj_Mask[i * 3 + m, j * 3 + n] = 1

        return joint_Adj_Mask,part_Adj_Mask,pose_Adj_Mask


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.joint_att.data.uniform_(-stdv, stdv)
        self.part_att.data.uniform_(-stdv, stdv)
        self.pose_att.data.uniform_(-stdv, stdv)
        self.global_att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        dynamic_att=self.joint_Adj*self.joint_att+self.part_Adj*self.part_att+self.pose_adj*self.pose_att+self.global_att
        output = torch.matmul(dynamic_att, support)
        if self.bias is not None:
            return input + output + self.bias
        else:
            return input + output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class Spatial_FC(nn.Module):
    def __init__(self, dim):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x

class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class HypergraphRepresentationExtractBlocks(nn.Module):

    def __init__(self, dim, seq, cur_used_dataset, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial',):
        super().__init__()

        if not use_spatial_fc:
            self.fc0 = Temporal_FC(seq)
        else:
            self.fc0 = Spatial_FC(dim)
        self.add_Temporal_FC=True
        if self.add_Temporal_FC:
            self.fc_temporal=Temporal_FC(seq)
            self.norm_temporal = LN_v2(seq)

        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()
        self.gcn_spatial = HyperGraphConvolution(seq, seq,cur_used_dataset=cur_used_dataset, bias=True,)
        self.gcn_spatial_fc = LN(dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)

        nn.init.constant_(self.fc0.fc.bias, 0)

        if self.add_Temporal_FC:
            nn.init.xavier_uniform_(self.fc_temporal.fc.weight, gain=1e-8)
            nn.init.constant_(self.fc_temporal.fc.bias, 0)


    def forward(self, x):
        y_=self.gcn_spatial(x)
        y_=self.gcn_spatial_fc(y_)

        x_ = self.fc0(x+y_)
        x_ = self.norm0(x_)

        x = x + x_

        return x



class HypergraphRep(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(HypergraphRep, self).__init__()

        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.HypergraphBlocks = nn.Sequential(*[
            HypergraphRepresentationExtractBlocks(dim=self.config.motion_mlp.hidden_dim, seq=self.config.motion_mlp.seq_len, cur_used_dataset=config._curUsed,use_norm=self.config.motion_mlp.with_normalization,
                     use_spatial_fc=self.config.motion_mlp.spatial_fc_only, layernorm_axis=self.config.motion_mlp.norm_axis)
            for i in range(self.config.motion_mlp.num_layers)])

        self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)
        self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)


    def forward(self, motion_input):

        motion_feats = self.motion_fc_in(motion_input)
        motion_feats = self.arr0(motion_feats)
        motion_feats = self.HypergraphBlocks(motion_feats)
        motion_feats = self.arr1(motion_feats)
        motion_feats = self.motion_fc_out(motion_feats)

        return motion_feats

