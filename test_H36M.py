import argparse
import numpy as np
from models.config  import config
from models.HypergraphModel import HypergraphRep as Model
from datasets_setting.h36m_eval import Datasets as H36MEval
import torch
import os
from torch.utils.data import DataLoader

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']
#test_times=   [80ms, 160ms, 320ms, 400ms,560ms, 720ms, 880ms, 1000ms]


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


dct_m,idct_m = get_dct_matrix(config.motion.input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)


def regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36):
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)

    for (motion_input, motion_target) in pbar:
        motion_input = motion_input.cuda()
        b,l,c = motion_input.shape
        num_samples += b

        motion_input = motion_input.reshape(b, l, 32, 3)
        motion_input = motion_input[:, :, joint_used_xyz].reshape(b, l, -1)
        outputs = []
        step = config.motion.target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.input_length], motion_input_.cuda())
                else:
                    motion_input_ = motion_input.clone()
                output = model(motion_input_)
                output = torch.matmul(idct_m[:, :config.motion.input_length, :], output)[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1,step,1)

            output = output.reshape(-1, 22*3)
            output = output.reshape(b,step,-1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)[:,:25]

        motion_target = motion_target.detach()
        b,l_out,c = motion_target.shape

        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b, l_out, config._curUsed.node_n, 3)
        motion_pred = motion_target.clone().reshape(b, l_out, 32, 3)
        motion_pred[:, :, joint_used_xyz] = pred_rot

        tmp = motion_gt.clone().reshape(b, l_out, 32, 3)
        tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
        motion_pred = tmp
        motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]

        motion_pred = motion_pred.reshape(b, l_out, -1, 3)
        motion_gt = motion_gt.reshape(b, l_out, -1, 3)

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36


def test(config, model, dataloader) :
    m_p3d_h36 = np.zeros([config.motion.target_length])
    titles = np.array(range(config.motion.target_length)) + 1
    joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    num_samples = 0

    pbar = dataloader

    m_p3d_h36 = regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36)

    ret = {}
    for j in range(config.motion.target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [round(ret[key][0], 3) for key in results_keys]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    config._curUsed = config.h36m
    config.motion_mlp.num_layers = 64
    for key in config:
        print(key, ":", config[key])
    model = Model(config)

    pth_name = 'pretrain/BestModel-AverageEror_9.633_21.791_46.700_57.595_75.611_89.685_101.540_109.543.pth'
    pth_full_pth=os.path.join(config.checkpoint,pth_name)
    parser.add_argument('--model-pth', type=str, default=pth_full_pth, help='=model path')
    args = parser.parse_args()
    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]

    shuffle = False
    sampler = None
    config.motion.target_length = config.motion.target_length_eval
    test_data = dict()
    for act in actions:
        test_dataset = H36MEval(config, actions=[act], split=2, data_aug=False)
        test_data[act] = DataLoader(test_dataset, batch_size=128, num_workers=0, drop_last=False, sampler=sampler,
                                    shuffle=shuffle, pin_memory=True)

    # model.eval()
    errs = np.zeros([len(actions) + 1, len(results_keys)])
    for i, act in enumerate(actions):
        errs[i] = test(config, model, test_data[act])
        print('act:',act)
        print(errs[i])
    errs[-1] = np.mean(errs[:-1], axis=0)
    print("Mean Error:", errs[-1])


