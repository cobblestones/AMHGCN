import numpy as np
from models.config  import config
import torch

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

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
    # the index is adopted form https://github.com/wei-mao-2019/LearnTrajDep
    joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36]).astype(np.int64)
    joint_equal = np.array([15, 15, 15, 23, 23, 32, 32]).astype(np.int64)

    for (motion_input, motion_target) in pbar:
        motion_input = motion_input.cuda()
        b,l,c = motion_input.shape
        num_samples += b

        motion_input = motion_input[:, :, joint_used_xyz]
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

            output = output.reshape(-1, config._curUsed.node_n*3)
            output = output.reshape(b,step,-1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)[:,:25]

        motion_target = motion_target.detach()
        b,l_out,c= motion_target.shape

        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b,l_out,config._curUsed.node_n,3).reshape(b,l_out,-1)
        motion_pred = motion_target.clone().reshape(b,l_out,38,3).reshape(b,l_out,-1)
        motion_pred[:, :, joint_used_xyz] = pred_rot

        tmp = motion_gt.clone()
        tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
        motion_pred = tmp
        motion_pred=motion_pred.reshape(b,l_out,38,3)
        motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]

        motion_pred = motion_pred.reshape(b,l_out,-1,3)
        motion_gt = motion_gt.reshape(b, l_out, -1, 3)
        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36


def test(config, model, dataloader) :

    m_p3d_h36 = np.zeros([config.motion.target_length])
    titles = np.array(range(config.motion.target_length)) + 1
    dim_used = np.array(
        [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 42, 43, 44,
         45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 75, 76, 77, 78, 79, 80, 84,
         85,86, 90, 91, 92, 93, 94, 95, 96, 97, 98, 102, 103, 104, 105, 106, 107, 111, 112, 113]).astype(np.int64)
    num_samples = 0
    pbar = dataloader
    m_p3d_h36 = regress_pred(model, pbar, num_samples, dim_used, m_p3d_h36)
    ret = {}
    for j in range(config.motion.target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [round(ret[key][0], 1) for key in results_keys]

