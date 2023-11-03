import argparse
import os, sys
import numpy as np
import copy
import time

from models.config import config
from models.HypergraphModel import HypergraphRep as Model
from datasets_setting.h36m import Datasets as H36MDataset
from datasets_setting.h36m_eval import Datasets as H36MEval

from test_H36M import results_keys
from test_H36M import test

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default=None, help='=exp name')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')
parser.add_argument('--reverse_weight', type=float, default=0., help='=the weight of the reverse loss')
parser.add_argument('--iter', type=int, default=40000, help='=iter number')

args = parser.parse_args()

config._curUsed=config.h36m
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.num_layers = args.num
config.reverse_weight=args.reverse_weight
config.cos_lr_total_iters=args.iter

current_train_id = time.strftime("%Y_%m_%d_%HH_%MM_%SS", time.localtime())
script_name = os.path.basename(sys.argv[0])[:-3]
exp_name="{}_{}.txt".format(script_name,current_train_id)
log_name = '{}_bs{}_num{}_reverseweight{}_{}'.format(script_name,config.batch_size,config.motion_mlp.num_layers,config.reverse_weight,current_train_id)
ckpt = os.path.join(config.checkpoint, log_name)
if not os.path.isdir(ckpt):
    os.makedirs(ckpt)

log_dir=os.path.join(ckpt,exp_name)
acc_log = open(log_dir, 'a')
writer = SummaryWriter()

for key in config :
    print(key,":",config[key])
    acc_log.write(''.join(str(key)+' : '+ str(config[key]) + '\n'))

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

def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    if nb_iter > 30000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :

    if config.deriv_input:
        b,n,c = h36m_motion_input.shape
        h36m_motion_input_ = h36m_motion_input.clone()
        h36m_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.input_length], h36m_motion_input_.cuda())
    else:
        h36m_motion_input_ = h36m_motion_input.clone()

    motion_pred = model(h36m_motion_input_.cuda())
    motion_pred = torch.matmul(idct_m[:, :config.motion.input_length, :], motion_pred)

    if config.deriv_output:
        offset = h36m_motion_input[:, -1:].cuda()
        motion_pred = motion_pred[:, :config.motion.target_length] + offset
    else:
        motion_pred = motion_pred[:, :config.motion.h36m_target_length]

    b,n,c = h36m_motion_target.shape
    motion_pred = motion_pred.reshape(b,n,22,3).reshape(-1,3)
    h36m_motion_target = h36m_motion_target.cuda().reshape(b,n,22,3).reshape(-1,3)
    loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1))

    if config.use_identity_loss:
        #receive the reverse input
        h36m_motion_input_reverse = h36m_motion_input[:, config.motion.target_length:, ].clone().cuda()
        # motion_pred=motion_pred.reshape(b,n,22,3).reshape(b,n,-1)
        h36m_motion_input_reverse = torch.cat([h36m_motion_input_reverse, h36m_motion_target.reshape(b,config.motion.target_length,-1)], dim=1)
        reverse_input_idx = [i for i in range(h36m_motion_input_reverse.size(1) - 1, -1, -1)]
        reverse_input_idx = torch.LongTensor(reverse_input_idx)
        h36m_motion_input_reverse = h36m_motion_input_reverse[:,reverse_input_idx]
        #convert the reverse input to the DCT-based features
        h36m_motion_input_reverse_ = torch.matmul(dct_m[:, :, :config.motion.input_length], h36m_motion_input_reverse.cuda())
        #prediction
        motion_pred_reverse = model(h36m_motion_input_reverse_.cuda())
        #reconvert the DCT-based feature to 3D pose features
        motion_pred_reverse = torch.matmul(idct_m[:, :config.motion.input_length, :], motion_pred_reverse)
        reverse_offset = h36m_motion_input_reverse[:, -1:].cuda()
        motion_pred_reverse = motion_pred_reverse[:, :config.motion.target_length] + reverse_offset
        motion_pred_reverse = motion_pred_reverse.reshape(b, n, 22, 3).reshape(-1, 3)
        h36m_motion_target_reverse=h36m_motion_input[:, :config.motion.target_length, ].clone().cuda()
        reverse_target_idx = [i for i in range(h36m_motion_target_reverse.size(1) - 1, -1, -1)]
        reverse_target_idx = torch.LongTensor(reverse_target_idx)
        h36m_motion_input_reverse = h36m_motion_input_reverse[:,reverse_target_idx]
        h36m_motion_input_reverse = h36m_motion_input_reverse.cuda().reshape(b, config.motion.target_length, 22, 3).reshape(-1, 3)
        reverse_loss = torch.mean(torch.norm(motion_pred_reverse - h36m_motion_input_reverse, 2, 1))
        loss=loss+config.reverse_weight*reverse_loss

    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b,n,22,3)
        dmotion_pred = gen_velocity(motion_pred)
        motion_gt = h36m_motion_target.reshape(b,n,22,3)
        dmotion_gt = gen_velocity(motion_gt)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr

model = Model(config)
model.train()
model.cuda()

config.motion.target_length = config.motion.target_length_train
dataset = H36MDataset(config,split=0, data_aug=config.data_aug)

shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, sampler=sampler, shuffle=shuffle, pin_memory=True)


eval_config = copy.deepcopy(config)
eval_config.motion.target_length = eval_config.motion.target_length_eval
shuffle = False

#set validation dataset
val_config = copy.deepcopy(config)
val_config.motion.target_length = val_config.motion.target_length_eval
val_dataset = H36MEval(val_config,split=2, data_aug=False)
val_shuffle = False
val_sampler = None
val_dataloader = DataLoader(val_dataset, batch_size=128,num_workers=0, drop_last=False, sampler=val_sampler, shuffle=val_shuffle, pin_memory=True)


actions = ["walking", "eating", "smoking", "discussion", "directions",
           "greeting", "phoning", "posing", "purchases", "sitting",
           "sittingdown", "takingphoto", "waiting", "walkingdog",
           "walkingtogether"]
eval_config = copy.deepcopy(config)
eval_config.motion.target_length = eval_config.motion.target_length_eval
shuffle = False

test_data = dict()
for act in actions:
    test_dataset = H36MEval(eval_config,actions=[act],split=2, data_aug=False)
    test_data[act] = DataLoader(test_dataset, batch_size=128,num_workers=0, drop_last=False,sampler=sampler, shuffle=shuffle, pin_memory=True)

# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),  lr=config.cos_lr_max,  weight_decay=config.weight_decay)

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

err_best = 10000

while (nb_iter + 1) < config.cos_lr_total_iters:


    for (h36m_motion_input, h36m_motion_target) in dataloader:

        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        avg_loss += loss
        avg_lr += current_lr
        # print('training.............')
        if (nb_iter + 1) % config.print_every ==  0 :
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every
            print('iter:{}   loss:{}'.format(nb_iter + 1, avg_loss))

        model.eval()


        if (((nb_iter + 1)<25000) &((nb_iter + 1) % (config.save_every*5) == 0))|(((nb_iter + 1)>25000) &((nb_iter + 1) % (config.save_every*5) == 0)):
            print('iter:', nb_iter+1)
            model.eval()

            valid_value= test(val_config, model, val_dataloader)
            if sum(valid_value) / len(valid_value) < err_best:
                err_best = sum(valid_value) / len(valid_value)
            else:
                continue

            errs = np.zeros([len(actions) + 1, len(results_keys)])
            for i, act in enumerate(actions):
                errs[i] = test(eval_config, model, test_data[act])
            errs[-1] = np.mean(errs[:-1], axis=0)
            print("Mean Error:",errs[-1])
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))
            line = ''
            err_name=''
            for ii in errs[-1]:
                line += str('{:.3f}'.format(ii)) + ' '
                err_name += str('{:.3f}'.format(ii)) + '_'
            line += '\n'
            err_name = err_name[:-1]
            acc_log.write(''.join(line))
            torch.save(model.state_dict(), config.checkpoint + '/' + log_name+ '/' + 'iter-' + str(nb_iter + 1) +'-AverageEror:'+err_name+ '.pth')
            model.train()

        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()
