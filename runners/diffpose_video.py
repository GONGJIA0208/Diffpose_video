import os
import logging
from time import time
import glob
import argparse

import os.path as path
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn


# from models.gcnpose import GCNpose, adj_mx_from_edges
from models.mixste import MixSTE2
from models.gcndiff import GCNdiff, adj_mx_from_edges
from models.ema import EMAHelper

from common.utils import *
from common.utils_diff import get_beta_schedule, generalized_steps
from common.data_utils import fetch_me, read_3d_data_me, create_2d_data
# from common.generators import PoseGenerator_gmm
from common.loss import mpjpe, p_mpjpe

from common.data_video_utils import *

from common.camera import *
from common.loss import *
from common.generators import ChunkedGenerator_Seq, UnchunkedGenerator_Seq, PoseGenerator_gmm


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Diffpose(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        # GraFormer mask
        self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                                True, True, True, True, True, True, True]]]).cuda()
        
        # Generate Diffusion sequence parameters
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    # prepare 2D and 3D skeleton for model training and testing 
    def prepare_mixste_data(self):
        args, config = self.args, self.config
        print('==> Using settings {}'.format(args))
        print('==> Using configures {}'.format(config))
        
        # load dataset
        if config.data.dataset == "human36m":
            from common.h36m_dataset import Human36mDataset
            dataset = Human36mDataset(config.data.dataset_path)
        else:
            raise KeyError('Invalid dataset')
        self.dataset = dataset

        print('Preparing data...')
        for subject in dataset.subjects():
            for action in dataset[subject].keys():
                anim = dataset[subject][action]
                if 'positions' in anim:
                    positions_3d = []
                    for cam in anim['cameras']:
                        pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                        pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                        positions_3d.append(pos_3d)
                    anim['positions_3d'] = positions_3d

        print('Loading 2D detections...')
        keypoints_valid = np.load(config.data.dataset_path_test_2d, allow_pickle=True)
        keypoints_metadata = keypoints_valid['metadata'].item()
        keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
        keypoints_valid = keypoints_valid['positions_2d'].item()
        self.keypoints_valid = create2Ddata(dataset, keypoints_valid)
        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
        
        self.subjects_train = args.subjects_train.split(',')
        self.subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
        self.subjects_test = args.subjects_test.split(',')

        self.action_filter = None if args.actions == '*' else args.actions.split(',')
        self.dataset = dataset
        if self.action_filter is not None:
            print('Selected actions:', self.action_filter)

        self.cameras_valid, self.poses_valid, self.poses_valid_2d = fetch(self.keypoints_valid, self.dataset, self.subjects_test, self.action_filter)

        # set receptive_field as number assigned
        self.receptive_field = args.number_of_frames
        print('INFO: Receptive field: {} frames'.format(self.receptive_field))
        self.pad = (self.receptive_field -1) // 2 # Padding on each side
        self.min_loss = args.min_loss
        self.width = cam['res_w']
        self.height = cam['res_h']
        self.num_joints = keypoints_metadata['num_joints']
        self.causal_shift = 0

    def prepare_Diffpose_data(self):
        args, config = self.args, self.config
        print('==> Using settings {}'.format(args))
        print('==> Using configures {}'.format(config))
        
        # load dataset
        if config.data.dataset == "human36m":
            from common.h36m_dataset_diff import Human36mDataset_diff, TRAIN_SUBJECTS
            dataset = Human36mDataset_diff(config.data.dataset_path)
            self.subjects_train_diff = TRAIN_SUBJECTS
            self.dataset_diff = read_3d_data_me(dataset)
            self.keypoints_train_diff = create_2d_data(config.data.dataset_path_train_2d, dataset)
        else:
            raise KeyError('Invalid dataset')

    # create diffusion model
    def create_diffusion_model(self, model_path = None):
        args, config = self.args, self.config
        edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        self.model_diff = GCNdiff(adj.cuda(), config).cuda()
        self.model_diff = torch.nn.DataParallel(self.model_diff)
        
        # load pretrained model
        if model_path:
            states = torch.load(model_path)
            self.model_diff.load_state_dict(states[0])
            
    def create_pose_model(self, model_path = None):
        args, config = self.args, self.config

        self.model_pose =  MixSTE2(num_frame=self.receptive_field, num_joints=self.num_joints, in_chans=2,
         embed_dim_ratio=512, depth=8, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0)

        self.model_pose = torch.nn.DataParallel(self.model_pose)
        self.model_pose = self.model_pose.cuda()
        # load pretrained model
        if model_path:
            logging.info('initialize model by:' + model_path)
            states = torch.load(model_path)
            self.model_pose.load_state_dict(states['model_pos'])
        else:
            logging.info('initialize model randomly')

    def train(self):
        args, config, src_mask = self.args, self.config, self.src_mask

        # initialize the recorded best performance
        best_p1, best_epoch = 1000, 0
        # skip rate when sample skeletons from video
        stride = self.args.downsample
        
        # create dataloader
        if config.data.dataset == "human36m":
            poses_train, poses_train_2d, actions_train, camerapara_train\
                = fetch_me(self.subjects_train_diff, self.dataset_diff, self.keypoints_train_diff, self.action_filter, stride)
            data_loader = train_loader = data.DataLoader(
                PoseGenerator_gmm(poses_train, poses_train_2d, actions_train, camerapara_train),
                batch_size=config.training.batch_size, shuffle=True,\
                    num_workers=config.training.num_workers, pin_memory=True)
        else:
            raise KeyError('Invalid dataset')
        
        optimizer = get_optimizer(self.config, self.model_diff.parameters())
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
        else:
            ema_helper = None
        
        start_epoch, step = 0, 0
        
        lr_init, decay, gamma = self.config.optim.lr, self.config.optim.decay, self.config.optim.lr_gamma
      
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time()
            data_time = 0

            # Switch to train mode
            torch.set_grad_enabled(True)
            self.model_diff.train()
            
            epoch_loss_diff = AverageMeter()

            for i, (targets_uvxyz, targets_noise_scale, _, targets_3d, _, _) in enumerate(data_loader):
                data_time += time() - data_start
                step += 1

                # to cuda
                targets_uvxyz, targets_noise_scale, targets_3d = \
                    targets_uvxyz.to(self.device), targets_noise_scale.to(self.device), targets_3d.to(self.device)
                
                # generate nosiy sample based on seleted time t and beta
                n = targets_3d.size(0)
                x = targets_uvxyz
                e = torch.randn_like(x)
                b = self.betas            
                t = torch.randint(low=0, high=self.num_timesteps,
                                  size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                e = e*(targets_noise_scale)
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                # generate x_t (refer to DDIM equation)
                x = x * a.sqrt() + e * (1.0 - a).sqrt()
                
                # predict noise
                output_noise = self.model_diff(x, src_mask, t.float(), 0)
                loss_diff = (e - output_noise).square().sum(dim=(1, 2)).mean(dim=0)
                
                optimizer.zero_grad()
                loss_diff.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model_diff.parameters(), config.optim.grad_clip)                
                optimizer.step()
            
                epoch_loss_diff.update(loss_diff.item(), n)
            
                if self.config.model.ema:
                    ema_helper.update(self.model_diff)
                
                if i%100 == 0 and i != 0:
                    logging.info('| Epoch{:0>4d}: {:0>4d}/{:0>4d} | Step {:0>6d} | Data: {:.6f} | Loss: {:.6f} |'\
                        .format(epoch, i+1, len(data_loader), step, data_time, epoch_loss_diff.avg))
            
            data_start = time()

            if epoch % decay == 0:
                lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma) 
                
            if epoch % 1 == 0:
                states = [
                    self.model_diff.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(states,os.path.join(self.args.log_path, "ckpt_{}.pth".format(epoch)))
                torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            
                logging.info('test the performance of current model')

                self.validat_hyber(is_train=True)

                # if p1 < best_p1:
                #     best_p1 = p1
                #     best_epoch = epoch
                # logging.info('| Best Epoch: {:0>4d} MPJPE: {:.2f} | Epoch: {:0>4d} MPJEPE: {:.2f} PA-MPJPE: {:.2f} |'\
                #     .format(best_epoch, best_p1, epoch, p1, p2))

    def test_per_action(self, test_generator, action=None, return_predictions=False, use_trajectory_model=False, newmodel=None):
        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_pos_scale = 0
        epoch_loss_3d_vel = 0

        args, config, src_mask = self.args, self.config, self.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
        
        if self.args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        with torch.no_grad():
            self.model_diff.eval()
            self.model_pose.eval()
            N = 0
            for _, batch, batch_2d in test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                inputs_3d = torch.from_numpy(batch.astype('float32'))

                ##### initialize x,y,z via MixSTE ####
                inputs_2d_flip = inputs_2d.clone()
                inputs_2d_flip [:, :, :, 0] *= -1
                inputs_2d_flip[:, :, self.kps_left + self.kps_right,:] = inputs_2d_flip[:, :, self.kps_right + self.kps_left,:]

                ##### convert size
                inputs_3d_p = inputs_3d
                inputs_2d, inputs_3d = eval_data_prepare(self.receptive_field, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare(self.receptive_field, inputs_2d_flip, inputs_3d_p)

                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_2d_flip = inputs_2d_flip.cuda()
                    inputs_3d = inputs_3d.cuda()
                    
                inputs_3d[:, :, 0] = 0
                
                predicted_3d_pos = self.model_pose(inputs_2d)
                predicted_3d_pos_flip = self.model_pose(inputs_2d_flip)
                predicted_3d_pos_flip[:, :, :, 0] *= -1
                predicted_3d_pos_flip[:, :, self.joints_left + self.joints_right] = predicted_3d_pos_flip[:, :,
                                                                       self.joints_right + self.joints_left]
                for i in range(predicted_3d_pos.shape[0]):
                    predicted_3d_pos[i,:,:,:] = (predicted_3d_pos[i,:,:,:] + predicted_3d_pos_flip[i,:,:,:])/2


                #### Peform Reverse Diffusion Process ####
                N_b, P_b, J_b, D_b = predicted_3d_pos.shape[0], predicted_3d_pos.shape[1], predicted_3d_pos.shape[2], predicted_3d_pos.shape[3]

                inputs_xyz = predicted_3d_pos.reshape(N_b*P_b, J_b, D_b).clone()
                # inputs_xyz[:, :, :] -= inputs_xyz[:, :1, :]
                inputs_uv = inputs_2d.reshape(N_b*P_b, J_b, D_b-1).clone()

                input_uvxyz = torch.cat([inputs_uv,inputs_xyz],dim=2)
                input_noise_scale = torch.tensor([0.1,0.1,1,1,1])
                input_noise_scale = input_noise_scale.repeat(N_b*P_b, J_b, 1).cuda()


                # generate distribution
                input_uvxyz = input_uvxyz.repeat(test_times,1,1)
                input_noise_scale = input_noise_scale.repeat(test_times,1,1)
                # select diffusion step
                t = torch.ones(input_uvxyz.size(0)).type(torch.LongTensor).to(self.device)*test_num_diffusion_timesteps
                
                # prepare the diffusion parameters
                x = input_uvxyz.clone()
                e = torch.randn_like(input_uvxyz)
                b = self.betas   
                e = e*input_noise_scale        
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                # x = x * a.sqrt() + e * (1.0 - a).sqrt()
                
                output_uvxyz = generalized_steps(x, src_mask, seq, self.model_diff, self.betas, eta=self.args.eta)
                output_uvxyz = output_uvxyz[0][-1]    
                output_uvxyz = torch.mean(output_uvxyz.reshape(test_times,-1,17,5),0)
                output_xyz = output_uvxyz[:,:,2:]
                output_xyz[:, :, :] -= output_xyz[:, :1, :]

                #### calculate the error
                predicted_3d_pos = output_xyz.reshape(N_b, P_b, J_b, D_b)
                error = mpjpe(predicted_3d_pos, inputs_3d)

                epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

                epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

                inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

                epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

                epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

        if action is None:
            print('----------')
        else:
            print('----'+action+'----')
        e1 = (epoch_loss_3d_pos / N)*1000
        e2 = (epoch_loss_3d_pos_procrustes / N)*1000
        e3 = (epoch_loss_3d_pos_scale / N)*1000
        ev = (epoch_loss_3d_vel / N)*1000
        print('Test time augmentation:', test_generator.augment_enabled())
        print('Protocol #1 Error (MPJPE):', e1, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
        print('Velocity Error (MPJVE):', ev, 'mm')
        print('----------')

        return e1, e2, e3, ev

    def validat_hyber(self, is_train=False):
        args, config, src_mask = self.args, self.config, self.src_mask

        # print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

        all_actions = {}
        all_actions_by_subject = {}
        for subject in self.subjects_test:
            if subject not in all_actions_by_subject:
                all_actions_by_subject[subject] = {}

            for action in self.dataset[subject].keys():
                action_name = action.split(' ')[0]
                if action_name not in all_actions:
                    all_actions[action_name] = []
                if action_name not in all_actions_by_subject[subject]:
                    all_actions_by_subject[subject][action_name] = []
                all_actions[action_name].append((subject, action))
                all_actions_by_subject[subject][action_name].append((subject, action))
        
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []
        # joints_errs_list=[]

        for action_key in all_actions.keys():
            if self.action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = fetch_actions(self.keypoints_valid , self.dataset, all_actions[action_key])
            gen = UnchunkedGenerator_Seq(None, poses_act, poses_2d_act,
                                        pad=self.pad, causal_shift=self.causal_shift, augment=True,
                                        kps_left=self.kps_left, kps_right=self.kps_right, joints_left=self.joints_left,
                                        joints_right=self.joints_right)
            e1, e2, e3, ev = self.test_per_action(gen, action_key)
            
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)

            break

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')

    def test_hyber(self, is_train=False):
        args, config, src_mask = self.args, self.config, self.src_mask

        # print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

        all_actions = {}
        all_actions_by_subject = {}
        for subject in self.subjects_test:
            if subject not in all_actions_by_subject:
                all_actions_by_subject[subject] = {}

            for action in self.dataset[subject].keys():
                action_name = action.split(' ')[0]
                if action_name not in all_actions:
                    all_actions[action_name] = []
                if action_name not in all_actions_by_subject[subject]:
                    all_actions_by_subject[subject][action_name] = []
                all_actions[action_name].append((subject, action))
                all_actions_by_subject[subject][action_name].append((subject, action))
        
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []
        # joints_errs_list=[]

        for action_key in all_actions.keys():
            if self.action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = fetch_actions(self.keypoints_valid , self.dataset, all_actions[action_key])
            gen = UnchunkedGenerator_Seq(None, poses_act, poses_2d_act,
                                        pad=self.pad, causal_shift=self.causal_shift, augment=True,
                                        kps_left=self.kps_left, kps_right=self.kps_right, joints_left=self.joints_left,
                                        joints_right=self.joints_right)
            e1, e2, e3, ev = self.test_per_action(gen, action_key)
            
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')

