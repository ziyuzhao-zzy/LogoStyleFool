import argparse
import os
parser = argparse.ArgumentParser(description='LogoStyleFool_attacking')
parser.add_argument('--model', type=str, default='C3D', choices=['C3D', 'I3D'], help='the attacked model')
parser.add_argument('--dataset', type=str, default='UCF101', choices=['UCF101', 'HMDB51'], help='the dataset')
parser.add_argument('--gpu',  type=int, default=0, help='use which gpu')
parser.add_argument('--video_npy_path', type=str, default='/home/zzy/UCF/UCF-101_npy/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.npy', help='the video path in npy forms')
parser.add_argument('--label', type=int, default=0, help='label of the video')
parser.add_argument('--target', action='store_true', help='targeted attack or untargeted attack (default)')
parser.add_argument('--target_class', type=int, default=1, help='target attack class')
parser.add_argument('--output_path', type=str, default='res/', help='output_adversarial_npy_path')
parser.add_argument('--rl_batch', type=int, default=50, help='batch size of rl')
parser.add_argument('--steps', type=int, default=50, help='steps of rl')
parser.add_argument('--sigma', type=float, default=0.004, help='sigma of rl reward')
parser.add_argument('--tau', type=float, default=0.2, help='tau of rl reward')
parser.add_argument('--logo_num', type=int, default=100, help='num of logos')
parser.add_argument('--style_num', type=int, default=5, help='num of style imgs')
parser.add_argument('--max_iters', type=int, default=150000, help='max iters of LogoS-DCT.')
parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon of LogoS-DCT.')
parser.add_argument('--linf_bound', type=float, default=0.1, help='linf bound of perturbation')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
import cv2 as cv
import numpy as np
from model_init import model_initial
import torchvision.transforms as transforms
from Runattack import untargetted_attack, targetted_attack
import logging
from generate_video import tensor_to_video
if __name__ == '__main__':
    target = args.target
    mod = args.model
    dataset = args.dataset
    vid_path = args.video_npy_path
    label = torch.tensor(args.label).cuda()
    target_label = torch.tensor(args.target_class).cuda()
    output_path = args.output_path
    rl_batch = args.rl_batch
    steps = args.steps
    sigma = args.sigma
    tau = args.tau
    logo_num = args.logo_num
    style_num = args.style_num
    max_iters = args. max_iters
    epsilon = args.epsilon
    linf_bound = args.linf_bound
    model = model_initial(mod, dataset)
    dir_title = vid_path.split('/')[-2]
    dir_video = vid_path.split('/')[-1].split('.')[0]

    if target:
        attack_dir = os.path.join(output_path,'target_{}'.format(target_label.item()))
        attack_dir = os.path.join(attack_dir, dir_title, dir_video)
        if not os.path.exists(attack_dir):
            os.makedirs(attack_dir)
        logging.basicConfig(filename=attack_dir+'/log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        perturb_style, logo_clip, adv_clip, success, queries = targetted_attack(vid_path=vid_path, model=model, label=label, target_label=target_label, rl_batch = rl_batch, steps = steps, sigma = sigma, tau = tau, 
                    logo_num = logo_num, style_num = style_num, max_iters = max_iters, epsilon = epsilon, linf_bound = linf_bound)
        np.save(attack_dir + '/logo_clip.npy', logo_clip.cpu())
        np.save(attack_dir + '/adv_clip.npy', adv_clip.cpu())
        tensor_to_video(logo_clip, attack_dir + '/logo_clip.avi')
        tensor_to_video(adv_clip, attack_dir + '/adv_clip.avi')
        with open(output_path+'results.txt', 'a') as f:
            f.write(vid_path+" "+str(int(success[0]))+" "+str(int(success[1]))+" "+str(queries[0])+" "+str(queries[1])+" "+str(queries[2])+" "+str(sum(queries))+"\n")
    else:
        attack_dir = os.path.join(output_path, 'non-target')
        attack_dir = os.path.join(attack_dir, dir_title, dir_video)
        if not os.path.exists(attack_dir):
            os.makedirs(attack_dir)
        logging.basicConfig(filename=attack_dir+'/log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        perturb_style, logo_clip, adv_clip, success, queries = untargetted_attack(vid_path=vid_path, model=model, label=label, steps = steps, logo_num = logo_num, style_num = style_num, epsilon = epsilon)
        np.save(attack_dir + '/logo_clip.npy', logo_clip.cpu())
        np.save(attack_dir + '/adv_clip.npy', adv_clip.cpu())
        tensor_to_video(adv_clip, attack_dir + '/adv_clip.avi')
        tensor_to_video(logo_clip, attack_dir + '/logo_clip.avi')
        with open(output_path+'results.txt', 'a') as f:
            f.write(vid_path+" "+str(int(success[0]))+" "+str(int(success[1]))+" "+str(queries[0])+" "+str(queries[1])+" "+str(queries[2])+" "+str(sum(queries))+"\n")