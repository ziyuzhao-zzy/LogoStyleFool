import os
import torch
from easydict import EasyDict as edict

# LogoAttack config
LogoA_cfg = edict()

# config LogoAttack
def configure_LogoA(rl_batch=50, steps=50, sigma = 0.004, tau = 0.2, target = False, target_label = torch.LongTensor([0])):
    # Attack's shared params
    LogoA_cfg.lr = 0.03  # learning rate for RL agent (default: 0.03)
    LogoA_cfg.rl_batch = rl_batch  # batch number when optimizing a RL agent (default: 50)
    LogoA_cfg.steps = steps  # steps to optimize each RL agent (default: 50)
    LogoA_cfg.sigma = sigma  # sigam to control the Area reward (default: 0.004)
    LogoA_cfg.tau = tau  # sigam to control the Distance reward (default: 0.2)
    LogoA_cfg.baseline_sub = True  # use baseline subtraction mode
    LogoA_cfg.target = target
    LogoA_cfg.target_label = target_label
    LogoA_cfg.attack_dir = os.path.join('attack')