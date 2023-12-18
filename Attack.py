import os
import torch
from easydict import EasyDict as edict
from Agent import agent
from config import LogoA_cfg
from transformlogo import logoset
from style_transfer import style_perturb_logo
import logging
class LogoA():
    def __init__(self, dir_title):
        # attack dirs
        self.attack_dir = os.path.join(LogoA_cfg.attack_dir, dir_title)

    def attack(self, model, input_tensor, label_tensor, input_name, logo_num, styles):
        # set up attack-dirs
        attack_dir = os.path.join(self.attack_dir, input_name)
        if not os.path.exists(attack_dir):
            os.makedirs(attack_dir)
        torch.cuda.empty_cache()
        # set records
        rcd = edict()
        rcd.masks = []
        rcd.RGB_paintings = []
        rcd.combos = []
        rcd.dis = []
        rcd.combo = []
        rcd.queries = []
        rcd.time_used = []
        # attack
        logos = logoset[:logo_num]
        mask_input_tensor, dis, combo, queries, time_used = agent.attack(
            model=model,
            input_tensor=input_tensor,
            target_tensor=label_tensor,
            sigma=LogoA_cfg.sigma,
            tau=LogoA_cfg.tau,
            logos=logos,
            styles=styles,
            lr=LogoA_cfg.lr,
            baseline_subtraction=LogoA_cfg.baseline_sub,
            rl_batch=LogoA_cfg.rl_batch,
            steps=LogoA_cfg.steps,
            target=LogoA_cfg.target,
            target_label=LogoA_cfg.target_label
        )
        # update records
        rcd.masks.append(mask_input_tensor)
        rcd.dis.append(dis)
        rcd.queries.append(queries)
        rcd.time_used.append(time_used)
        rcd.combo.append(combo)
        # print records
        logging.info('RLres:queries: {:.4f} dis: {:.4f}'.format(rcd.queries[0], rcd.dis[0].item()))
        logging.info('RL combo:{}'.format(rcd.combo[0]))
        print('queries: {:.4f} | dis: {:.4f}'.format(rcd.queries[0], rcd.dis[0].item()))
        print(rcd.combo[0])
        return mask_input_tensor, rcd, attack_dir
