import logging
import numpy as np
import torch
import Attack
from config import configure_LogoA
from simba import simba_perturb, simba_dct


ks = [0.75, 0.8125, 0.875, 0.9375, 1.0] # the scaling ratio of logo size


def untargetted_attack(vid_path, model, label, rl_batch = 30, steps = 50, sigma = 250, tau = 5, logo_num = 10, style_num = 5, epsilon = 0.2):
    vid = np.load(vid_path).transpose(0, 3, 1, 2)/255
    vid = torch.tensor(vid, dtype=torch.float, device='cuda')
    configure_LogoA(rl_batch=rl_batch, steps=steps, sigma=sigma, tau=tau)
    prob, predict, _ = model(vid)
    logging.info('Begin Untargetted Attack with rl_batch = {}, steps = {}, sigma = {}, tau = {}, logo_num = {}, style_num = {}, epsilon = {}'.format(rl_batch, steps, sigma, tau, logo_num, style_num, epsilon))
    if predict == label:
        dir_title = vid_path.split('/')[-2]
        LA = Attack.LogoA(dir_title)

        perturb_style, query1 = simba_perturb(model=model, y=label, num=style_num, targeted = False)
        logging.info('{} queries to find perturb_stylies'.format(query1))
        logo_clip, rcd, _ = LA.attack(model=model, input_tensor=vid, label_tensor=label, input_name='{}'.format(vid_path.split('/')[-1].split('.')[0]), logo_num = logo_num, styles = perturb_style)
        _, predict1, _ = model(logo_clip)

        adv_clip = logo_clip
        success1 = False
        success2 = False
        query3 = 0
        if predict1 != label:
            success1 = True
            success2 = True
        else:
            adv_clip, probs, succs, dctqueries, l2_norms, linf_norms = simba_dct(model=model, clip=logo_clip, label=label, h=rcd.combo[0][0], w=rcd.combo[0][1], size=int(32 * ks[rcd.combo[0][2]]), max_iters=150000, freq_dims=int(32 * ks[rcd.combo[0][2]]), epsilon=epsilon, linf_bound=0.1,
                    targeted=False, log_every=10)
            _, predict2, _ = model(adv_clip)
            query3 = int(dctqueries.sum())
            if predict2 != label:
                success2 = True
        queries = [query1, rcd.queries[0], query3]
        if success1 == True:
            logging.info('After {} queries to misclassify the video without simba-DCT'.format(sum(queries)))
        elif success2 == True:
            logging.info('After {} queries to misclassify the video'.format(sum(queries)))
        else:
            logging.info('fail with {} queries'.format(sum(queries)))
        success = [success1, success2]
        return perturb_style, logo_clip, adv_clip, success, queries

def targetted_attack(vid_path, model, label, target_label, rl_batch = 50, steps = 50, sigma = 250, tau = 5, logo_num = 10, style_num = 5, max_iters = 150000, epsilon = 0.2, linf_bound = 0.1):
    vid = np.load(vid_path).transpose(0, 3, 1, 2)/255
    vid = torch.tensor(vid, dtype=torch.float, device='cuda')
    
    configure_LogoA(rl_batch=rl_batch, steps=steps, sigma=sigma, tau=tau, target = True, target_label = target_label)
    prob, predict, _ = model(vid)
    logging.info('Begin Targetted Attack with rl_batch = {}, steps = {}, sigma = {}, tau = {}, logo_num = {}, style_num = {}, max_iters = {}, epsilon = {}, linf_bound = {}'.format(rl_batch, steps, sigma, tau, logo_num, style_num, max_iters, epsilon, linf_bound))
    if predict != target_label:
        dir_title = vid_path.split('/')[-2]
        LA = Attack.LogoA(dir_title)

        perturb_style, query1 = simba_perturb(model=model, y=target_label, num=style_num)
        logging.info('{} queries to find perturb_stylies'.format(query1))
        logo_clip, rcd, _ = LA.attack(model=model, input_tensor=vid, label_tensor=label, input_name='{}'.format(vid_path.split('/')[-1].split('.')[0]), logo_num = logo_num, styles = perturb_style)
        _, predict1, _ = model(logo_clip)
        
        adv_clip = logo_clip
        success1 = False
        success2 = False
        query3 = 0
        if predict1 == target_label:
            success1 = True
            success2 = True
        else:
            prob1 = model.get_target_val(logo_clip, target_label.expand(1, 1).contiguous())
            if prob1.item() < 1e-4:
                queries = [query1, rcd.queries[0], query3]
                logging.info('prob too low, fail with {} queries'.format(sum(queries)))
                success = [success1, success2]
                return perturb_style, logo_clip, adv_clip, success, queries
            adv_clip, probs, succs, dctqueries, l2_norms, linf_norms = simba_dct(model=model, clip=logo_clip, label=target_label, h=rcd.combo[0][0], w=rcd.combo[0][1], size=int(32 * ks[rcd.combo[0][2]]), max_iters=max_iters, freq_dims=int(32 * ks[rcd.combo[0][2]]), epsilon=epsilon, linf_bound=linf_bound,
                    targeted=True, log_every=10)
            _, predict2, _ = model(adv_clip)
            query3 = int(dctqueries.sum())
            if predict2 == target_label:
                success2 = True
        queries = [query1, rcd.queries[0], query3]
        if success1 == True:
            logging.info('After {} queries to misclassify the video as the target class without simba-DCT'.format(sum(queries)))
        elif success2 == True:
            logging.info('After {} queries to misclassify the video as the target class'.format(sum(queries)))
        else:
            logging.info('fail with {} queries'.format(sum(queries)))
        success = [success1, success2]
        return perturb_style, logo_clip, adv_clip, success, queries