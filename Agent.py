import cv2
import time
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Resize
from torch.distributions.categorical import Categorical
from torch.utils.data import TensorDataset, DataLoader
import utils
from torchvision import transforms
from style_transfer import style_perturb_logo
# global variables
eps = np.finfo(np.float32).eps.item()

class robot():
    class p_pi(nn.Module):
        """
        policy network
        """
        def __init__(self, space, embedding_size=30, stable=True):
            super().__init__()
            self.embedding_size = embedding_size
            embedding_space = [space[-1]] + space[:-1]
            # create embedding space
            self.embedding_list = nn.ModuleList([nn.Embedding(embedding_space[i], self.embedding_size) for i in range(len(embedding_space))])
            if stable:
                self._stable_first_embedding()
            # create linear heads
            self.lstm = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)  # (batch, seq, features)
            self.linear_list = nn.ModuleList([nn.Linear(self.embedding_size, space[i]) for i in range(len(space))])
            # set necessary parameters
            self.stage = 0
            self.hidden = None

        def forward(self, x):
            x = self.embedding_list[self.stage](x)
            # extract feature of current state
            self.lstm.flatten_parameters()
            x, self.hidden = self.lstm(x, self.hidden)  # hidden: hidden state plus cell state
            # get action prob given the current state
            prob = self.linear_list[self.stage](x.view(x.size(0), -1))
            return prob

        def increment_stage(self):
            self.stage += 1

        def _stable_first_embedding(self):
            target = self.embedding_list[0]
            for param in target.parameters():
                param.requires_grad = False

        def reset(self):
            """
            reset stage to 0
            clear hidden state
            """
            self.stage = 0
            self.hidden = None

    def __init__(self, space, rl_batch, gamma, lr, stable=True):
        # policy network
        self.mind = self.p_pi(space, stable=stable)
        # reward setting
        self.gamma = gamma  # back prop rewards
        # optimizer
        self.optimizer = optim.Adam(self.mind.parameters(), lr=lr)
        # useful parameters
        self.combo_size = len(space)
        self.rl_batch = rl_batch

    def select_action(self, state):
        p_a = F.softmax(self.mind(state), dim=1)
        # select action with prob
        dist = Categorical(probs=p_a)
        action = dist.sample()
        log_p_action = dist.log_prob(action)
        return action.unsqueeze(-1), log_p_action.unsqueeze(-1)

    def select_combo(self):
        state = torch.zeros((self.rl_batch, 1)).long().cuda()
        combo = []
        log_p_combo = []
        for _ in range(self.combo_size):
            action, log_p_action = self.select_action(state)
            combo.append(action)
            log_p_combo.append(log_p_action)
            state = action
            self.mind.module.increment_stage()
        combo = torch.cat(combo, dim=1)
        log_p_combo = torch.cat(log_p_combo, dim=1)
        return combo, log_p_combo


class agent(robot):
    def __init__(self, model, clip_tensor, target_tensor, sigma, tau, logos, styles, target, target_label, shrink=1):
        """
        the __init__ function needs to create action space because this relates with
        the __init__ of the policy network
        """
        # build environment
        self.model = model
        self.clip_tensor = clip_tensor
        self.target_tensor = target_tensor
        self.logos = logos
        self.styles = styles

        self.shrink = shrink
        self.H = clip_tensor.size(-2)
        self.W = clip_tensor.size(-1)
        self.ks = [0.75, 0.8125, 0.875, 0.9375, 1.0] # the scaling ratio of logo size
        self.space = self.create_searching_space(self)

        self.sigma = sigma
        self.tau = tau

        self.target = target
        self.target_label = target_label

    def build_robot(self, rl_batch, gamma, lr, stable=True):
        super().__init__(self.space, rl_batch, gamma, lr, stable)

    @staticmethod
    def create_searching_space(self):
        W = self.W
        H = self.H
        if W > H:
            W = W // self.shrink
        elif H > W:
            H = H // self.shrink
        # create space
        h = self.logos[-1].size(-2)
        w = self.logos[-1].size(-1)
        search_space = [int(H - 0.8 * h // self.shrink), int(W - 0.8 * w // self.shrink), len(self.ks),len(self.logos), len(self.styles)]
        return search_space

    @staticmethod
    def create_logo(self, clip, combo, C=3, T=16, H=112, W=112):
        """
        clip: torch.floattensor with size (bs, 16, 3, 112, 112)
        return:
        logo_clip: torch.floattensor with size (bs, 16, 3, 112, 112)
        area: torch.floattensor with size (bs, 1)
        """
        bs = combo.size(0)
        combo_size = combo.size(-1)
        # post process combo
        pos_combo = combo[:, 0:3]
        logo_combo = torch.index_select(combo, dim=1, index=torch.LongTensor([3]).cuda())
        style_combo = torch.index_select(combo, dim=1, index=torch.LongTensor([4]).cuda())

        logo_clip = clip[:bs, :, :, :, :].clone().detach()
        area = torch.zeros(bs, 1)
        dis = torch.zeros(bs, 1)
        # make masks
        for item in range(bs):
            for t in range(T):
                k = self.ks[pos_combo[item, 2]]
                h = int(k * self.logos[logo_combo[item, 0]].size(-2))
                w = int(k * self.logos[logo_combo[item, 0]].size(-1))
                logo_clip[item, t, : ,
                                    pos_combo[item, 0]: 
                                    pos_combo[item, 0] + h,
                                    pos_combo[item, 1]: 
                                    pos_combo[item, 1] + w]\
                = style_perturb_logo(logo_combo[item, 0], self.styles[style_combo[item, 0]], h, w)
            area[item, 0] = h * w
            dis[item, 0] = min(utils.get_l2dis(0, 0, pos_combo[item, 0], pos_combo[item, 1]), \
                            utils.get_l2dis(0, self.W, pos_combo[item, 0], pos_combo[item, 1] + w), \
                            utils.get_l2dis(self.H, 0, pos_combo[item, 0] + h, pos_combo[item, 1]), \
                            utils.get_l2dis(self.H, self.W, pos_combo[item, 0] + h, pos_combo[item, 1] + w))
        return logo_clip, area, dis

    @staticmethod
    def get_reward(self, model, logo_input_tensor, target_tensor, target_label, area, dis):
        """
        input:
        model: utils.agent.model
        mask_input_tensor: torch.floattensor with size (bs, 16, 3, 112, 112)
        target_tensor: torch.longtensor with size (bs, 1)
        area: torch.floattensor with size (bs, 1)
        sigma: controls penalization for the area, the smaller, the more powerful
        return:
        reward: torch.floattensor with size (bs, 1)
        acc: list of accs, label_acc and target_acc [default None]
        """
        with torch.no_grad():
            deal_dataset = TensorDataset(logo_input_tensor.cpu(), target_tensor.cpu())
            deal_dataloader = DataLoader(deal_dataset, batch_size=deal_dataset.__len__(), shuffle=False, pin_memory=True)
            for deal_data in deal_dataloader:
                masked_input_tensor, _ = deal_data
            prob, pred, _ = model(masked_input_tensor)
            
            target_tensor, target_label, area, dis = target_tensor.cuda(), target_label.cuda(), area.cuda(), dis.cuda()
            label_filter = (pred==target_tensor).view(-1)
            target_filter = None
            label_acc = label_filter.float().mean()
            target_acc = None
            if self.target == False:
                p_cl = 1. - prob
            else:
                p_cl = model.get_target_val(masked_input_tensor, target_label)
                target_filter = (pred==target_label).view(-1)
                target_acc = target_filter.float().mean()
            reward = torch.log(p_cl + eps) - area * self.sigma - dis * self.tau
            acc = [label_acc, target_acc]
            filters = [label_filter, target_filter]
            return reward, acc, filters

    @staticmethod
    def reward_backward(rewards, gamma):
        """
        input:
        reward: torch.floattensor with size (bs, something)
        gamma: discount factor

        return:
        updated_reward: torch.floattensor with the same size as input
        """
        gamma = 1
        R = 0
        updated_rewards = torch.zeros(rewards.size()).cuda()
        for i in range(rewards.size(-1)):
            R = rewards[:, -(i + 1)] + gamma * R
            updated_rewards[:, -(i + 1)] = R
        return updated_rewards

    def reinforcement_learn(self, steps=150, baseline_subtraction=False):
        """
        input:
        steps: the steps to interact with the environment for the agent
        baseline_subtraction: flag to use baseline subtraction technique.
        return:
        floating_logo_clip: torch.floattensor with size (3, 16, 112, 112)
        area: torch.floattensor with size (1)
        """
        C = self.clip_tensor.size(-3)
        T = self.clip_tensor.size(-4)
        H = self.clip_tensor.size(-2)
        W = self.clip_tensor.size(-1)
        queries = 0
        
        clip_batch = self.clip_tensor.expand(self.rl_batch, self.clip_tensor.size(-4), self.clip_tensor.size(-3), self.clip_tensor.size(-2), self.clip_tensor.size(-1)).contiguous()
        target_batch = self.target_tensor.expand(self.rl_batch, 1).contiguous()
        self.mind.cuda()
        self.mind = nn.DataParallel(self.mind, device_ids=None)
        self.mind.train()
        self.optimizer.zero_grad()
        # set up non-target attack records
        floating_logo_clip = None
        t_floating_logo_clip = None
        floating_dis = torch.Tensor([1.414 * H * W]).cuda()
        t_floating_dis = torch.Tensor([1.414 * H * W]).cuda()
        floating_combo = None
        t_floating_combo = None
        # start learning, interacting with the environments
        orig_r_record = []
        for s in range(steps):
            # make combo and get reward
            logging.info('RL step:{}'.format(s))
            combo, log_p_combo = self.select_combo()
            v_combo = []
            v_log_p_combo = []
            for i in range(combo.size(0)):
                if combo[i, 0] + int(self.ks[combo[i, 2]] * self.logos[combo[i, 3]].size(-2)) < self.H and combo[i, 1] + int(self.ks[combo[i, 2]] * self.logos[combo[i, 3]].size(-1)) < self.W:
                    v_combo.append(combo[i])
                    v_log_p_combo.append(log_p_combo[i])
            v_combo = torch.stack(v_combo)
            v_log_p_combo = torch.stack(v_log_p_combo)
            rewards = torch.zeros(v_combo.size()).cuda()
            logo_clip_batch, area, dis = self.create_logo(self, clip_batch, v_combo, T=T, H=H, W=W)
            target_batch = self.target_tensor.expand(logo_clip_batch.size(0), 1).contiguous()
            target_label_batch = self.target_label.expand(logo_clip_batch.size(0), 1).contiguous()
            r, acc, filters = self.get_reward(self, self.model, logo_clip_batch, target_batch, target_label_batch, area, dis)
            queries += logo_clip_batch.size(0)
            orig_r_record.append(r.mean())
            rewards[:, -1] = r.squeeze(-1)
            rewards = self.reward_backward(rewards, self.gamma)
            # update records
            wrong_filter = ~filters[0]

            dis, area = dis.cuda(), area.cuda()
            if self.target == False and (1 - acc[0]) > 1e-4:
                dis_candidate = dis[wrong_filter]
                temp_floating_dis, temp = dis_candidate.min(dim=0)
                if temp_floating_dis < floating_dis:
                    floating_logo_clip = logo_clip_batch[wrong_filter][temp].squeeze(0)
                    floating_dis = temp_floating_dis
                    floating_combo = v_combo[wrong_filter][temp].squeeze(0)
                if temp_floating_dis <= 15. and floating_combo[2] < 3 :
                        break
            if self.target == True and acc[1] != 0:
                target_filter = filters[1]
                dis_candidate = dis[target_filter]
                t_temp_floating_dis, temp = dis_candidate.min(dim=0)
                if t_temp_floating_dis < t_floating_dis:
                    t_floating_logo_clip = logo_clip_batch[target_filter][temp].squeeze(0)
                    t_floating_dis = t_temp_floating_dis
                    t_floating_combo = v_combo[target_filter][temp].squeeze(0)
                    if t_temp_floating_dis <= 15. and t_floating_combo[2] < 3:
                        break
            # baseline subtraction
            if baseline_subtraction:
                rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
            # calculate loss
            loss = (-v_log_p_combo * rewards).sum(dim=1).mean()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer.zero_grad()
            # reset mind to continuously interact with the environment
            self.mind.module.reset()
            if self.target == True:
                if s >= 2:
                    if abs(orig_r_record[-1] + orig_r_record[-3] - 2*orig_r_record[-2]) < 1e-4:
                        prob = self.model.get_target_val(logo_clip_batch, target_label_batch)
                        _, tmp = prob.max(dim=0)
                        t_floating_logo_clip = logo_clip_batch[tmp].squeeze(0)
                        t_floating_dis = dis[tmp]
                        t_floating_combo = v_combo[tmp].squeeze(0)
                        break
                if s == steps - 1 and t_floating_logo_clip == None:
                    prob = self.model.get_target_val(logo_clip_batch, target_label_batch)
                    _, tmp = prob.max(dim=0)
                    t_floating_logo_clip = logo_clip_batch[tmp].squeeze(0)
                    t_floating_dis = dis[tmp]
                    t_floating_combo = v_combo[tmp].squeeze(0)
            else:
                if s >= 2:
                    if abs(orig_r_record[-1] + orig_r_record[-3] - 2*orig_r_record[-2]) < 1e-4 and floating_logo_clip == None:
                        _, prob, _ = self.model(logo_clip_batch)
                        _, tmp = prob.min(dim=0)
                        floating_logo_clip = logo_clip_batch[tmp].squeeze(0)
                        floating_dis = dis[tmp]
                        floating_combo = v_combo[tmp].squeeze(0)
                        break
                if s == steps - 1 and floating_logo_clip == None:
                    _, prob, _ = self.model(logo_clip_batch)
                    _, tmp = prob.min(dim=0)
                    floating_logo_clip = logo_clip_batch[tmp].squeeze(0)
                    floating_dis = dis[tmp]
                    floating_combo = v_combo[tmp].squeeze(0)
        if self.target:
            return t_floating_logo_clip, t_floating_dis, t_floating_combo, queries
        else:
            return floating_logo_clip, floating_dis, floating_combo, queries

    @staticmethod
    def attack(model, input_tensor, target_tensor, sigma, tau,  logos, styles, lr=0.03, baseline_subtraction=True, 
               color=False, num_occlu=4, rl_batch=500, steps=50, target = False, target_label = torch.LongTensor([0])):
        """
        input:
        model: pytorch model
        input_tensor: torch.floattensor with size (3, 16, 112, 112)
        target_tensor: torch.longtensor
        sigma: scalar, contrain the area of the occlusion
        lr: learning rate for p_pi, scalar
        baseline_subtraction: flag to use reward normalization
        color: flag to search the RGB channel values
        return:
        mask_input_tensor: torch.floattensor with size (3, 16, 112, 112)
        area: scalar with size (1)
        """
        # time to start
        attack_begin = time.time()
        actor = agent(model, input_tensor, target_tensor, sigma, tau, logos, styles, target, target_label)
        actor.build_robot(rl_batch=rl_batch, gamma=1, lr=lr, stable=True)
        mask_input_tensor, dis, combo, queries = actor.reinforcement_learn(steps=steps, baseline_subtraction=baseline_subtraction)
        return mask_input_tensor, dis, combo, queries, time.time() - attack_begin