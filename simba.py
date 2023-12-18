import torch
from torch import nn
import numpy as np
from scipy.fftpack import idct
import torchvision.transforms as transforms
from torchvision.transforms import Resize
import logging
unloader = transforms.ToPILImage()
def expand_vector(x, size, logo_size):
    x = x.view(16, 3, size, size)
    z = torch.zeros(16, 3, logo_size, logo_size)
    z[:, :, :size, :size] = x
    return z

def block_idct(x, block_size=8, masked=False, ratio=0.5, linf_bound=0.0):
    z = torch.zeros(x.size()).cuda()
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    if type(ratio) != float:
        mask[:, :, :int(block_size * ratio[i]), :int(block_size * ratio[i])] = 1
    else:
        mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)].cpu().numpy()
            if masked:
                submat = submat * mask
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = torch.from_numpy(idct(idct(submat, axis=3, norm='ortho'), axis=2, norm='ortho')).cuda()
    if linf_bound > 0:
        return z.clamp(-linf_bound, linf_bound)
    else:
        return z

def simba_perturb(model, y, num, targeted = True, num_iters=1500, epsilon=0.1):
    n_dims = 3 * 6 * 6
    perm = torch.randperm(n_dims).cuda()
    y = y.expand(1, 1).contiguous()
    res = []
    queries = []
    n = 0
    transform = Resize((112, 112))
    while n < num:
        x = torch.rand(3, 4, 4).cuda()
        x = transform(x)
        x = x.expand(16, 3, 112, 112).contiguous()
        last_prob = model.get_target_val(x, y)
        if targeted is True and last_prob < 0.001:
            queries.append(1)
            continue
        query = 1
        for i in range(num_iters):
            diff = torch.zeros(n_dims).cuda()
            diff[perm[i % n_dims]] = epsilon
            perturbation = diff.view(3, 6, 6).clone()
            perturbation = transform(perturbation)
            perturbation = perturbation.expand(16, 3, 112, 112).contiguous()
            temp = x.clone()
            temp -= perturbation
            left_prob = model.get_target_val(temp.clamp(0, 1), y)
            query += 1
            if targeted != (left_prob < last_prob):
                x = temp.clamp(0, 1)
                last_prob = left_prob
            else:
                temp = x.clone()
                temp += perturbation
                right_prob = model.get_target_val(temp.clamp(0, 1), y)
                query += 1
                if targeted != (right_prob < last_prob):
                    x = temp.clamp(0, 1)
                    last_prob = right_prob
            _, pred, _ = model(x)
            pred = pred.squeeze(0)
            if targeted and pred == y:
                break
            if not targeted and pred != y:
                break
        queries.append(query)
        if (targeted and pred == y) or (not targeted and pred != y):
            res.append(x[0, :, :, :].squeeze(0))
    return res, sum(queries)

def simba_dct(model, clip, label, h, w, size, max_iters, freq_dims, epsilon, linf_bound=0.0,
                targeted=False, log_every=1):
    label = label.expand(1, 1).contiguous()

    expand_dims = freq_dims
    n_dims = 16 * 3 * expand_dims * expand_dims
    indices = torch.randperm(n_dims)[:max_iters]
    x = torch.zeros(n_dims).cuda()

    probs = torch.zeros(max_iters).cuda()
    succs = torch.zeros(max_iters).cuda()
    queries = torch.zeros(max_iters).cuda()
    l2_norms = torch.zeros(max_iters).cuda()
    linf_norms = torch.zeros(max_iters).cuda()
    prev_probs = model.get_target_val(clip, label)
    _, preds, _ = model(clip)
    
    trans = lambda z: block_idct(z, block_size=size, masked=False, ratio=0.5, linf_bound=linf_bound)

    for k in range(max_iters):
        expanded = clip.clone()
        expanded[:, :, h:h+size, w:w+size] = (expanded[:, :, h:h+size, w:w+size] + trans(expand_vector(x, expand_dims, size))).clamp(0, 1)
        perturbation = trans(expand_vector(x, expand_dims, size))
        l2_norms[k] = perturbation.view(-1).norm(2, 0)
        linf_norms[k] = perturbation.view(-1).abs().max(0)[0]
        _, preds_next, _ = model(expanded)
        preds = preds_next
        if targeted:
            remaining = preds.ne(label)
        else:
            remaining = preds.eq(label)

        if remaining.sum() == 0:
            adv = clip.clone()
            adv[:, :, h:h+size, w:w+size] = (adv[:, :, h:h+size, w:w+size] + trans(expand_vector(x, expand_dims, size))).clamp(0, 1)
            probs_k = model.get_target_val(adv, label)
            probs[k:] = probs_k.squeeze(0).repeat(max_iters - k)
            succs[k:] = torch.ones(max_iters - k).cuda()
            queries[k:] = torch.zeros(max_iters - k).cuda()
            break
        if k > 0:
            succs[k-1] = ~remaining
        
        
        dim = torch.randint(0, 3 * expand_dims * expand_dims, (1, )).cuda()
        for i in range(15):
            dim = torch.cat((dim, torch.randint((i + 1)* 3 * expand_dims * expand_dims, (i + 2) * 3 * expand_dims * expand_dims, (1, )).cuda()), 0)
        
        queries_k = 0
        probs_k = prev_probs.clone()
        
        for i in range(16):
            diff = torch.zeros(n_dims).cuda()
            direct = torch.randint(0, 2, (16, )).cuda()
            direct[direct == 0] = -1
            diff[dim] = direct * epsilon * 0.25
            vec = x + diff
            adv = clip.clone()
            adv[:, :, h:h+size, w:w+size] = (adv[:, :, h:h+size, w:w+size] + trans(expand_vector(vec, expand_dims, size))).clamp(0, 1)
            vec_probs = model.get_target_val(adv, label)

            queries_k += 1
            if targeted:
                improved = vec_probs.gt(prev_probs)
            else:
                improved = vec_probs.lt(prev_probs)
            if improved.sum() > 0:
                x = vec
                probs_k = vec_probs
                break

        probs[k] = probs_k
        queries[k] = queries_k
        prev_probs = probs[k]
        if targeted == True and k >= int(n_dims/16) and (probs[k] - probs[k - int(n_dims/2)]) < 0.005 and probs[k] < 0.05:
            logging.info('prob increase too low, early interuption')
            break
        if targeted == False and k >= int(n_dims/16) and (probs[k - int(n_dims/2)] - probs[k]) < 0.05 and probs[k] > 0.9:
            logging.info('prob decrease too low, early interuption')
            break
        if (k + 1) % log_every == 0 or k == max_iters - 1:
            logging.info('Iteration {}: queries = {}, prob = {}'.format(
                    k + 1, queries.sum(0), probs[k]))
    expanded = clip.clone()
    expanded[:, :, h:h+size, w:w+size] = (expanded[:, :, h:h+size, w:w+size] + trans(expand_vector(x, expand_dims, size))).clamp(0, 1)
    _, preds, _ = model(expanded)
    if targeted:
        remaining = preds.ne(label)
    else:
        remaining = preds.eq(label)
    succs[max_iters-1] = ~remaining
    return expanded, probs, succs, queries, l2_norms, linf_norms