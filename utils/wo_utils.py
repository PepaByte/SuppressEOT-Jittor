
import torch
import jittor as jt
import random
from scipy.spatial.distance import cdist

CALC_SIMILARITY = False

def punish_wight(wo_batch, latent_size, alpha, method):
    if method == 'weight':
        wo_batch *= alpha
    elif method in ['alpha', 'beta', 'delete', 'soft-weight']:
        u, s, vh = jt.linalg.svd(wo_batch)
        u = u[:,:latent_size]
        zero_idx = int(latent_size * alpha)
        if method == 'alpha':
            s[:zero_idx] = 0
        elif method == 'beta':
            s[zero_idx:] = 0
        elif method == 'delete':
            s = s[zero_idx:] if zero_idx < latent_size else jt.zeros(latent_size).to(s.device)
            u = u[:, zero_idx:] if zero_idx < latent_size else u
            vh = vh[zero_idx:, :] if zero_idx < latent_size else vh
        elif method == 'soft-weight':
            if CALC_SIMILARITY:
                _s = s.clone()
                _s[zero_idx:] = 0
                _wo_batch = u @ jt.diag(_s) @ vh
                dist = cdist(wo_batch[:,0].unsqueeze(0).cpu(), _wo_batch[:,0].unsqueeze(0).cpu(), metric='cosine')
                print(f'The distance between the word embedding before and after the punishment: {dist}')
            if alpha == -.001:
                s *= (jt.exp(.001 * s) * 1.2)  # strengthen objects (our Appendix.F)
            else:
                s *= jt.exp(-alpha*s)  # suppression EOT (our main paper)

        wo_batch = u @ jt.diag(s) @ vh
    else:
        raise ValueError('Unsupported method')
    return wo_batch

def woword_eot_context(context, token_indices, alpha, method, n):
    for i, batch in enumerate(context):# i是指标，context是[1,77,768]的，现在提取出一列[77,768]tensor，以token为索引
        indices = token_indices + [num for num in range(n-1, 77)]# 取出neg列和EOT列索引
        wo_batch = batch[indices]# 提取特定token的embedding
        wo_batch = punish_wight(wo_batch.transpose(), len(indices), alpha, method).transpose()
        batch[indices] = wo_batch# 替换batch的内容
    return context

def woword_reweight(attn, token_indices, alpha):
    # if attn.shape[1] > 32 ** 2:  # avoid memory overhead
    #     return attn
    latent_size = int(attn.shape[1]**0.5)
    assert latent_size**2 == attn.shape[1]
    for head_attn in attn:
        for indice in token_indices:
            wo_attn = head_attn[:, indice].reshape(latent_size, latent_size)
            wo_attn *= alpha  # same as Reweight of P2P
            head_attn[:, indice] = wo_attn.reshape(latent_size**2)
    return attn