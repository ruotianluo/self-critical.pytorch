from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from collections import OrderedDict
import torch

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Bleu_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        if arr[i] >= 20000:
            continue
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    out = out.strip()
    import os
    if int(os.getenv('NO_EOS', 0)) == 1:
        if out[-2:] == ' 0':
            out = out[:-2]
    return out.strip()

def get_self_critical_reward(model, fc_feats, att_feats, att_masks, data, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    
    # get greedy decoding baseline
    model.eval()
    with torch.no_grad():
        greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')
    model.train()

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards



def get_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i // seq_per_img] for i in range(batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0

    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    return scores


# def get_scores(data, gen_result, opt):
#     batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
#     seq_per_img = batch_size // len(data['gts'])

#     res = OrderedDict()
    
#     gen_result = gen_result.data.cpu().numpy()
#     for i in range(batch_size):
#         res[i] = [array_to_str(gen_result[i])]

#     gts = OrderedDict()
#     for i in range(len(data['gts'])):
#         gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

#     res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
#     res__ = {i: res[i] for i in range(batch_size)}
#     gts = {i: gts[i // seq_per_img] for i in range(batch_size)}
#     if opt.cider_reward_weight > 0:
#         _, cider_scores = CiderD_scorer.compute_score(gts, res_)
#         print('Cider scores:', _)
#     else:
#         cider_scores = 0
#     if opt.bleu_reward_weight > 0:
#         _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
#         bleu_scores = np.array(bleu_scores[3])
#         print('Bleu scores:', _[3])
#     else:
#         bleu_scores = 0

#     scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores
#     if getattr(opt, 'force_short'):
#         scores -= np.array([abs(len(res[i][0].split(' ')) - 4) / 6 * opt.force_short for i in range(batch_size)])
#         print('Average length:', np.array([len(res[i][0].split(' '))-1 for i in range(batch_size)]).mean())
#     if getattr(opt, 'force_long'):
#         scores -= np.array([abs(len(res[i][0].split(' ')) - 16) / 6 * opt.force_long for i in range(batch_size)])
#         print('Average length:', np.array([len(res[i][0].split(' '))-1 for i in range(batch_size)]).mean())

#     if getattr(opt, 'grammar_rules'):
#         bad_ends = {'7961':'a', '2597':'an', '3029':'the', '290':'with', '4437':'of', 'on':'702', 'in':'3636'}
#         tmp = [res[i][0].split(' ') for i in range(batch_size)]
#         scores -= np.array([tmp[i][-2] in bad_ends if len(tmp[i]) > 2 else False for i in range(batch_size)]).astype('float32') * opt.grammar_rules
#     return scores