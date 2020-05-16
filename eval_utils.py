from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import eval_multi

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0

def language_eval(dataset, preds, preds_n, eval_kwargs, split):
    model_id = eval_kwargs['id']
    eval_oracle = eval_kwargs.get('eval_oracle', 0)
    
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    mean_perplexity = sum([_['perplexity'] for _ in preds_filt]) / len(preds_filt)
    mean_entropy = sum([_['entropy'] for _ in preds_filt]) / len(preds_filt)
    mean_length = sum([len(_['caption'].split(' ')) for _ in preds_filt]) / len(preds_filt)
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # Add mean perplexity
    out['perplexity'] = mean_perplexity
    out['entropy'] = mean_entropy
    out['length'] = mean_length

    imgToEval = cocoEval.imgToEval
    for k in list(imgToEval.values())[0]['SPICE'].keys():
        if k != 'All':
            out['SPICE_'+k] = np.array([v['SPICE'][k]['f'] for v in imgToEval.values()])
            out['SPICE_'+k] = (out['SPICE_'+k][out['SPICE_'+k]==out['SPICE_'+k]]).mean()
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    if len(preds_n) > 0:
        cache_path_n = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '_n.json')
        spice_n = eval_multi.eval_spice_n(preds_n, model_id, split)
        out.update(spice_n['overall'])
        div_stats = eval_multi.eval_div_stats(preds_n, model_id, split)
        out.update(div_stats['overall'])
        if eval_oracle:
            oracle = eval_multi.eval_oracle(preds_n, model_id, split)
        out.update(oracle['overall'])
        with open(cache_path_n, 'w') as outfile:
            json.dump({'spice_n': spice_n, 'div_stats': div_stats, 'oracle': oracle}, outfile)
        
    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    sample_n_method = eval_kwargs.get('sample_n_method', 'sample')
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # Use this nasty way to make other code clean since it's a global configuration

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    n_predictions = [] # when sample_n > 1
    if os.getenv('LENGTH_PREDICT'):
        lenpred_acc = 0
        lenpred_accs = [0 for _ in range(20)]
        lenpred_l2 = 0
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            with torch.no_grad():
                loss = crit(model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:]).item()

                # if os.getenv('LENGTH_PREDICT'):
                #     gen_result, sample_logprobs = model(fc_feats, att_feats, att_masks, opt={'sample_max':0, 'len_pred':1, 'block_trigrams': 1}, mode='sample')
                        
                #     states = model.states
                #     model.states = []

                #     def length_loss(input, seq):
                        
                #         input = input + [input[-1]] * (model.seq_length-len(input))
                #         input = input[:model.seq_length]
                #         input = torch.stack(input, dim=1)
                #         # input = input.reshape(-1, input.shape[-1])
                #         mask = (seq>0).float()
                #         mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)

                #         target = (mask.sum(1,keepdim=True) - mask.cumsum(1)).long()

                #         if 'cls' in os.getenv('LENGTH_PREDICT'): # classificaiton
                #             # import pdb;pdb.set_trace()
                #             output = torch.nn.functional.cross_entropy(\
                #                 input.reshape(-1, input.shape[-1]),
                #                 target.reshape(-1),reduction='none') * mask.reshape(-1)
                #             lenpred_prob = input.softmax(2).gather(2, target.unsqueeze(2)).squeeze(2)
                #             _acc = (((input.max(-1)[1] == target).float() * mask).sum() / torch.sum(mask)).item()
                #             _accs = [0 for _ in range(20)]
                #             for t in range(min(20, input.shape[1])):
                #                 _accs[t] += (((input[:,t].max(-1)[1] == target[:,t]).float() * mask[:,t]).sum() / torch.sum(mask[:,t])).item()
                #             _reg = (((input.max(-1)[1].float() - target.float()).pow(2) * mask).sum() / torch.sum(mask)).item()
                #             print('length_prediction:', _acc)
                #             print('regression:', _reg)
                #         elif 'reg' in os.getenv('LENGTH_PREDICT'): # regression
                #             output = torch.nn.functional.mse_loss(\
                #                 input.reshape(-1),
                #                 target.reshape(-1).float(),reduction='none') * mask.reshape(-1)
                            
                #             _acc = (((input.squeeze(-1).round().long() == target).float() * mask).sum() / torch.sum(mask)).item()
                #             _accs = [0 for _ in range(20)]
                #             _reg = (((input.squeeze(-1) - target.float()).pow(2) * mask).sum() / torch.sum(mask)).item()
                #             print('length_prediction:', _acc)
                #             print('regression:', _reg)
                #             lenpred_prob = 0
                #         else:
                #             raise Exception
                #         output = torch.sum(output) / torch.sum(mask)

                #         return _acc, _accs, _reg, lenpred_prob

                #     _acc, _accs, _reg, ___ = length_loss(states, gen_result)
                #     lenpred_acc += _acc
                #     lenpred_accs = [lenpred_accs[_] + _accs[_] for _ in range(20)]
                #     lenpred_l2 += _reg
            
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            if os.getenv('DESIRED_LENGTH'):
                # import pdb;pdb.set_trace()
                desired_sents = [{} for _ in range(len(fc_feats))]
                for desired_length in os.getenv('DESIRED_LENGTH').split(','):
                    model.vocab = loader.get_vocab()
                    model.vocab['0'] = '<eos>'
                    model.desired_length = int(desired_length)
                    seq, __ = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')
                    _sents = utils.decode_sequence(loader.get_vocab(), seq, remove_bad_endings)
                    for _s in _sents:
                        print(_s)
                    for k, _s in enumerate(_sents):
                        desired_sents[k][int(desired_length)] = _s
                    del model.desired_length
                    model.states = []
            # temporary
            if os.getenv('GOLDEN_DESIRED', 'none') != 'none':
                if os.getenv('GOLDEN_DESIRED') == 'MEDIAN':
                    model.desired_length = torch.tensor([np.median((gt>0).sum(1)+1) for gt in data['gts']]).long()
                elif os.getenv('GOLDEN_DESIRED') == 'MAX':
                    model.desired_length = torch.tensor([np.max((gt>0).sum(1)+1) for gt in data['gts']]).long()
                elif os.getenv('GOLDEN_DESIRED') == 'MIN':
                    model.desired_length = torch.tensor([np.min((gt>0).sum(1)+1) for gt in data['gts']]).long()
                elif os.getenv('GOLDEN_DESIRED').isdigit():
                    model.desired_length = torch.tensor([int(os.getenv('GOLDEN_DESIRED')) for gt in data['gts']]).long()

            if eval_kwargs['beam_size'] == 1 and os.getenv('LENGTH_PREDICT'):
                eval_kwargs.update({'len_pred':1})
                seq, seq_logprobs = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')
                _acc, _accs, _reg, lenpred_prob = length_loss(model.states, seq)
                model.states = []
            else:
                seq, seq_logprobs = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')
                model.states = []
                lenpred_prob = torch.zeros(seq.shape)

            seq = seq.data
            eos_prob = F.softmax(seq_logprobs, dim=2)[:,:,0]
            tmp_seq_logprobs = seq_logprobs.clone(); tmp_seq_logprobs[torch.isinf(tmp_seq_logprobs)] = -1e10
            entropy = - (F.softmax(seq_logprobs, dim=2) * tmp_seq_logprobs).sum(2).sum(1) / ((seq>0).float().sum(1)+1)
            
            perplexity = - seq_logprobs.gather(2, (seq % 20000).unsqueeze(2)).squeeze(2).sum(1) / ((seq>0).float().sum(1)+1)
        
        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'],\
                     'caption': sent,
                     'eos_prob': eos_prob[k].tolist(),
                     'lenpred_prob': lenpred_prob[k].tolist(),
                     'perplexity': perplexity[k].item(),
                     'entropy': entropy[k].item()}
            if getattr(model, 'decide_length', 'none') != 'none':
                # print('seq[:, 0] should be length +20000')
                if hasattr(model, 'desired_length'):
                    # this condition only satifies when golden is provided
                    # DESIRED_LENGTH will delete model.desired_length
                    assert os.getenv('GOLDEN_DESIRED', 'none') != 'none'
                    try:
                        assert (seq[:, 0] - 20000 == model.desired_length).all()
                    except:
                        pass
                        # It's fine to be not equal, because seq saved what the model chose
                        # to be the best, but not the desired length
                        # The state or input would have been changed, but it's not 
                        # reflected in seq.
                    entry.update({'desired_length': model.desired_length[k].item() if torch.is_tensor(model.desired_length) else model.desired_length})
                else:
                    # import pudb;pu.db
                    # print('This is when the model predict its own desired length')
                    entry.update({'desired_length': seq[k][0].item() - 20000})
            else:
                # if none decide_length
                if hasattr(model, 'desired_length'):
                    # this condition only satifies when golden is provided
                    # DESIRED_LENGTH will delete model.desired_length
                    assert os.getenv('GOLDEN_DESIRED', 'none') != 'none'
                    entry.update({'desired_length': model.desired_length[k].item() if torch.is_tensor(model.desired_length) else model.desired_length})
                else:
                    # it can only be a2i2!!! because lenembadd and lenemb has to have desired_length
                    # import pudb;pu.db
                    print('what should I do here?')
                    print('do nothing')

            if os.getenv('DESIRED_LENGTH'):
                entry.update({'desired_caption': desired_sents[k]})
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        if sample_n > 1:
            tmp_eval_kwargs = eval_kwargs.copy()
            if sample_n_method == 'bs':
                # case 1 sample_n == beam size
                tmp_eval_kwargs.update({'beam_size': sample_n, 'group_size': 1}) # randomness from softmax
                with torch.no_grad():
                    model(fc_feats, att_feats, opt=tmp_eval_kwargs, mode='sample')
                for k in range(loader.batch_size):
                    _sents = utils.decode_sequence(loader.get_vocab(), torch.stack([model.done_beams[k][_]['seq'] for _ in range(beam_size)]))
                    for sent in _sents:
                        entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                        n_predictions.append(entry)
            # case 2 sample_max =0 temperature xx
            elif sample_n_method == 'sample':
                tmp_eval_kwargs.update({'sample_max': 0, 'beam_size': 1}) # randomness from sample
                with torch.no_grad():
                    _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
                _sents = utils.decode_sequence(loader.get_vocab(), _seq)
                for k, sent in enumerate(_sents):
                    entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent}
                    n_predictions.append(entry)
            # case 3 gumbel max
            elif sample_n_method == 'gumbel':
                tmp_eval_kwargs.update({'sample_max': 2, 'beam_size': 1}) # randomness from sample
                with torch.no_grad():
                    _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
                _sents = utils.decode_sequence(loader.get_vocab(), _seq)
                for k, sent in enumerate(_sents):
                    entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent}
                    n_predictions.append(entry)
            else:
                # Use diverse beam search
                tmp_eval_kwargs.update({'beam_size': sample_n * beam_size, 'group_size': sample_n}) # randomness from softmax
                with torch.no_grad():
                    model(fc_feats, att_feats, opt=tmp_eval_kwargs, mode='sample')
                for k in range(loader.batch_size):
                    _sents = utils.decode_sequence(loader.get_vocab(), torch.stack([model.done_beams[k][_]['seq'] for _ in range(0, sample_n*beam_size, beam_size)]))
                    for sent in _sents:
                        entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                        n_predictions.append(entry)
            if verbose:
                for entry in sorted(n_predictions[-loader.batch_size * sample_n:], key=lambda x: x['image_id']):
                    print('image %s: %s' %(entry['image_id'], entry['caption']))
            
        
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    torch.save((predictions, n_predictions), os.path.join('eval_results/', '.saved_pred_'+ eval_kwargs['id'] + '_' + split + '.pth'))
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, n_predictions, eval_kwargs, split)

    # Switch back to training mode
    model.train()

    if os.getenv('LENGTH_PREDICT'):
        print('Length prediction acc:', lenpred_acc/loss_evals)
        print('Length prediction accs:', [_/loss_evals for _ in lenpred_accs])
        print('Length prediction l2:', lenpred_l2/loss_evals)
    return loss_sum/loss_evals, predictions, lang_stats
