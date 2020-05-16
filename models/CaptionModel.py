# This file contains ShowAttendTell and AllImg model

# ShowAttendTell is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

# AllImg is a model where
# img feature is concatenated with word embedding at every time step as the input of lstm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
import os


class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    # implements beam search
    # calls beam_step and returns the final set of beams
    # augments log-probabilities with diversity terms when number of groups > 1

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_'+mode)(*args, **kwargs)

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf

        # does one step of classical beam search

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, trigrams):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            _logprobsf = logprobsf.clone()
            has_to_end = None
            if hasattr(logprobsf, 'eos_change'):
                tmp = logprobsf.new_zeros(logprobsf.size())
                tmp[logprobsf.eos_change == 1, 0] = float('-inf')
                tmp[logprobsf.eos_change == 2, 1:] = float('-inf')
                # If has_to_end the cols has to be 1
                has_to_end = logprobsf.eos_change == 2
                _logprobsf = _logprobsf + tmp
                # _logprobsf[logprobsf.eos_change == 1, 0] = float('-inf')
                # _logprobsf[logprobsf.eos_change == 2, 0] = _logprobsf[logprobsf.eos_change == 2, 0] + 10000
            if self.decide_length != 'none' or self.__class__.__name__ == 'LenEmbModel':
                logprobsf = _logprobsf

            if trigrams is not None and t >= 3+int(self.decide_length != 'none'):
                assert (beam_seq[t-3:t-1] < 20000).all() # no length token
                # Store trigram generated at last step
                prev_two_batch = beam_seq[t-3:t-1].t()
                for i in range(beam_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = beam_seq[t-1][i]
                    if t == 3+int(self.decide_length != 'none'): # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3+int(self.decide_length != 'none'):
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = beam_seq[t-2:t].t()
                mask = torch.zeros_like(logprobsf) # batch_size x vocab_size
                for i in range(beam_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                # logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)
                _logprobsf = _logprobsf + (mask * -0.693 * alpha)
                logprobsf = logprobsf + (mask * -0.693 * alpha)

            ys,ix = torch.sort(_logprobsf,1,True)
            ys = logprobsf.gather(1, ix)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            # Because desired_length are the same for each beam
            # to avoid duplicate, we constrain to one row
            elif t == 1:
                if hasattr(self, 'desired_length'): # nasty
                    # print('')
                    rows = 1
                elif len(set(beam_seq[t-1].tolist())) == 1:
                    assert int(os.getenv('SAME_LEGNTH', '0')) == 1
                    rows = 1
            for q in range(rows): # for each beam expansion
                for c in range(cols): # for each column (word, essentially)
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q,c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    # local_unaug_logprob = unaug_logprobsf[q,ix[q,c]]
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':unaug_logprobsf[q]})
                    if has_to_end is not None and has_to_end[q]:
                        # col==1 is eos, and ignore other words
                        # if including other words, they will contaminate the beams.
                        break

            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.clone() for _ in state]
            new_trigrams = [None for _ in trigrams] if trigrams else trigrams

            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
                # Rearrage trigrams accordingly
                if trigrams:
                    new_trigrams[vix] = trigrams[v['q']]
            state = new_state
            trigrams = new_trigrams
            return beam_seq,beam_seq_logprobs,beam_logprobs_sum,state,trigrams,candidates

        # Start diverse_beam_search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        max_ppl = opt.get('max_ppl', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        block_trigrams = opt.get('block_trigrams', 0) or int(os.getenv('BLOCK_TRIGRAMS', '0'))
        trigrams_table = [[] if block_trigrams else None for _ in range(group_size)]
        bdash = beam_size // group_size # beam per group

        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(self.seq_length+int(self.decide_length != 'none'), bdash).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length+int(self.decide_length != 'none'), bdash, self.vocab_size + 1).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[] for _ in range(group_size)]
        # state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        state_table = list(zip(*[_.chunk(group_size, 1) for _ in init_state]))
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        if hasattr(init_logprobs, 'eos_change'):
            for a,b in zip(logprobs_table, list(init_logprobs.eos_change.chunk(group_size, 0))): a.eos_change = b
        # END INIT

        # Chunk elements in the args
        args = list(args)
        args = [_.chunk(group_size) if _ is not None else [None]*group_size for _ in args]
        args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.seq_length +int(self.decide_length != 'none')+ group_size - 1):
            for divm in range(group_size): 
                if t >= divm and t <= self.seq_length +int(self.decide_length != 'none')+ divm - 1:
                    # add diversity
                    logprobsf = logprobs_table[divm]
                    # suppress previous word
                    if decoding_constraint and t-divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t-divm-1].unsqueeze(1).cuda(), float('-inf'))
                    # suppress UNK tokens in the decoding
                    logprobsf[:,logprobsf.size(1)-1] = logprobsf[:, logprobsf.size(1)-1] - 1000  
                    # diversity is added here
                    # the function directly modifies the logprobsf values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    unaug_logprobsf = add_diversity(beam_seq_table,logprobsf,t,divm,diversity_lambda,bdash)

                    # infer new beams
                    beam_seq_table[divm],\
                    beam_seq_logprobs_table[divm],\
                    beam_logprobs_sum_table[divm],\
                    state_table[divm],\
                    trigrams_table[divm],\
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                bdash,
                                                t-divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm],
                                                trigrams_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    for vix in range(bdash):
                        if beam_seq_table[divm][t-divm,vix] == 0 or t == self.seq_length + int(self.decide_length != 'none') + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(), 
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(),
                                'p': beam_logprobs_sum_table[divm][vix].item()
                            }
                            final_beam['p'] = length_penalty(t-divm+1, final_beam['p'])
                            # if max_ppl:
                            #     final_beam['p'] = final_beam['p'] / (t-divm+1)
                            done_beams_table[divm].append(final_beam)
                            # don't continue beams from finished sequences
                            beam_logprobs_sum_table[divm][vix] = -1000

                    # move the current group one step forward in time
                    
                    it = beam_seq_table[divm][t-divm].clone()
                    if t-divm == 0: # length token
                        if getattr(self, 'decide_length', 'none') != 'none':
                            beam_seq_table[divm][t-divm] += 20000
                            state = list(state_table[divm])
                            if hasattr(self, 'desired_length'):
                                # Add current length to state
                                # state = list(state[:2]) + [state[0].new_tensor([self.desired_length]).unsqueeze(0).unsqueeze(2).expand(state[0].shape)]
                                # print('This should have been done in sample_beam')
                                pass
                            else:
                                state = list(state[:2]) + [it.to(state[0].device).unsqueeze(0).unsqueeze(2).expand(state[0].shape)]
                            if 'marker' in self.decide_length:
                                if hasattr(self, 'desired_length'):
                                    # since there is desired_length
                                    # the state must already contain length
                                    it.copy_(state[-1][0][:,0])
                                it.add_(20000)
                            else: # init, lenemb_predict
                                if self.decide_length == 'init': # init, lenemb_predict
                                    if os.getenv('LEN_INIT_SANITY') is not None:
                                        it.fill_(0)
                                    if not hasattr(self, 'desired_length'):
                                        state = [self.memory.reshape(1,1,-1).expand_as(state[0]) * it.unsqueeze(1).float().to(self.memory.device)] + list(state)[1:]
                                    else:
                                        # if there is desired_length, state[-1] should be the length
                                        state = [self.memory.reshape(1,1,-1).expand_as(state[0]) * state[-1][0][:, 0].unsqueeze(1).float()] + list(state)[1:]
                                it.fill_(0)
                            state_table[divm] = state
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it.cuda(), *(args[divm] + [state_table[divm]]))

        # all beams are sorted by their log-probabilities
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = sum(done_beams_table, [])
        return done_beams