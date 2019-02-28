from AttModel import *


class _LenEmbModel(AttModel):
    def __init__(self, opt):
        super(LenEmbModel, self).__init__(opt)
        self.rem_len_embed = nn.Sequential(nn.Embedding(21, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))

        opt.input_encoding_size = self.input_encoding_size * 2
        self.core = Att2in2Core(opt)
        opt.input_encoding_size = self.input_encoding_size

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # Change start
        if self.decide_length == 'none':
            mask = (seq>0).float()
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
            len_input = (mask.sum(1,keepdim=True) - mask.cumsum(1)).long()
            len_input = mask.sum(1).long()
            assert (len_input == (seq>0 + 1).sum(1)).all()

            state = list(state) + [len_input.unsqueeze(0).unsqueeze(2).expand(state[0].shape)]
        # Change end

        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        for i in range(seq.size(1) - 1):
            if self.decide_length != 'none':
                start_index = 2
            else:
                start_index = 1 # Start schedule sampling or start consider early stop after this index
            if self.training and i >= start_index and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= start_index and seq[:, i].sum() == 0:
                break

            if i == 0:
                if self.decide_length == 'marker':
                    #change start
                    # xt = self.embed(seq[:, i])
                    xt = self.bos_token.reshape(1,-1).expand(batch_size, -1)
                    #change end
                    output, state = self.core(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks)
                    outputs[:, i] = F.log_softmax(self.len_logit(output))
                elif 'init' in self.decide_length:
                    outputs[:, i] = F.log_softmax(self.len_logit(p_fc_feats)) # this only works only when fc_embed is itendity
                if self.decide_length != 'none':
                    continue
            elif i == 1:
                if 'marker' in self.decide_length:
                    it.add_(20000)
                elif self.decide_length == 'init':
                    if os.getenv('LEN_INIT_SANITY') is not None:
                        it.fill_(0)
                    state = [self.memory.reshape(1,1,-1).expand_as(state[0]) * it.unsqueeze(1).float()] + list(state)[1:]
                    it.fill_(0)
                        

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1, len_pred=False):
        # 'it' contains a word index
        if it.dtype is torch.long:
            if (it >= 20000).all():
                xt = self.len_embed(it-20000)
            else:
                xt = self.embed(it)
        else:
            # Used in marker mode, we directly feed embedded vector in
            xt = it

        if len(state) > 2: # only when decide_length != 'none' or it's lenemb model or hasattr self.desired_length
            len_input = state[-1][-1][:, 0].long()
            state = state[:2]
        else:
            if self.decide_length == 'none':
                # When decide_length is none and len_input is not given len_input is set to 9 or desired_length or DEFUALT_LENGTH
                len_input = torch.tensor([getattr(self, 'desired_length', int(os.getenv('DEFAULT_LENGTH', 9)))]).long().expand(fc_feats.shape[0]).to(fc_feats.device)
            # elif hasattr(self, 'desired_length'):
            #     len_input = self.desired_length


        # Change start
        # if len(state) > 2:
        #     len_input = state[-1][-1][:, 0].long()
        # else:
        #     len_input = torch.tensor([getattr(self, 'desired_length', int(os.getenv('DEFAULT_LENGTH', 9)))]).long().expand(fc_feats.shape[0]).to(fc_feats.device)
        xt = torch.cat([xt, self.rem_len_embed(len_input)], 1)
        state = state[:2]
        # Change end

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        if os.getenv('LENGTH_PREDICT') and len_pred:
            self.states.append(self.len_fc(output.detach()))
            # self.states.append(self.len_fc(output))

        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        if hasattr(self, 'desired_length') and os.getenv('LENGTH_PREDICT'):
            # print(self.vocab[str(logprobs.max(-1)[1].item())])
            # import pdb;pdb.set_trace()
            def obj_func(input):
                len_obj = F.log_softmax(self.len_fc(input)).gather(1, len_input.unsqueeze(1)).squeeze(1) # higher
                dist_obj = F.kl_div(F.log_softmax(self.logit(input), dim=1), logprobs.exp(), reduction='none')
                # _grad_l = torch.autograd.grad(len_obj.sum(), input, retain_graph=True)[0]
                # _grad_d = torch.autograd.grad(dist_obj.sum(), input, retain_graph=True)[0]
                # import pdb;pdb.set_trace()
                return (len_obj - LP_CONF['kl_w'] * dist_obj.sum(1)).sum()

            def get_grad(func, input):
                input.requires_grad = True
                input.retain_grad()
                with torch.enable_grad():
                    grad = torch.autograd.grad(func(input), input)
                input.requires_grad = False
                return grad[0]

            new_state = output.data.clone()
            for i in range(LP_CONF['iter']):
                new_state += LP_CONF['lr'] * get_grad(obj_func, new_state)

            logprobs = F.log_softmax(self.logit(new_state), dim=1)
            # print(self.vocab[str(logprobs.max(-1)[1].item())])

            state[0][0].copy_(new_state)

        if 'len_input' in locals():
            state = list(state)[:2] + [F.relu(len_input.float()-1).unsqueeze(0).unsqueeze(2).expand(state[0].shape)]
        if int(os.getenv('CONSTRAINED', 0)):
            logprobs.eos_change = (len_input - 1 >= 0) + (len_input - 1 == 0)
        
        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k+1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k+1].expand(*((beam_size,)+pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(*((beam_size,)+p_att_masks.size()[1:])).contiguous() if att_masks is not None else None

            if hasattr(self, 'desired_length') and getattr(self, 'decide_length', 'none') == 'none':
                state = list(state)[:2] + [state[0].new_tensor([self.desired_length]).unsqueeze(0).unsqueeze(2).expand(state[0].shape)]

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                if t == 0 and self.decide_length != 'none':
                    if self.decide_length == 'marker':
                        xt = self.embed(seq[:, t]) # all zero
                        output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state, tmp_att_masks)
                        logprobs = F.log_softmax(self.len_logit(output))
                    elif 'init' in self.decide_length:
                        logprobs = F.log_softmax(self.len_logit(p_fc_feats))
                else:
                    logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        len_pred = opt.get('len_pred', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size*sample_n)

        if hasattr(self, 'desired_length') and getattr(self, 'decide_length', 'none') == 'none':
            state = list(state)[:2] + [state[0].new_tensor([self.desired_length]).unsqueeze(0).unsqueeze(2).expand(state[0].shape)]

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats = p_fc_feats.unsqueeze(1).repeat(1,sample_n,1).reshape((batch_size*sample_n,)+p_fc_feats.shape[1:])
            p_att_feats = p_att_feats.unsqueeze(1).repeat(1,sample_n,1,1).reshape((batch_size*sample_n,)+p_att_feats.shape[1:])
            pp_att_feats = pp_att_feats.unsqueeze(1).repeat(1,sample_n,1,1).reshape((batch_size*sample_n,)+pp_att_feats.shape[1:])
            p_att_masks = p_att_masks.unsqueeze(1).repeat(1,sample_n,1,1).reshape((batch_size*sample_n,)+p_att_masks.shape[1:]) if att_masks is not None else None

        trigrams = [] # will be a list of batch_size dictionaries

        seq = fc_feats.new_zeros((batch_size*sample_n, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 1)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size*sample_n, dtype=torch.long)

            if self.decide_length != 'none':
                start_offset = 1
            else:
                start_offset = 0

            if t == 1 and self.decide_length != 'none':
                if hasattr(self, 'desired_length'):
                    # Add current length to state
                    state = list(state[:2]) + [state[0].new_tensor([self.desired_length]).unsqueeze(0).unsqueeze(2).expand(state[0].shape)]
                else:
                    state = list(state[:2]) + [it.float().to(state[0].device).unsqueeze(0).unsqueeze(2).expand(state[0].shape)]
                if 'marker' in self.decide_length:
                    if hasattr(self, 'desired_length'):
                        it.fill_(self.desired_length)
                    it.add_(20000)
                elif self.decide_length == 'init':
                    if os.getenv('LEN_INIT_SANITY') is not None:
                        it.fill_(0)
                    state = [self.memory.reshape(1,1,-1).expand_as(state[0]) * it.unsqueeze(1).float()] + list(state)[1:]
                    it.fill_(0)

            if t == 0 and self.decide_length != 'none':
                if self.decide_length == 'marker':
                    xt = self.embed(seq[:, t]) # all zero
                    output, state = self.core(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks)
                    logprobs = self.len_logit(output)
                elif 'init' in self.decide_length:
                    logprobs = self.len_logit(p_fc_feats)
                if output_logsoftmax:
                    logprobs = F.log_softmax(logprobs, 1)
            else:
                logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state, output_logsoftmax=output_logsoftmax, len_pred=len_pred)

            if hasattr(logprobs, 'eos_change'):
                logprobs[logprobs.eos_change == 1, 0] = float('-inf')
                logprobs[logprobs.eos_change == 2, 0] = logprobs[logprobs.eos_change == 2, 0] + 10000

            if decoding_constraint and t > 0+start_offset:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # Mess with trigrams
            if block_trigrams and t >= 3+start_offset:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            elif sample_max == 2: # gumbel softmax
                # ref: https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
                def sample_gumbel(shape, eps=1e-20):
                    U = torch.rand(shape).cuda()
                    return -torch.log(-torch.log(U + eps) + eps)
                def gumbel_softmax_sample(logits, temperature):
                    y = logits + sample_gumbel(logits.size())
                    return F.softmax(y / temperature, dim=-1)
                _logprobs = gumbel_softmax_sample(logprobs, temperature)
                _, it = torch.max(_logprobs.data, 1)
                sampleLogprobs = logprobs.gather(1, it.unsqueeze(1)) # gather the logprobs at sampled positions
            else:
                it = torch.distributions.Categorical(logits=logprobs.detach() / temperature).sample()
                sampleLogprobs = logprobs.gather(1, it.unsqueeze(1)) # gather the logprobs at sampled positions

            # stop when all finished
            if t < 0 + start_offset:
                unfinished = torch.ones_like(it).long()
            elif t == 0+start_offset:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = logprobs
            if output_logsoftmax:
                seqLogprobs[:,t] = F.log_softmax(logprobs, dim=1)
            # Change the length marker so that the decoder and self critical can omit it.
            if t == 0 and self.decide_length != 'none':
                seq[:, 0] += 20000
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs