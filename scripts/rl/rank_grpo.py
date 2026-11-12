"""
Rank-GRPO (Zhu et al., arXiv 2510.20150) on top of TRL 0.23.1 GRPOTrainer, for one relevant item per prompt

The completion is a ranking of the N candidates, e.g. {"ranking": [12, 5, 7, ...]}. Each rank is an action unit:

- segment k (k = 0..N-1) holds the tokens of the k-th listed item plus the separators before it; tokens after the
  last listed item (closing brackets, EOS, any extra numbers) form a tail segment N, except that a list closed after
  m < N items puts those tokens in segment m: stopping early is the action taken at rank m (--rank_stop action,
  default since run2; run1 used --rank_stop tail, where the stop was judged only in the tail segment and the policy
  learned to drop items, see segment_completion);
- rank reward (the sequence reward of train_grpo.py split over ranks, so both trainers optimise the same total):
      r_k = 0.70 * 1[y_k = gold] / log2(k + 2) * 1[k < 5] + 0.25 * 1[y_k = gold] * 1[k = 0] - lam * 1[y_k invalid]
  with --rank_reward exp_inf instead r_k = 1[y_k = gold] / log2(k + 2) - lam * 1[y_k invalid] (the Gamma = inf variant
  of the paper, signal at every rank); early stop at rank m: r_m = -lam; tail reward = -lam if IDs follow the N-th
  item, else 0 (with --rank_stop tail: -1 if the list is not a permutation of the candidates, else 0);
- advantage: group-normalised per rank over the completions of the same prompt that have segment k,
      A_{i,k} = (r_{i,k} - mean_k) / (std_k + 1e-4),
  computed separately for "open" completions (gold not listed before rank k) and "closed" ones (gold already listed).
  Adaptation to one relevant item: once the gold is placed, later choices do not change the metric, so a closed
  completion is only compared with other closed ones (on validity). Pooling them with open ones would give the best
  lists (gold at rank 1) negative advantage at every later rank, pushing down the only valid continuations.
- importance ratio per rank: geometric mean of the token ratios, w_{i,k} = exp(mean_{t in k} log pi/pi_old)
- loss: -sum_{i,k} min(w A, clip(w) A) / sum_i (N_i + 1), times TRL's vLLM truncated-IS ratio per token.

Single process only (one GPU): group statistics are computed on the local generation batch.
"""
import math
import re

import torch
import torch.nn.functional as F
from trl import GRPOTrainer

INT = re.compile(r'-?\d+')


def item_spans(text):
    """Character spans of the listed IDs: inside the "ranking" array when present, else every integer"""
    m = re.search(r'"ranking"\s*:\s*\[', text)
    if m:
        end = text.find(']', m.end())
        end = len(text) if end < 0 else end
        return [(x.start() + m.end(), x.end() + m.end(), int(x.group())) for x in INT.finditer(text[m.end():end])]
    return [(x.start(), x.end(), int(x.group())) for x in INT.finditer(text)]


def token_char_ends(tok, ids):
    """End character offset of each token in the decoded completion"""
    pieces = [tok.decode([i]) for i in ids]
    text = tok.decode(ids)
    if ''.join(pieces) != text:  # multi-byte characters split across tokens: fall back to prefix decoding
        ends = [len(tok.decode(ids[:j + 1])) for j in range(len(ids))]
        return ends, text
    ends, acc = [], 0
    for p in pieces:
        acc += len(p)
        ends.append(acc)
    return ends, text


def segment_completion(tok, ids, candidates, target, reward='matched', lam=0.5, stop='action'):
    """Per-token segment ids (len(ids)), per-segment rewards / presence / open flags (N + 1), target position, valid

    stop='action': closing the list after m < N items is the action taken at rank m (reward -lam, compared with the
    items other completions place at rank m, in the same open/closed pool); the tail segment exists only after N
    items and costs -lam if extra IDs follow. stop='tail' (run1): every token after the last item is the tail
    segment, reward -1 when the list is not a permutation."""
    n = len(candidates)
    ends, text = token_char_ends(tok, ids)
    spans = item_spans(text)
    cand = set(candidates)
    seg = [n] * len(ids)
    rewards = [0.0] * (n + 1)
    present = [False] * (n + 1)
    open_ = [True] * n + [False]  # tail segment: one pool
    seen, pos, j0 = set(), None, 0
    for k, (_, cend, item) in enumerate(spans[:n]):
        j = j0
        while j < len(ids) and ends[j] < cend:  # last token whose text reaches the end of this item
            j += 1
        j = min(j, len(ids) - 1)
        for t in range(j0, j + 1):
            seg[t] = k
        present[k] = True
        open_[k] = pos is None
        j0 = j + 1
        invalid = item not in cand or item in seen
        seen.add(item)
        hit = (item == target and not invalid)
        if hit:
            pos = k
        if reward == 'exp_inf':
            r = hit / math.log2(k + 2)
        else:
            r = 0.70 * hit / math.log2(k + 2) * (k < 5) + 0.25 * hit * (k == 0)
        rewards[k] = r - lam * invalid
        if j0 >= len(ids):
            break
    valid = len(spans) == n and seen == cand
    m = sum(present[:n])  # items placed
    if stop == 'action' and m < n and j0 < len(ids):  # early stop = the action at rank m
        for t in range(j0, len(ids)):
            seg[t] = m
        present[m], open_[m], rewards[m] = True, pos is None, -lam
    elif stop == 'action':
        present[n] = j0 < len(ids)
        rewards[n] = -lam if len(spans) > n else 0.0
    else:
        present[n] = j0 < len(ids)
        rewards[n] = 0.0 if valid else -1.0
    return seg, rewards, present, open_, pos, valid


class RankGRPOTrainer(GRPOTrainer):
    rank_reward = 'matched'
    invalid_penalty = 0.5
    rank_stop = 'action'

    def _generate_and_score_completions(self, inputs):
        assert self.accelerator.num_processes == 1, 'RankGRPOTrainer computes group statistics locally'
        assert self.top_entropy_quantile == 1.0 and not self.use_liger_loss
        out = super()._generate_and_score_completions(inputs)
        tok = self.processing_class
        comp, mask = out['completion_ids'], out['completion_mask']
        B, T = comp.shape
        K = max(len(x['candidates']) for x in inputs) + 1
        seg = torch.full((B, T), -1, dtype=torch.long)
        rew = torch.zeros(B, K)
        pres = torch.zeros(B, K, dtype=torch.bool)
        opn = torch.zeros(B, K, dtype=torch.bool)
        norm = torch.zeros(B)
        hits, valids, early = [], [], []
        for b, x in enumerate(inputs):
            L = int(mask[b].sum().item())
            if L == 0:
                continue
            n = len(x['candidates'])
            s, r, p, o, pos, valid = segment_completion(tok, comp[b, :L].tolist(), x['candidates'], x['target'],
                                                     self.rank_reward, self.invalid_penalty, self.rank_stop)
            early.append(len(item_spans(tok.decode(comp[b, :L].tolist()))) < n)
            s = [k if k < n else K - 1 for k in s]  # tail segment always last
            seg[b, :L] = torch.tensor(s)
            rew[b, :n], rew[b, K - 1] = torch.tensor(r[:n]), r[n]
            pres[b, :n], pres[b, K - 1] = torch.tensor(p[:n]), p[n]
            opn[b, :n] = torch.tensor(o[:n])
            norm[b] = n + 1
            hits.append(pos)
            valids.append(valid)
        G = self.num_generations
        rg = rew.view(-1, G, K)
        adv = torch.zeros_like(rg)
        informative = []
        for pool in (opn, ~opn):  # open / closed completions are normalised separately
            pg = (pres & pool).view(-1, G, K).float()
            cnt = pg.sum(1, keepdim=True)
            mean = (rg * pg).sum(1, keepdim=True) / cnt.clamp(min=1)
            var = (((rg - mean) ** 2) * pg).sum(1, keepdim=True) / (cnt - 1).clamp(min=1)
            adv = adv + (rg - mean) / (var.sqrt() + 1e-4) * pg
            informative.append(((var > 1e-8) & (cnt > 1)).float().squeeze(1))
        adv = adv.view(B, K)
        device = comp.device
        out['rank_seg'] = seg.to(device)
        out['rank_adv'] = adv.to(device)
        out['rank_norm'] = norm.to(device)
        mode = 'train' if self.model.training else 'eval'
        found = [h for h in hits if h is not None]
        self._metrics[mode]['rank/target_listed'].append(len(found) / max(len(hits), 1))
        self._metrics[mode]['rank/hit1'].append(sum(h == 0 for h in found) / max(len(hits), 1))
        self._metrics[mode]['rank/valid'].append(sum(valids) / max(len(valids), 1))
        self._metrics[mode]['rank/short_list'].append(sum(early) / max(len(early), 1))
        self._metrics[mode]['rank/informative_open'].append(informative[0][:, :-1].mean().item())
        self._metrics[mode]['rank/informative_closed'].append(informative[1].mean().item())
        return out

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs['prompt_ids'], inputs['prompt_mask']
        completion_ids, completion_mask = inputs['completion_ids'], inputs['completion_mask']
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True)

        old = inputs.get('old_per_token_logps')
        old = per_token_logps.detach() if old is None else old
        log_ratio = per_token_logps - old

        seg, A = inputs['rank_seg'], inputs['rank_adv']
        K = A.size(1)
        m = completion_mask.float() * (seg >= 0).float()
        onehot = F.one_hot(seg.clamp(min=0), K).float() * m.unsqueeze(-1)  # (B, T, K)
        seg_len = onehot.sum(1)  # (B, K)
        present = (seg_len > 0).float()
        seg_lr = torch.einsum('bt,btk->bk', log_ratio, onehot) / seg_len.clamp(min=1)
        w = torch.exp(seg_lr)
        w2 = torch.clamp(w, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            w = torch.clamp(w, max=self.args.delta)
        seg_loss = -torch.min(w * A, w2 * A) * present  # (B, K)
        share = onehot / seg_len.clamp(min=1).unsqueeze(1)  # each token carries 1/|segment| of its segment
        tok_loss = torch.einsum('bk,btk->bt', seg_loss, share)
        if self.use_vllm and self.vllm_importance_sampling_correction:
            tok_loss = tok_loss * inputs['importance_sampling_ratio']
        if self.beta != 0.0:
            ref = inputs['ref_per_token_logps']
            kl = torch.exp(ref - per_token_logps) - (ref - per_token_logps) - 1
            tok_loss = tok_loss + self.beta * kl * share.sum(-1)
        loss = (tok_loss * m).sum() / inputs['rank_norm'].sum().clamp(min=1.0)
        loss = loss / self.current_gradient_accumulation_steps

        mode = 'train' if self.model.training else 'eval'
        tokens = completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]['entropy'].append(((entropies * completion_mask).sum() / tokens).item())
        if self.beta != 0.0:
            self._metrics[mode]['kl'].append(((kl * completion_mask).sum() / tokens).item())
        clipped = (((w < 1 - self.epsilon_low) & (A < 0)) | ((w > 1 + self.epsilon_high) & (A > 0))).float()
        self._metrics[mode]['clip_ratio/region_mean'].append(
            ((clipped * present).sum() / present.sum().clamp(min=1)).item())
        return loss
