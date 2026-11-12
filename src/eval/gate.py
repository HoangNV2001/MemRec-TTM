"""
Confidence-gated fusion of an LLM ranking with the transition-graph prior (MemRec-TTM, component C5)

    s_c = gamma * ( l_c + alpha_u * g_c ),   alpha_u = softplus(a0 + beta . z_u)

l_c = 1/log2(rank_c + 1) from the LLM ranking, g_c = min-max of graph scores inside the candidate list,
z_u = gold-agnostic per-user features (standardized). Parameters are fit with a listwise softmax loss over the
candidates of labelled events (e.g. MemRec warm-up events, which are strictly before the test cutoff).
With beta = 0 this reduces to fixed-alpha late fusion.
"""
import math
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

FEATURES = ['graph_top_share', 'graph_mass_log', 'llm_graph_agree', 'llm_margin', 'history_log']


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def event_arrays(llm_rec: Dict, graph_scores: Dict[int, float], history_len: int):
    """Per-candidate (l, g) and per-user feature vector for one event"""
    cands = llm_rec['candidates']
    rank = {c: k for k, c in enumerate(llm_rec['ranked'])}
    l = np.array([1.0 / math.log2(rank[c] + 2) for c in cands])
    raw = np.array([max(graph_scores.get(c, 0.0), 0.0) for c in cands])
    lo, hi = raw.min(), raw.max()
    g = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
    total = raw.sum()
    llm_scores = sorted((float(v) for v in (llm_rec.get('scores') or {}).values()), reverse=True)
    margin = llm_scores[0] - llm_scores[1] if len(llm_scores) > 1 else 0.0
    agree = spearmanr(-raw, [rank[c] for c in cands]).correlation if hi > lo else 0.0
    z = np.array([
        raw.max() / total if total > 0 else 0.0,
        math.log10(total + 1e-12),
        0.0 if agree is None or np.isnan(agree) else agree,
        margin,
        math.log(max(history_len, 1)),
    ])
    y = cands.index(llm_rec['target'])
    return l, g, z, y


class GatedFusion:
    def __init__(self, l2: float = 1.0, use_gate: bool = True):
        self.l2 = l2
        self.use_gate = use_gate

    def fit(self, events: List):
        L = np.stack([e[0] for e in events])
        G = np.stack([e[1] for e in events])
        Z = np.stack([e[2] for e in events])
        Y = np.array([e[3] for e in events])
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        Zs = (Z - self.mu) / self.sd
        d = Zs.shape[1] if self.use_gate else 0

        def loss(theta):
            log_gamma, a0, beta = theta[0], theta[1], theta[2:]
            alpha = np.broadcast_to(_softplus(a0 + (Zs[:, :d] @ beta if d else 0.0)), (len(Y),))
            S = np.exp(log_gamma) * (L + alpha[:, None] * G)
            S = S - S.max(1, keepdims=True)
            logp = S[np.arange(len(Y)), Y] - np.log(np.exp(S).sum(1))
            return -logp.mean() + self.l2 * (beta @ beta) / len(Y)

        theta0 = np.r_[np.log(5.0), np.log(np.expm1(0.8)), np.zeros(d)]  # start at fixed alpha = 0.8
        res = minimize(loss, theta0, method='L-BFGS-B')
        self.theta, self.d = res.x, d
        return self

    def alpha(self, z):
        zs = (np.asarray(z) - self.mu) / self.sd
        a0, beta = self.theta[1], self.theta[2:]
        return float(_softplus(a0 + (zs[:self.d] @ beta if self.d else 0.0)))

    def rank(self, llm_rec: Dict, l, g, z) -> List[int]:
        a = self.alpha(z)
        s = l + a * g
        cands = llm_rec['candidates']
        order = sorted(range(len(cands)), key=lambda k: (-s[k], llm_rec['ranked'].index(cands[k])))
        return [cands[k] for k in order]
