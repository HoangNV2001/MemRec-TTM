"""
Test set of 1,000 users per dataset = the original 300 test users (runs *_t300 / R1 / R5) + 700 more users
(runs *_x700: next users of the seed-42 permutation after val2, RL-world users excluded; own data world).

Prints absolute NDCG@5 / Hit@1 / malformed per arm and paired deltas with a stratified bootstrap
(strata = dataset x part, users resampled within a stratum), per dataset and pooled over the two datasets.
Also re-pools the validation + test comparisons of RESULTS §12 with the larger test set.

    python scripts/analyze_test1000.py [--out results/test1000.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.stats import load_predictions, ndcg_at, stratified_bootstrap  # noqa: E402

RUNS = ROOT / 'results' / 'runs'
DS = ['amz', 'ml']

# arm -> (run of the original 300 test users, run of the 700 extra users); {d} = amz | ml
ARMS = {
    # no LLM (CPU); SASRec on the 700 is trained for the epoch count early stopping chose on the original world
    'random': ('{d}_popmatched_random', '{d}x_popmatched_random'),
    'pop': ('{d}_popmatched_pop', '{d}x_popmatched_pop'),
    'sasrec': ('{d}_popmatched_sasrec', '{d}x_popmatched_sasrec'),
    'lightgcn': ('{d}_popmatched_lightgcn', '{d}x_popmatched_lightgcn'),  # same epoch rule as SASRec
    'bprmf': ('{d}_popmatched_bprmf', '{d}x_popmatched_bprmf'),  # LightGCN with 0 layers
    'sasrecce': ('{d}_popmatched_sasrecce', '{d}x_popmatched_sasrecce'),  # SASRec, full-softmax cross-entropy
    'ppr': ('{d}_popmatched_ppr', '{d}x_popmatched_ppr'),
    # Gemini 3.5 Flash-Lite
    'vanilla': ('{d}_pm_vanilla_t300', '{d}_pm_vanilla_x700'),
    'vanilla_evid': ('{d}_pm_vanilla_evid_t300', '{d}_pm_vanilla_evid_x700'),
    'memrec': ('{d}_pm_memrec_t300', '{d}_pm_memrec_x700'),
    'memrec_recent': ('{d}_pm_memrec_recent_t300', '{d}_pm_memrec_recent_x700'),
    'ttm': ('{d}_pm_ttm_evid_recent_t300', '{d}_pm_ttm_evid_recent_x700'),
    'memrec_evid': ('{d}_pm_ttm_evid_t300', '{d}_pm_ttm_evid_x700'),  # MemRec + C3 (no C0)
    'nofacet': ('{d}_pm_ttm_nofacet_t300', '{d}_pm_ttm_nofacet_x700'),
    # memory-paradigm baselines inside the MemRec framework (taxonomy of the MemRec paper)
    'isolated': ('{d}_pm_isolated_t300', '{d}_pm_isolated_x700'),  # dynamic isolated memory (AgentCF / i2Agent-like)
    'static': ('{d}_pm_static_t300', '{d}_pm_static_x700'),  # static memory (iAgent-like, no instruction)
    # Qwen3-8B zero-shot, scores + reasons (R1, as in the RQ6 zero-shot table)
    'q_vanilla': ('{d}_pm_vanilla_q8b', '{d}_pm_vanilla_q8b_x700'),
    'q_vanilla_evid': ('{d}_pm_vanilla_evid_q8b', '{d}_pm_vanilla_evid_q8b_x700'),
    'q_memrec': ('{d}_pm_memrec_q8b', '{d}_pm_memrec_q8b_x700'),
    'q_ttm': ('{d}_pm_ttm_q8b', '{d}_pm_ttm_q8b_x700'),
    'q_nofacet': ('{d}_pm_ttm_nofacet_q8b', '{d}_pm_ttm_nofacet_q8b_x700'),
    # Qwen3-8B zero-shot, ranked list (R1)
    'q_vanilla_evid_rank': ('{d}_pm_vanilla_evid_q8b_rank', '{d}_pm_vanilla_evid_q8b_rank_x700'),
    'q_memrec_rank_r1': ('{d}_pm_memrec_q8b_rank', '{d}_pm_memrec_q8b_rank_x700'),
    'q_ttm_rank_r1': ('{d}_pm_ttm_q8b_rank', '{d}_pm_ttm_q8b_rank_x700'),
    # RL references: same 32-worker warm-up as the adapter runs (R5)
    'q_ttm_rank': ('{d}_pm_ttm_q8b_rank_w32', '{d}_pm_ttm_q8b_rank_x700'),
    'q_memrec_rank': ('{d}_pm_memrec_q8b_rank_w32', '{d}_pm_memrec_q8b_rank_x700'),
    'q_ttm_score': ('{d}_pm_ttm_q8b_w32', '{d}_pm_ttm_q8b_x700'),
}
for a in ['grpo', 'rgrpo', 'rgrpo2', 'grpo_s2', 'rgrpo_s2']:
    ARMS[f'ttm_{a}'] = (f'{{d}}_pm_ttm_q8b_rank_{a}', f'{{d}}_pm_ttm_q8b_rank_{a}_x700')
for a in ['grpo', 'rgrpo', 'rgrpo2']:
    ARMS[f'memrec_{a}'] = (f'{{d}}_pm_memrec_q8b_rank_{a}', f'{{d}}_pm_memrec_q8b_rank_{a}_x700')
    ARMS[f'vevid_{a}'] = (f'{{d}}_pm_vanilla_evid_q8b_rank_{a}', f'{{d}}_pm_vanilla_evid_q8b_rank_{a}_x700')
    ARMS[f'perturb_{a}'] = (f'{{d}}_pm_ttm_q8b_rank_{a}_perturb', f'{{d}}_pm_ttm_q8b_rank_{a}_perturb_x700')
ARMS['perturb_zs'] = ('{d}_pm_ttm_q8b_rank_perturb', '{d}_pm_ttm_q8b_rank_perturb_x700')
# two-seed means (per user): Rank-GRPO = rgrpo2 (seed 42) + rgrpo_s2 (seed 7); GRPO = grpo + grpo_s2
AVG = {'rgrpo_avg': ('ttm_rgrpo2', 'ttm_rgrpo_s2'), 'grpo_avg': ('ttm_grpo', 'ttm_grpo_s2')}

COMPARE = [  # (label, A, B): delta = B - A
    ('MemRec - Vanilla', 'vanilla', 'memrec'),
    ('C0 on MemRec', 'memrec', 'memrec_recent'),
    ('(MemRec+C0) - Vanilla', 'vanilla', 'memrec_recent'),
    ('C3 on MemRec+C0 (TTM - MemRec+C0)', 'memrec_recent', 'ttm'),
    ('TTM - MemRec', 'memrec', 'ttm'),
    ('C3 on MemRec (no C0)', 'memrec', 'memrec_evid'),
    ('C3 on Vanilla', 'vanilla', 'vanilla_evid'),
    ('nofacet - MemRec', 'memrec', 'nofacet'),
    ('nofacet - TTM', 'ttm', 'nofacet'),
    ('TTM - (Vanilla+C3)', 'vanilla_evid', 'ttm'),
    ('TTM - Vanilla', 'vanilla', 'ttm'),
    # main table: delta vs MemRec
    ('Vanilla - MemRec', 'memrec', 'vanilla'),
    ('Vanilla+C3 - MemRec', 'memrec', 'vanilla_evid'),
    ('PPR - MemRec', 'memrec', 'ppr'),
    ('SASRec - MemRec', 'memrec', 'sasrec'),
    ('LightGCN - MemRec', 'memrec', 'lightgcn'),
    ('BPR-MF - MemRec', 'memrec', 'bprmf'),
    ('SASRec-CE - MemRec', 'memrec', 'sasrecce'),
    ('static - MemRec', 'memrec', 'static'),
    ('isolated - MemRec', 'memrec', 'isolated'),
    ('TTM - isolated', 'isolated', 'ttm'),
    ('TTM - static', 'static', 'ttm'),
    ('TTM - LightGCN', 'lightgcn', 'ttm'),
    ('Rank-GRPO avg - LightGCN', 'lightgcn', 'rgrpo_avg'),
    # Qwen vs Gemini, same arm
    ('Qwen-Gemini vanilla', 'vanilla', 'q_vanilla'),
    ('Qwen-Gemini vanilla_evid', 'vanilla_evid', 'q_vanilla_evid'),
    ('Qwen-Gemini memrec', 'memrec', 'q_memrec'),
    ('Qwen-Gemini ttm', 'ttm', 'q_ttm'),
    ('Qwen-Gemini nofacet', 'nofacet', 'q_nofacet'),
    ('Qwen: TTM - MemRec', 'q_memrec', 'q_ttm'),
    ('Qwen: TTM - Vanilla', 'q_vanilla', 'q_ttm'),
    ('Qwen list - score: vanilla_evid', 'q_vanilla_evid', 'q_vanilla_evid_rank'),
    ('Qwen list - score: memrec', 'q_memrec', 'q_memrec_rank_r1'),
    ('Qwen list - score: ttm', 'q_ttm', 'q_ttm_rank_r1'),
    # RL on the TTM prompt
    ('GRPO s42 - zero-shot', 'q_ttm_rank', 'ttm_grpo'),
    ('Rank-GRPO run1 - zero-shot', 'q_ttm_rank', 'ttm_rgrpo'),
    ('Rank-GRPO s42 - zero-shot', 'q_ttm_rank', 'ttm_rgrpo2'),
    ('GRPO s7 - zero-shot', 'q_ttm_rank', 'ttm_grpo_s2'),
    ('Rank-GRPO s7 - zero-shot', 'q_ttm_rank', 'ttm_rgrpo_s2'),
    ('GRPO avg - zero-shot', 'q_ttm_rank', 'grpo_avg'),
    ('Rank-GRPO avg - zero-shot', 'q_ttm_rank', 'rgrpo_avg'),
    ('zero-shot score - zero-shot list', 'q_ttm_rank', 'q_ttm_score'),
    ('Gemini TTM - zero-shot list', 'q_ttm_rank', 'ttm'),
    ('Rank-GRPO s42 - GRPO s42', 'ttm_grpo', 'ttm_rgrpo2'),
    ('Rank-GRPO run1 - GRPO s42', 'ttm_grpo', 'ttm_rgrpo'),
    ('Rank-GRPO s7 - GRPO s7', 'ttm_grpo_s2', 'ttm_rgrpo_s2'),
    ('Rank-GRPO avg - GRPO avg', 'grpo_avg', 'rgrpo_avg'),
    ('Rank-GRPO s7 - s42', 'ttm_rgrpo2', 'ttm_rgrpo_s2'),
    ('GRPO s7 - s42', 'ttm_grpo', 'ttm_grpo_s2'),
    ('Rank-GRPO s42 - zero-shot score', 'q_ttm_score', 'ttm_rgrpo2'),
    ('Rank-GRPO s42 - Gemini TTM', 'ttm', 'ttm_rgrpo2'),
    ('Rank-GRPO s42 - Gemini nofacet', 'nofacet', 'ttm_rgrpo2'),
    ('Rank-GRPO s7 - Gemini TTM', 'ttm', 'ttm_rgrpo_s2'),
    ('Rank-GRPO s7 - Gemini nofacet', 'nofacet', 'ttm_rgrpo_s2'),
    ('Rank-GRPO avg - Gemini TTM', 'ttm', 'rgrpo_avg'),
    ('Rank-GRPO avg - Gemini nofacet', 'nofacet', 'rgrpo_avg'),
    # memory architecture under the RL-trained LLM_Rec (seed-42 adapters)
    ('TTM - MemRec | zero-shot', 'q_memrec_rank', 'q_ttm_rank'),
    ('TTM - MemRec | GRPO', 'memrec_grpo', 'ttm_grpo'),
    ('TTM - MemRec | Rank-GRPO', 'memrec_rgrpo2', 'ttm_rgrpo2'),
    ('no memory + evidence - TTM | zero-shot', 'q_ttm_rank', 'q_vanilla_evid_rank'),
    ('no memory + evidence - TTM | GRPO', 'ttm_grpo', 'vevid_grpo'),
    ('no memory + evidence - TTM | Rank-GRPO', 'ttm_rgrpo2', 'vevid_rgrpo2'),
    ('permuted - TTM | zero-shot', 'q_ttm_rank', 'perturb_zs'),
    ('permuted - TTM | GRPO', 'ttm_grpo', 'perturb_grpo'),
    ('permuted - TTM | Rank-GRPO', 'ttm_rgrpo2', 'perturb_rgrpo2'),
    # RL gain by prompt (adapter - zero-shot list on the same prompt and memory)
    ('RL gain evid-only | GRPO', 'q_vanilla_evid_rank', 'vevid_grpo'),
    ('RL gain evid-only | Rank-GRPO', 'q_vanilla_evid_rank', 'vevid_rgrpo2'),
    ('RL gain MemRec | GRPO', 'q_memrec_rank', 'memrec_grpo'),
    ('RL gain MemRec | Rank-GRPO', 'q_memrec_rank', 'memrec_rgrpo2'),
    ('RL gain permuted | GRPO', 'perturb_zs', 'perturb_grpo'),
    ('RL gain permuted | Rank-GRPO', 'perturb_zs', 'perturb_rgrpo2'),
]

# validation + test pooling (RESULTS §12): extra validation strata per dataset
VAL = {'vanilla': ['{d}_pm_vanilla_v100'], 'memrec': ['{d}_pm_memrec_v100'],
       'memrec_recent': ['{d}_pm_memrec_recent_v100'], 'vanilla_evid': ['{d}_pm_vanilla_evid_v100', '{d}v2_vanilla_evid'],
       'ttm': ['{d}_pm_ttm_evid_recent_v100', '{d}v2_ttm_evid_recent'], 'nofacet': ['{d}_pm_ttm_nofacet_v100', '{d}v2_ttm_nofacet']}
POOLED = [  # (label, A, B, use val2): val2 has no Vanilla / MemRec runs
    ('MemRec - Vanilla', 'vanilla', 'memrec', False), ('C0 on MemRec', 'memrec', 'memrec_recent', False),
    ('(MemRec+C0) - Vanilla', 'vanilla', 'memrec_recent', False),
    ('C3 on MemRec+C0', 'memrec_recent', 'ttm', False), ('TTM - MemRec', 'memrec', 'ttm', False),
    ('nofacet - MemRec', 'memrec', 'nofacet', False),
    ('nofacet - TTM', 'ttm', 'nofacet', True), ('TTM - (Vanilla+C3)', 'vanilla_evid', 'ttm', True),
    ('nofacet - (Vanilla+C3)', 'vanilla_evid', 'nofacet', True),
]


def per_user(run, split):
    path = RUNS / run / 'predictions.jsonl'
    if not path.exists():
        return None
    recs = load_predictions(str(path))
    recs = {u: r for u, r in recs.items() if r.get('split', split) == split}
    return recs


def arm_vectors(arm, d):
    """dict part -> {user: record} for the 300 + 700 test users; None if a run is missing"""
    if arm in AVG:
        return None
    out = {}
    for part, tmpl in zip(('t300', 'x700'), ARMS[arm]):
        r = per_user(tmpl.format(d=d), 'test')
        if r is None:
            return None
        out[part] = r
    return out


def scores(arm, d, users, metric):
    """per-user metric aligned to users[part]; two-seed averages handled here"""
    if arm in AVG:
        a, b = (scores(x, d, users, metric) for x in AVG[arm])
        return None if a is None or b is None else {p: (a[p] + b[p]) / 2 for p in a}
    v = arm_vectors(arm, d)
    if v is None:
        return None
    out = {}
    for part, us in users.items():
        recs = v[part]
        if metric.startswith('ndcg'):
            out[part] = np.array([ndcg_at(recs[u]['position'], int(metric[4:])) for u in us])
        elif metric.startswith('hit'):
            k = int(metric[3:])
            out[part] = np.array([1.0 if 0 <= recs[u]['position'] < k else 0.0 for u in us])
        else:
            raise ValueError(metric)
    return out


def common_users(arms, d):
    users = None
    for arm in arms:
        base = AVG.get(arm, (arm,))
        for x in base:
            v = arm_vectors(x, d)
            if v is None:
                return None
            u = {p: set(r) for p, r in v.items()}
            users = u if users is None else {p: users[p] & u[p] for p in users}
    return {p: sorted(s) for p, s in users.items()}


def pooled_delta(a, b, use_v2):
    """B - A, NDCG@5, pooled over test (300 + 700) and validation (val, plus val2 if use_v2) of both datasets"""
    deltas = []
    for d in DS:
        us = common_users([a, b], d)
        if us is not None:
            sa, sb = scores(a, d, us, 'ndcg5'), scores(b, d, us, 'ndcg5')
            deltas += [sb[p] - sa[p] for p in ('t300', 'x700')]
        for i, (ra, rb) in enumerate(zip(VAL[a], VAL[b])):
            if i == 1 and not use_v2:
                continue
            A = per_user(ra.format(d=d), 'val'); B = per_user(rb.format(d=d), 'val')
            if A is None or B is None:
                continue
            u = sorted(set(A) & set(B))
            deltas.append(np.array([ndcg_at(B[x]['position'], 5) - ndcg_at(A[x]['position'], 5) for x in u]))
    return stratified_bootstrap(deltas) if deltas else None


def fuse(llm_rec, graph_rec, alpha=0.8):
    """late fusion as scripts/fusion_headroom.py: 1/log2(LLM rank + 1) + alpha * min-max graph score"""
    g = {int(k): v for k, v in graph_rec['scores'].items()}
    lo, hi = min(g.values()), max(g.values())
    if hi <= lo:
        return list(llm_rec['ranked'])
    rank = {c: k + 1 for k, c in enumerate(llm_rec['ranked'])}
    s = {c: 1.0 / np.log2(rank[c] + 1) + alpha * (g[c] - lo) / (hi - lo) for c in llm_rec['candidates']}
    return sorted(llm_rec['ranked'], key=lambda c: -s[c])


def extras(res):
    print('\n## Extras (test, 300 + 700 users)')
    res['extras'] = {}
    for d in DS:
        e = {}
        us = common_users(['memrec', 'ttm', 'ppr'], d)
        n = {a: scores(a, d, us, 'ndcg5') for a in ('memrec', 'ttm', 'ppr')}
        cat = {a: np.concatenate([v['t300'], v['x700']]) for a, v in n.items()}
        e['corr_memrec_ppr'] = float(np.corrcoef(cat['memrec'], cat['ppr'])[0, 1])
        e['oracle_memrec_ppr'] = float(np.maximum(cat['memrec'], cat['ppr']).mean())
        dt = cat['ttm'] - cat['memrec']
        e['ttm_vs_memrec_better_worse_same'] = [int((dt > 1e-9).sum()), int((dt < -1e-9).sum()), int((abs(dt) <= 1e-9).sum())]
        for arm in ('memrec', 'ttm'):
            L, G = arm_vectors(arm, d), arm_vectors('ppr', d)
            deltas = []
            for p in ('t300', 'x700'):
                f = []
                for u in us[p]:
                    ranked = fuse(L[p][u], G[p][u])
                    f.append(ndcg_at(ranked.index(L[p][u]['target']), 5) - ndcg_at(L[p][u]['position'], 5))
                deltas.append(np.array(f))
            e[f'late_fusion_{arm}'] = stratified_bootstrap(deltas)
        for arm in ('ttm_rgrpo2', 'ttm_rgrpo_s2', 'ttm_grpo', 'ttm_grpo_s2'):
            v = arm_vectors(arm, d)
            e[f'stop_after5_pct_{arm}'] = 100 * float(np.mean([len(v[p][u]['raw_ranked']) == 5 for p in us for u in us[p]]))
        res['extras'][d] = e
        print(d, json.dumps(e))


def fmt(r):
    star = '*' if r['ci_low'] > 0 or r['ci_high'] < 0 else ''
    return f"{r['delta']:+.3f} [{r['ci_low']:+.3f}; {r['ci_high']:+.3f}]{star} (n={r['n']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'results' / 'test1000.json'))
    args = ap.parse_args()
    res = {'absolute': {}, 'compare': {}, 'pooled_val_test': {}}

    print('## Absolute (test, 300 + 700 users): NDCG@5 / Hit@1 / malformed % ; NDCG@5 of the 300 and the 700')
    for arm in list(ARMS) + list(AVG):
        row = {}
        for d in DS:
            us = common_users([arm], d)
            if us is None:
                continue
            n = scores(arm, d, us, 'ndcg5'); h = scores(arm, d, us, 'hit1'); h5 = scores(arm, d, us, 'hit5')
            h3 = scores(arm, d, us, 'hit3'); n3 = scores(arm, d, us, 'ndcg3')
            allv = np.concatenate([n['t300'], n['x700']])
            mal = None
            if arm not in AVG:
                v = arm_vectors(arm, d)
                mal = 100 * np.mean([v[p][u].get('n_missing', 0) > 0 for p in us for u in us[p]])
            row[d] = {'ndcg5': float(allv.mean()), 'hit1': float(np.concatenate([h['t300'], h['x700']]).mean()),
                      'hit5': float(np.concatenate([h5['t300'], h5['x700']]).mean()),
                      'hit3': float(np.concatenate([h3['t300'], h3['x700']]).mean()),
                      'ndcg3': float(np.concatenate([n3['t300'], n3['x700']]).mean()),
                      'malformed_pct': mal, 'ndcg5_t300': float(n['t300'].mean()), 'ndcg5_x700': float(n['x700'].mean()),
                      'n': int(len(allv))}
        if row:
            res['absolute'][arm] = row
            print(f'{arm:22s} ' + '  '.join(
                f"{d}: {x['ndcg5']:.3f} / {x['hit1']:.3f} / {('%.1f' % x['malformed_pct']) if x['malformed_pct'] is not None else '-':>5}"
                f" ({x['ndcg5_t300']:.3f} | {x['ndcg5_x700']:.3f}, n={x['n']})" for d, x in row.items()))

    print('\n## Paired deltas, NDCG@5 (B - A), 95% CI stratified bootstrap')
    for label, a, b in COMPARE:
        out = {}
        pooled = []
        for d in DS:
            us = common_users([a, b], d)
            if us is None:
                continue
            sa, sb = scores(a, d, us, 'ndcg5'), scores(b, d, us, 'ndcg5')
            deltas = [sb[p] - sa[p] for p in ('t300', 'x700')]
            out[d] = stratified_bootstrap(deltas)
            pooled += deltas
        if len(out) == 2:
            out['pooled'] = stratified_bootstrap(pooled)
        if out:
            res['compare'][label] = out
            print(f'{label:42s} ' + '  '.join(f'{k}: {fmt(v)}' for k, v in out.items()))

    print('\n## Validation + test pooled (RESULTS §12 with the 1,000-user test set)')
    for label, a, b, use_v2 in POOLED:
        r = pooled_delta(a, b, use_v2)
        if r is not None:
            res['pooled_val_test'][label] = r
            print(f'{label:42s} {fmt(r)}')

    extras(res)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f'\n-> {args.out}')


if __name__ == '__main__':
    main()
