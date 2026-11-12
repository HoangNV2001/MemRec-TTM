"""
Build the RL dataset for LLM_Rec from a logged MemRec run on the training world

For every cohort user, take the final (evaluation) Stage-ReRank prompt from conversations.jsonl: the rerank prompt of
that user whose candidate list is the cohort's evaluation candidates (warm-up prompts use other candidates). Output
JSONL rows: user_id, split, messages, candidates, target, zero_shot_position.

    python scripts/rl/build_rl_dataset.py --run results/runs/ml_train08_ttm_q8b --cohort \
        data/cohorts/temporal-ml32m-train08_popmatched_t2000_v200.json --out data/rl/ml_train08_ttm.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RERANK_MARKERS = ('recommendation scoring system',)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cohort = json.load(open(PROJECT_ROOT / args.cohort))
    users = {u['user_id']: u for u in cohort['users']}
    preds = {}
    for line in open(PROJECT_ROOT / args.run / 'predictions.jsonl'):
        r = json.loads(line)
        preds[int(r['user_id'])] = r
    found = {}
    for line in open(PROJECT_ROOT / args.run / 'conversations.jsonl'):
        x = json.loads(line)
        text = x['messages'][0]['content']
        if not any(m in text for m in RERANK_MARKERS):
            continue
        m = re.search(r'\*\*Target User:\*\* User (\d+)', text)
        if not m or int(m.group(1)) not in users:
            continue
        u = users[int(m.group(1))]
        if all(f'Item {c}' in text for c in u['candidates']):
            found[u['user_id']] = x['messages']
    out = Path(PROJECT_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = {'train': 0, 'val': 0}
    with open(out, 'w') as f:
        for uid, msgs in found.items():
            u = users[uid]
            f.write(json.dumps({'user_id': uid, 'split': u['split'], 'messages': msgs, 'candidates': u['candidates'],
                                'target': u['target'],
                                'zero_shot_position': preds.get(uid, {}).get('position')}, ensure_ascii=False) + '\n')
            n[u['split']] = n.get(u['split'], 0) + 1
    print(f'{len(found)}/{len(users)} users with an evaluation prompt: {n} -> {out}')


if __name__ == '__main__':
    sys.exit(main())
