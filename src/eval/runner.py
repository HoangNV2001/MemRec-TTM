"""
MemRec experiment runner on a fixed cohort

Cohort JSON (built once per dataset/negative scheme, shared by every arm):
    {"dataset": ..., "users": [{"user_id": int, "split": "test"|"val",
                                "target": int, "candidates": [int, ...],
                                "warmup": [{"target": int, "candidates": [...]}, ...]}]}

Protocol
- Warm-up (memory construction) runs before evaluation, in waves: every user of a wave reads
  the same memory snapshot, then Stage-W plans are applied in fixed cohort order. Given cached
  LLM responses this is deterministic, unlike upstream's thread-interleaved writes.
- Evaluation never writes memory (no test labels reach shared memory).
- Rankings are completed: valid candidate ids in LLM order, missing candidates appended in
  candidate order; malformed/error flags are recorded per user.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.data import RecDataset
from src.eval.stats import metrics_from_positions
from src.models.llm_client import LLMClient, CallBudgetExceeded
from src.models.memrec_agent import MemRecAgent


class FakeLLMClient:
    """Schema-shaped responses without any API call (wiring smoke only; scores are meaningless)"""

    def __init__(self, seed: int = 0, log_path: str = None):
        self.log_path = log_path
        self.rng = np.random.RandomState(seed)
        self.model = 'fake'
        self.api_endpoint = 'local'
        self.total_requests = 0

    def generate_json(self, messages, properties, temperature=0.0, max_tokens=4000, debug_logger=None):
        out = self._fake(messages, properties)
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'a') as f:
                f.write(json.dumps({'messages': messages, 'response': json.dumps(out)}) + '\n')
        return out

    def _fake(self, messages, properties):
        self.total_requests += 1
        prompt = messages[-1]['content']
        if 'scores' in properties:
            ids = list(dict.fromkeys(int(x) for x in re.findall(r'Item (\d+)', prompt)))
            return {'scores': [{'item_id': i, 'score': float(self.rng.rand()), 'rationale': 'fake'} for i in ids]}
        if 'facets' in properties:
            nb = re.findall(r'\[(User-\d+|Item-\d+)\]', prompt)[:2]
            return {'facets': [{'facet': 'fake facet', 'confidence': 0.5, 'supporting_neighbors': nb}],
                    'support_edges': []}
        if 'user_memory' in properties:
            nb = re.findall(r'\d+\. ((?:User|Item)-\d+) \(collaborative', prompt)[:1]
            return {'user_memory': 'fake user memory', 'item_memory': 'fake item memory',
                    'neighbor_updates': [{'neighbor_id': n, 'memory_update': 'fake', 'rationale': 'fake'} for n in nb]}
        return {k: '' for k in properties}

    def get_token_stats(self):
        return {'total_requests': self.total_requests, 'physical_calls': 0, 'cache_hits': 0,
                'total_input_tokens': 0, 'total_output_tokens': 0}


def complete_ranking(raw_ranked: List[int], candidates: List[int]):
    """Keep first occurrence of valid candidates, append missing ones in candidate order"""
    cand_set = set(candidates)
    seen, ranked = set(), []
    for i in raw_ranked:
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if i in cand_set and i not in seen:
            ranked.append(i)
            seen.add(i)
    n_missing = len(candidates) - len(ranked)
    ranked += [c for c in candidates if c not in seen]
    return ranked, n_missing


class MemRecRunner:
    def __init__(self, cfg: Dict, project_root: Path):
        self.cfg = cfg
        self.root = project_root
        self.arm = cfg.get('arm', 'memrec')
        self.workers = cfg.get('workers', 16)
        self.wave_size = cfg.get('warmup_wave_size', self.workers)

        with open(self.root / cfg['cohort']) as f:
            self.cohort = json.load(f)
        # Temporal cohorts: only cohort users are held out; everyone else stays fully in the graph
        holdout = [u['user_id'] for u in self.cohort['users']] if self.cohort.get('holdout_cohort_only') else None
        data_dir = self.root / cfg['data_dir']
        t0 = time.time()
        self.dataset = RecDataset(str(data_dir / f"{cfg['dataset']}.inter"), seed=cfg.get('seed', 42),
                                  holdout_users=holdout)
        self.dataset.name = cfg.get('domain_rules', cfg['dataset'])
        self.dataset.load_item_metadata()
        self.use_instruction = cfg.get('use_instruction', False)
        if self.use_instruction:
            self.dataset.load_instructions()
        print(f"Dataset loaded in {time.time() - t0:.1f}s: {self.dataset}")

        prov = cfg.get('provider', {})
        if prov.get('name') == 'fake':
            self.llm = FakeLLMClient(seed=cfg.get('seed', 42), log_path=str(self.root / cfg['log_conversations'])
                                     if cfg.get('log_conversations') else None)
        else:
            self.llm = LLMClient(
                provider_name=prov.get('name', 'gemini'),
                model=prov['model'],
                api_endpoint=prov.get('endpoint'),
                api_key=prov.get('api_key'),
                cache_path=str(self.root / cfg['cache_path']) if cfg.get('cache_path') else None,
                max_physical_calls=cfg.get('max_physical_calls'),
                extra_body=prov.get('extra_body'),
                save_conversations=bool(cfg.get('log_conversations')),
                conversation_log_path=str(self.root / cfg['log_conversations']) if cfg.get('log_conversations') else None,
            )

        # optional separate model for LLM_Rec (e.g. a LoRA adapter served by vLLM); LM_Mem keeps provider.model
        self.rerank_llm = None
        if cfg.get('rerank_model') and prov.get('name') != 'fake':
            self.rerank_llm = LLMClient(
                provider_name=prov.get('name', 'gemini'), model=cfg['rerank_model'], api_endpoint=prov.get('endpoint'),
                api_key=prov.get('api_key'), extra_body=prov.get('extra_body'),
                cache_path=str(self.root / cfg['cache_path']) if cfg.get('cache_path') else None,
                max_physical_calls=cfg.get('max_physical_calls'),
                save_conversations=bool(cfg.get('log_conversations')),
                conversation_log_path=str(self.root / cfg['log_conversations']) if cfg.get('log_conversations') else None)

        m = cfg.get('memrec', {})
        ttm_cfg = dict(m.get('ttm') or {})
        if ttm_cfg.get('trend_evidence'):
            ttm_cfg['trend_cutoff'] = self.cohort['cutoff']
        if ttm_cfg.get('semantic_evidence'):
            ttm_cfg['semantic_emb'] = str(data_dir / f"item_emb_{ttm_cfg.get('semantic_tag', 'bge-small')}.npy")
        self.agent = MemRecAgent(
            dataset=self.dataset,
            llm_client=self.llm,
            k=m.get('k', 16),
            tau=m.get('tau_tokens', 1800),
            n_facets=m.get('n_facets', 7),
            temperature=m.get('temperature', 0.0),
            max_tokens=m.get('max_tokens', 4000),
            mix_min_users=m.get('mix_min_users', 4),
            mix_min_items=m.get('mix_min_items', 6),
            fanout_cap=m.get('write', {}).get('fanout_cap', 8),
            reranker_mode='llm',
            pruner_mode=m.get('pruner', {}).get('mode', 'llm_rules'),
            enable_stage_r=m.get('enable_stage_r', True),
            vanilla=(self.arm == 'vanilla'),
            vanilla_history_len=m.get('vanilla_history_len', 10),
            ttm=ttm_cfg or None,
            rerank_output=m.get('rerank_output', 'scores'),
            isolated=m.get('isolated', False),
            reranker_llm_client=None,  # rerank_model is swapped in after warm-up (see run())
        )

        splits = cfg.get('splits', ['test'])
        self.users = [u for u in self.cohort['users'] if u['split'] in splits]
        if cfg.get('limit_users'):
            self.users = self.users[:cfg['limit_users']]
        print(f"Cohort {cfg['cohort']}: {len(self.users)} users (splits={splits}), arm={self.arm}")

    def _instruction(self, user_id):
        if self.use_instruction and self.dataset.instructions and user_id in self.dataset.instructions:
            return self.dataset.instructions[user_id].get('instruction')
        return None

    # ------------------------------------------------------------------ warm-up
    def _warm_one(self, user_id: int, w: Dict):
        ranked, details = self.agent.rerank(user_id=user_id, candidates=w['candidates'],
                                            instruction=None, return_details=True,
                                            history_items=self.dataset.get_user_train_items(user_id))
        ranked, _ = complete_ranking(ranked, w['candidates'])
        feedback = {'action': 'CLICK', 'item_id': w['target'], 'position': ranked.index(w['target'])}
        return self.agent.plan_write(user_id=user_id, feedback=feedback,
                                     recent_facets=details.get('facets', []),
                                     pruned_subgraph=details.get('pruned_subgraph'))

    def warmup(self):
        if self.arm == 'vanilla' or not self.cfg.get('warmup', {}).get('enabled', True):
            print("Warm-up skipped")
            return 0
        n_rounds = max((len(u.get('warmup', [])) for u in self.users), default=0)
        n_applied, t0 = 0, time.time()
        for r in range(n_rounds):
            entries = [(u['user_id'], u['warmup'][r]) for u in self.users if len(u.get('warmup', [])) > r]
            for s in range(0, len(entries), self.wave_size):
                wave = entries[s:s + self.wave_size]
                with ThreadPoolExecutor(max_workers=self.workers) as ex:
                    futs = [ex.submit(self._warm_one, uid, w) for uid, w in wave]
                    plans = []
                    for (uid, _), fu in zip(wave, futs):
                        try:
                            plans.append(fu.result())
                        except CallBudgetExceeded:
                            raise
                        except Exception as e:
                            print(f"  warm-up error user {uid}: {str(e)[:200]}")
                            plans.append(None)
                for plan in plans:  # fixed order
                    if plan is not None:
                        self.agent.apply_write(plan)
                        n_applied += 1
                print(f"  warm-up round {r + 1}/{n_rounds}: {min(s + self.wave_size, len(entries))}/{len(entries)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        return n_applied

    # --------------------------------------------------------------- evaluation
    def _eval_one(self, u: Dict) -> Dict:
        uid, cands, target = u['user_id'], u['candidates'], u['target']
        rec = {'user_id': uid, 'split': u['split'], 'target': target, 'candidates': cands}
        if u.get('skip_eval'):
            # full-catalog evaluation: target not retrieved, every arm scores 0 whatever the LLM says, so no call
            # (the user still takes part in warm-up, so the shared memory is identical to the base cohort)
            rec.update({'ranked': list(cands), 'n_missing': 0, 'position': -1, 'skipped': True, 'error': False})
            return rec
        try:
            raw, details = self.agent.rerank(
                user_id=uid, candidates=cands, instruction=self._instruction(uid), return_details=True,
                history_items=self.dataset.get_user_history(uid, 'valid'))
            err = any(s.get('rationale') == 'Error' for s in details.get('rerank_scores', []))
            rec.update({'error': bool(err), 'raw_ranked': [s.get('item_id') for s in details.get('rerank_scores', [])],
                        'scores': {str(s.get('item_id')): s.get('score') for s in details.get('rerank_scores', [])},
                        'facets': [(f"[{f['horizon']}] " if f.get('horizon') else '') + f.get('facet', '')
                                   for f in details.get('facets', [])],
                        'n_neighbors': len((details.get('pruned_subgraph') or {}).get('neighbors', [])),
                        'ttm': details.get('ttm')})
        except CallBudgetExceeded:
            raise
        except Exception as e:
            raw = []
            rec.update({'error': True, 'error_msg': str(e)[:300], 'raw_ranked': [], 'scores': {}, 'facets': []})
        ranked, n_missing = complete_ranking(raw, cands)
        # retrieved candidates may miss the target (full-catalog evaluation): position -1 scores 0 everywhere
        rec.update({'ranked': ranked, 'n_missing': n_missing,
                    'position': ranked.index(target) if target in ranked else -1})
        return rec

    def evaluate(self, out_dir: Path):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            records = list(ex.map(self._eval_one, self.users))
        with open(out_dir / 'predictions.jsonl', 'w') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        metrics = {}
        for split in sorted({r['split'] for r in records}):
            pos = [r['position'] for r in records if r['split'] == split]
            metrics[split] = metrics_from_positions(pos)
            metrics[split]['malformed'] = sum(r['n_missing'] > 0 for r in records if r['split'] == split)
            metrics[split]['errors'] = sum(bool(r.get('error')) for r in records if r['split'] == split)
        print(f"Evaluation done in {time.time() - t0:.0f}s")
        return metrics

    def run(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_cfg = {k: v for k, v in self.cfg.items() if k != 'provider'}
        safe_cfg['provider'] = {k: v for k, v in self.cfg.get('provider', {}).items() if k != 'api_key'}
        (out_dir / 'config.json').write_text(json.dumps(safe_cfg, indent=2, ensure_ascii=False))
        t0 = time.time()
        initial_items = dict(self.agent.storage.item_descriptions)
        n_applied = self.warmup()
        stats_warm = self.llm.get_token_stats()
        if self.cfg.get('save_memory', True) and self.arm != 'vanilla':
            st = self.agent.storage
            changed = {str(i): d for i, d in st.item_descriptions.items() if initial_items.get(i) is not d}
            mem = {'user_profiles': {str(u): p for u, p in st.user_profiles.items()}, 'item_memories': changed}
            (out_dir / 'memory_after_warmup.json').write_text(json.dumps(mem, ensure_ascii=False))
        if self.cfg.get('eval_max_tokens'):  # long candidate lists need longer outputs; warm-up keeps its cache keys
            self.agent.max_tokens = self.cfg['eval_max_tokens']
        if self.rerank_llm is not None:
            # rerank_model (e.g. an RL-trained LoRA adapter) scores only the evaluation rerank. Warm-up reranks feed
            # Stage-W (the click position goes into the write prompt), so they stay on the main model and the memory
            # is identical to the base run
            self.agent.reranker.llm = self.rerank_llm
        if self.cfg.get('perturb'):  # memory ablation at evaluation only, after an unperturbed warm-up
            self.agent.perturb = list(self.cfg['perturb'])
        metrics = self.evaluate(out_dir)
        stats_all = self.llm.get_token_stats()
        summary = {'metrics': metrics, 'warmup_writes_applied': n_applied,
                   'llm_stats_after_warmup': stats_warm, 'llm_stats_total': stats_all,
                   'wall_seconds': round(time.time() - t0, 1)}
        if self.rerank_llm is not None:
            summary['rerank_llm_stats'] = self.rerank_llm.get_token_stats()
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        return summary
