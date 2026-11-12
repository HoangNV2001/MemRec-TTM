"""
GRPO / Rank-GRPO for MemRec's LLM_Rec (Qwen3-8B + LoRA, TRL with vLLM colocate, one GPU)

Prompts are the real MemRec-TTM Stage-ReRank prompts of the training world (scripts/rl/build_rl_dataset.py), rendered
with the Qwen3 chat template in non-thinking mode. The completion is a JSON ranking of the candidate IDs.

    reward = 0.70 * NDCG@5 + 0.25 * Hit@1 + 0.05 * valid      (valid: every candidate exactly once)

Three reward functions (ndcg5, hit1, valid) weighted by TRL, so the log shows each component.

--algo grpo       sequence-level reward, standard group-relative advantage (TRL GRPOTrainer)
--algo rank_grpo  rank-level credit (Rank-GRPO, Zhu et al. 2025): see RankGRPOTrainer below

    python scripts/rl/train_grpo.py --data data/rl/ml_train08_ttm.jsonl data/rl/amz_train08_ttm.jsonl \
        --model Qwen/Qwen3-8B --out outputs/grpo-hnv/run1
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

INT = re.compile(r'-?\d+')


def parse_ranking(text, candidates):
    """Ordered candidate IDs from a completion; tolerant to broken JSON. Returns (ranking, valid)"""
    ids = None
    try:
        obj = json.loads(text[text.index('{'):text.rindex('}') + 1])
        ids = [int(i) for i in obj.get('ranking', [])]
    except Exception:
        ids = [int(i) for i in INT.findall(text)]
    cand = set(candidates)
    seen, ranking = set(), []
    for i in ids:
        if i in cand and i not in seen:
            ranking.append(i)
            seen.add(i)
    valid = len(ranking) == len(candidates) and len(ids) == len(candidates)
    return ranking, valid


def components(text, candidates, target):
    """(NDCG@5, Hit@1, valid) of one completion; an unranked target counts as last"""
    ranking, valid = parse_ranking(text, candidates)
    pos = ranking.index(target) if target in ranking else len(candidates)
    return (1.0 / math.log2(pos + 2) if pos < 5 else 0.0), float(pos == 0), float(valid)


def rank_reward(text, candidates, target):
    n5, h1, v = components(text, candidates, target)
    return 0.70 * n5 + 0.25 * h1 + 0.05 * v


def completion_text(c):
    return c[0]['content'] if isinstance(c, list) else c


def _component_fn(idx, name):
    def fn(prompts, completions, candidates, target, **kwargs):
        return [components(completion_text(c), cs, t)[idx] for c, cs, t in zip(completions, candidates, target)]
    fn.__name__ = name
    return fn


# TRL logs each reward function separately and sums them with REWARD_WEIGHTS: total = rank_reward
REWARD_FUNCS = [_component_fn(0, 'ndcg5'), _component_fn(1, 'hit1'), _component_fn(2, 'valid')]
REWARD_WEIGHTS = [0.70, 0.25, 0.05]


def load_rows(paths, split):
    rows = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r['split'] == split:
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--algo', choices=['grpo', 'rank_grpo'], default='grpo')
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--num_generations', type=int, default=8)
    ap.add_argument('--batch', type=int, default=8, help='completions per device step')
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--max_steps', type=int, default=-1)
    ap.add_argument('--max_prompt', type=int, default=6144)
    ap.add_argument('--max_completion', type=int, default=160)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--vllm_mem', type=float, default=0.35)
    ap.add_argument('--save_steps', type=int, default=100)
    ap.add_argument('--logging_steps', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rank_reward', choices=['matched', 'exp_inf'], default='matched')
    ap.add_argument('--invalid_penalty', type=float, default=0.5)
    ap.add_argument('--rank_stop', choices=['action', 'tail'], default='action',
                    help='rank_grpo: early stop = action at its rank (default) or judged in the tail segment (run1)')
    ap.add_argument('--limit', type=int, default=0, help='use only the first N training prompts (smoke)')
    args = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(args.model)

    def render(msgs):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    train = load_rows(args.data, 'train')
    ds = Dataset.from_list([{'prompt': render(r['messages']), 'candidates': r['candidates'], 'target': r['target']}
                            for r in train]).shuffle(seed=args.seed)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    lengths = [len(tok(x)['input_ids']) for x in ds['prompt']]
    over = sum(n > args.max_prompt for n in lengths)
    print(f'{len(ds)} training prompts; prompt tokens max {max(lengths)}, mean {sum(lengths) / len(lengths):.0f}, '
          f'{over} longer than --max_prompt {args.max_prompt} (TRL truncates them from the left)', flush=True)

    cfg = GRPOConfig(
        output_dir=args.out, learning_rate=args.lr, num_generations=args.num_generations,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps, max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion, temperature=args.temperature, beta=args.beta,
        use_vllm=True, vllm_mode='colocate', vllm_gpu_memory_utilization=args.vllm_mem,
        bf16=True, gradient_checkpointing=True, logging_steps=args.logging_steps, save_steps=args.save_steps,
        save_only_model=True, report_to='none', seed=args.seed, log_completions=False,
        reward_weights=REWARD_WEIGHTS,
    )
    print(f'loss_type={cfg.loss_type} scale_rewards={cfg.scale_rewards} beta={cfg.beta} '
          f'generation_batch={cfg.generation_batch_size} epsilon={cfg.epsilon}', flush=True)
    peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, target_modules='all-linear',
                          task_type='CAUSAL_LM')
    trainer_cls = GRPOTrainer
    if args.algo == 'rank_grpo':
        from rank_grpo import RankGRPOTrainer
        RankGRPOTrainer.rank_reward = args.rank_reward
        RankGRPOTrainer.invalid_penalty = args.invalid_penalty
        RankGRPOTrainer.rank_stop = args.rank_stop
        trainer_cls = RankGRPOTrainer
    trainer = trainer_cls(model=args.model, reward_funcs=REWARD_FUNCS, args=cfg, train_dataset=ds,
                          peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(str(Path(args.out) / 'final'))
    print('### training done', flush=True)


if __name__ == '__main__':
    sys.exit(main())
