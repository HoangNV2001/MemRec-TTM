"""
Greedy evaluation of LLM_Rec checkpoints on the RL prompts (checkpoint selection on the training-world val split)

Decoding matches the MemRec harness: temperature 0, JSON schema {"ranking": [int]} enforced by vLLM guided decoding,
chat template in non-thinking mode. Each LoRA checkpoint is a vLLM LoRARequest on one base engine; "base" is the
zero-shot model. Position = index of the target in the deduplicated candidate list (absent target = -1, counted as a
miss, like the harness).

    python scripts/rl/eval_ckpt.py --model Qwen/Qwen3-8B \
        --data data/rl/ml_train08_ttm.jsonl data/rl/amz_train08_ttm.jsonl --split val \
        --ckpts base outputs/grpo-hnv/run1/checkpoint-100 ... --out outputs/grpo-hnv/run1/val_eval.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_grpo import INT, parse_ranking  # noqa: E402

SCHEMA = {'type': 'object', 'properties': {'ranking': {'type': 'array', 'items': {'type': 'integer'}}},
          'required': ['ranking'], 'additionalProperties': False}


def metrics(positions):
    import math
    n = max(len(positions), 1)
    out = {'n': len(positions)}
    for k in (1, 3, 5, 10):
        out[f'Hit@{k}'] = sum(0 <= p < k for p in positions) / n
        out[f'NDCG@{k}'] = sum(1 / math.log2(p + 2) for p in positions if 0 <= p < k) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--ckpts', nargs='+', default=['base'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_tokens', type=int, default=256)
    ap.add_argument('--gpu_mem', type=float, default=0.85)
    ap.add_argument('--max_model_len', type=int, default=10240)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = []
    for p in args.data:
        name = Path(p).stem
        for line in open(p):
            r = json.loads(line)
            if r['split'] == args.split:
                r['dataset'] = name
                rows.append(r)
    prompts = [tok.apply_chat_template(r['messages'], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for r in rows]
    print(f'{len(rows)} {args.split} prompts from {args.data}', flush=True)

    llm = LLM(model=args.model, enable_lora=True, max_lora_rank=32, max_loras=1, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, guided_decoding=GuidedDecodingParams(json=SCHEMA))
    report = {}
    for i, ck in enumerate(args.ckpts):
        lora = None if ck == 'base' else LoRARequest(f'ck{i}', i + 1, ck)
        outs = llm.generate(prompts, sp, lora_request=lora)
        per = {}
        dump = []
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            ranking, valid = parse_ranking(text, r['candidates'])
            pos = ranking.index(r['target']) if r['target'] in ranking else -1
            per.setdefault(r['dataset'], []).append((pos, valid))
            per.setdefault('all', []).append((pos, valid))
            dump.append({'ckpt': ck, 'dataset': r['dataset'], 'user_id': r['user_id'], 'position': pos,
                         'valid': valid, 'n_cand': len(r['candidates']), 'n_ids': len(INT.findall(text)),
                         'n_unique': len(ranking), 'finish': o.outputs[0].finish_reason, 'text': text})
        report[ck] = {d: {**metrics([p for p, _ in v]), 'valid': sum(x for _, x in v) / len(v)} for d, v in per.items()}
        a = report[ck]['all']
        print(f'### {ck}: NDCG@5 {a["NDCG@5"]:.4f} Hit@1 {a["Hit@1"]:.4f} valid {a["valid"]:.3f} '
              + ' '.join(f'{d}:{m["NDCG@5"]:.4f}' for d, m in report[ck].items() if d != 'all'), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(Path(args.out).with_suffix(f'.{i}.positions.jsonl'), 'w') as f:
            f.write(''.join(json.dumps(x) + '\n' for x in dump))
        json.dump(report, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
