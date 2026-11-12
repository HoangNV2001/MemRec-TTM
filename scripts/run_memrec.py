"""
Run one MemRec arm on a fixed cohort

    python scripts/run_memrec.py --config configs/runs/books_memrec.yaml [--limit 20] [--fake] [--name smoke]

Outputs results/runs/<name>/{config.json, predictions.jsonl, summary.json, memory_after_warmup.json}
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.config import load_run_config, load_secrets  # noqa: E402
from src.eval.runner import MemRecRunner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--name', default=None, help='run name (default: config stem)')
    ap.add_argument('--limit', type=int, default=None, help='only the first N cohort users (smoke)')
    ap.add_argument('--splits', default=None, help='comma list, e.g. val or test,val')
    ap.add_argument('--fake', action='store_true', help='fake LLM: wiring check, no API call')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--max_calls', type=int, default=None, help='override max_physical_calls')
    ap.add_argument('--log_conversations', action='store_true', help='write every prompt/response to the run dir')
    args = ap.parse_args()

    load_secrets()
    cfg = load_run_config(args.config)
    if args.limit:
        cfg['limit_users'] = args.limit
    if args.splits:
        cfg['splits'] = args.splits.split(',')
    if args.fake:
        cfg['provider'] = {'name': 'fake', 'model': 'fake'}
    if args.workers:
        cfg['workers'] = args.workers
    if args.max_calls is not None:
        cfg['max_physical_calls'] = args.max_calls

    name = args.name or Path(args.config).stem
    if args.log_conversations:
        cfg['log_conversations'] = f'results/runs/{name}/conversations.jsonl'
    out_dir = PROJECT_ROOT / 'results' / 'runs' / name
    runner = MemRecRunner(cfg, PROJECT_ROOT)
    summary = runner.run(out_dir)
    print(json.dumps(summary, indent=2))
    print(f"-> {out_dir}")


if __name__ == '__main__':
    main()
