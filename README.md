# MemRec-TTM

Code and experimental results of the master's thesis *Applying Retrieval-Augmented Generation to Recommender Systems*
(the thesis itself is written in Vietnamese).

This repository builds on the official implementation of **MemRec**
([rutgerswiselab/MemRec](https://github.com/rutgerswiselab/MemRec); Chen et al., *MemRec: Collaborative Memory-Augmented
Agentic Recommender System*, ACL 2026). MemRec's architecture, original code and prompts belong to its authors. The
additions made for the thesis are listed under [Changes relative to MemRec](#changes-relative-to-memrec).

**Results:** [`notebooks/results.ipynb`](notebooks/results.ipynb). The notebook is already
executed and contains its outputs. It recomputes every results table from the saved per-user predictions, without any
LLM call.

## Overview

**MemRec-TTM** (*MemRec with Temporal Transition Memory*) keeps the MemRec architecture unchanged (graph-structured
collaborative memory, LM_Mem, Stage-R, Stage-ReRank, Stage-W) and adds two retrieval sources to the ranking prompt
(Stage-ReRank):

- **C0, recent trajectory:** the 10 items the user interacted with most recently, with title and short description.
- **C3, transition evidence:** Personalized PageRank (seeded by the 6 most recent items, restart probability 0.15) on a
  directed, time-ordered item transition graph built from interactions before the cutoff. Each candidate receives one
  evidence sentence with its rank in the whole catalog by PPR score and the strongest direct transition from the
  recent items, e.g. *"#1 of 25,723 items by next-step likelihood; 60 user(s) went directly from '1917 (2019)' to it"*.

**Reinforcement learning for LLM_Rec:** GRPO and Rank-GRPO (adapted to a single relevant item per list) fine-tune
Qwen3-8B (LoRA) on the MemRec-TTM Stage-ReRank prompts of a separate training world whose time cutoff lies before the
test period. LM_Mem and the memory stay frozen.

**Evaluation:** Amazon Books and MovieLens-32M, split by a global time cutoff; 10 candidates per user (the target item
and 9 negatives from the same popularity bin); 1,000 test users per dataset. InstructRec Books, MemRec's original
benchmark, serves as a control where the temporal signal is weak.

## Main results

NDCG@5 on the test set (1,000 users per dataset, 10 candidates). The LLM is Gemini 3.5 Flash-Lite unless stated
otherwise; RL rows are the mean of two training seeds. Best value per column in bold. The notebook reports Hit@1/3/5, NDCG@3/5 and paired bootstrap
confidence intervals.

| Group (MemRec taxonomy) | Method | MovieLens-32M | Amazon Books |
|---|---|---:|---:|
| No LLM | Pop | 0.345 | 0.382 |
| | BPR-MF | 0.630 | 0.404 |
| | LightGCN | 0.619 | 0.562 |
| | SASRec (cross-entropy) | 0.595 | 0.331 |
| | PPR on the transition graph | 0.576 | 0.542 |
| No explicit memory | Vanilla LLM | 0.628 | 0.617 |
| Static memory | MemRec w/o collaborative neighbors and memory writes (\*) | 0.573 | 0.603 |
| Dynamic isolated memory | MemRec w/o collaborative neighbors (\*) | 0.572 | 0.622 |
| Dynamic collaborative memory | MemRec | 0.546 | 0.605 |
| | **MemRec-TTM** | 0.622 | **0.627** |
| | MemRec-TTM, Qwen3-8B (zero-shot) | 0.517 | 0.513 |
| | MemRec-TTM, Qwen3-8B + GRPO | 0.623 | 0.576 |
| | MemRec-TTM, Qwen3-8B + Rank-GRPO | **0.654** | 0.590 |

(\*) Stand-ins for the memory groups of the MemRec paper, built inside the MemRec pipeline with the same LLM, prompts
and candidates. Only the memory changes:

| | Neighbors read by Stage-R | Memory writes (Stage-W, one warm-up round) |
|---|---|---|
| MemRec | other users with co-interactions + the user's own items | user, clicked item, propagated to up to 8 neighbors |
| MemRec w/o collaborative neighbors (≈ AgentCF, i²Agent) | the user's own items only | user and clicked item, no propagation |
| MemRec w/o collaborative neighbors and memory writes (≈ iAgent) | the user's own items only | none (no warm-up) |

They are not re-implementations of AgentCF, i²Agent or iAgent: those methods need user instructions (available only in
InstructRec) or an LLM-driven optimization over all training interactions. On InstructRec Books, the paper's published
numbers for these agents are listed next to our reruns in the notebook.

## Changes relative to MemRec

| Path | Content |
|---|---|
| `src/memory/transition.py`, `src/memory/ttm.py` | Directed transition graph, PPR, C3 evidence sentences, C0 trajectory |
| `src/memory/semantic.py`, `src/memory/trend.py`, `src/retrieval/channels.py` | Alternative retrieval sources (semantic, trend) and the stage-1 candidate retrieval channels |
| `src/models/memrec_agent.py`, `src/models/reranker_llm.py`, `src/memory/manager.py`, `src/memory/packer.py` | C0/C3 in the Stage-ReRank prompt; variants (no Stage-R, no Stage-W, ranked-list output) |
| `src/memory/pruner_llm_rules.py` (`memrec.isolated`) | Memory-group stand-ins: own items only as neighbors, no propagation; add `warmup.enabled: false` for the static variant |
| `src/models/llm_client.py`, `src/models/response_cache.py` | Gemini and vLLM (Qwen3-8B) clients, response cache, JSON-constrained decoding |
| `src/eval/runner.py`, `src/eval/stats.py` | Running one configuration on a fixed cohort; metrics and stratified bootstrap |
| `scripts/prepare_temporal.py` | Building the temporal datasets and fixed cohorts |
| `scripts/run_memrec.py` | Running one configuration (`configs/runs/*.yaml`) |
| `scripts/cpu_baselines.py`, `scripts/sasrec_baseline.py`, `scripts/lightgcn_baseline.py` | Non-LLM baselines: Random, Pop, PPR, SASRec (BCE and cross-entropy), LightGCN, BPR-MF |
| `scripts/rl/` | RL dataset construction, GRPO / Rank-GRPO training (TRL + vLLM), checkpoint selection |
| `scripts/analyze_test1000.py` | Tables on the 1,000-user test set, paired deltas, validation + test pooling |

## Repository layout

```
configs/runs/          configuration of every run (ext/: the 700 additional test users)
data/cohorts/          fixed cohorts: users, target items, 10 candidates (re-indexed IDs only)
results/runs/<run>/    predictions.jsonl (per-user predictions), summary.json (metrics), config.json
results/test1000.json  aggregated output of scripts/analyze_test1000.py
notebooks/             results notebook
```

Run names: `{amz,ml,books}_pm_<config>_{t300,x700,v100}`, where `t300` / `x700` are the original 300 and the additional
700 test users and `v100` is validation; `q8b` means LLM_Rec = Qwen3-8B, `rank` the ranked-list output, `grpo` /
`rgrpo2` the GRPO / Rank-GRPO adapters (seed 42) and `_s2` seed 7. Configurations: `vanilla` (Vanilla LLM), `memrec`
(original MemRec), `memrec_recent` (MemRec + C0), `ttm_evid` (MemRec + C3), `ttm_evid_recent` (MemRec-TTM),
`ttm_nofacet` (MemRec-TTM without Stage-R), `vanilla_evid` (Vanilla LLM + C3), `isolated` (MemRec w/o collaborative
neighbors), `static` (the same without memory writes). Non-LLM baselines: `{amz,ml}_popmatched_<baseline>` (original
300 users) and `{amz,ml}x_popmatched_<baseline>` (additional 700 users).

## Reproducing

Viewing the results only needs `numpy`, `pandas` and `jupyter`:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb
python scripts/analyze_test1000.py
```

Rerunning the experiments from scratch (requires the raw data and LLM access):

1. Environment: `conda env create -f env/environment.yml` or `pip install -r requirements.txt`. The notebook also needs
   `jupyter`; reinforcement learning also needs `trl==0.23.1`, `peft` and `vllm`.
2. Raw data (not redistributed here, per each dataset's terms):
   [Amazon Books Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) (`Books_rating.csv` and
   `books_data.csv` into `data/data_downloaded/amazon_books/raw/`) and
   [MovieLens-32M](https://grouplens.org/datasets/movielens/32m/) (unzipped into `data/data_downloaded/ml-32m/`).
3. Datasets and cohorts:
   ```bash
   python scripts/prepare_temporal.py --dataset amazon_books
   python scripts/prepare_temporal.py --dataset ml32m --sample_users 20000
   ```
   The additional test users, val2 and the RL training world use the same script with `--tag`, `--cutoff_q`,
   `--target_max_q`, `--exclude_q`, `--exclude_n` and `--exclude_users_file` (see `--help`; the training-world command
   is in the docstring of `scripts/rl_world_users.py`). The cutoff and user counts of every cohort are stored in its
   file under `data/cohorts/`.
4. One LLM configuration (Gemini needs the `GEMINI_API_KEY` environment variable):
   ```bash
   python scripts/run_memrec.py --config configs/runs/amz_popmatched_ttm_evid_recent.yaml
   ```
5. Non-LLM baselines:
   ```bash
   python scripts/cpu_baselines.py --config configs/runs/amz_popmatched_memrec.yaml
   python scripts/sasrec_baseline.py --config configs/runs/amz_popmatched_memrec.yaml --prefix amz_popmatched
   python scripts/lightgcn_baseline.py --config configs/runs/amz_popmatched_memrec.yaml --prefix amz_popmatched --device cuda
   python scripts/lightgcn_baseline.py ... --layers 0 --tag bprmf                    # BPR-MF
   python scripts/sasrec_baseline.py ... --loss ce --tag sasrecce --device cuda      # SASRec with cross-entropy
   ```
6. Reinforcement learning: `scripts/rl/build_rl_dataset.py` → `scripts/rl/train_grpo.py --algo {grpo,rank_grpo}` →
   `scripts/rl/eval_ckpt.py` (one H100 GPU, about 4.6 hours per training run). The LoRA adapters are not included.

## Citing MemRec

```bibtex
@inproceedings{chen2026memrec,
  title     = {MemRec: Collaborative Memory-Augmented Agentic Recommender System},
  author    = {Chen, Weixin and Zhao, Yuhan and Huang, Jingyuan and Ye, Zihe and Ju, Clark Mingxuan and Zhao, Tong and Shah, Neil and Chen, Li and Zhang, Yongfeng},
  year      = 2026,
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
  url       = {https://aclanthology.org/2026.acl-long.2061.pdf}
}
```
