"""Side script for sweeping over `lambda_e`.

Imports `run` and `get_args` from `lightning_train_and_eval.py` so the main
training entrypoint stays untouched. Every CLI flag accepted by
`lightning_train_and_eval.py` is forwarded through; this script adds one
sweep-specific flag: `--lambda_e_values`.

Example:
    python sweep.py \
        --lambda_e_values 0.0 0.01 0.1 \
        --lambda_t_values 0.1 0.5 1.0 \
        --seed 0 1 2 \
        --group C4xC4 --dataset ddmnist_c4 --model GxGregularfunctor \
        --num_epochs 150
"""

import argparse
import copy
import csv
import os
import sys
import time

import numpy as np

from lightning_train_and_eval import get_args, run


def aggregate_per_lambda(per_seed_results):
    """Given a list (per seed) of lists (per dataloader) of dicts, return
    {dl_idx: {metric: {'mean': float, 'std': float, 'raw': [..]}}}."""
    num_dataloaders = len(per_seed_results[0])
    aggregated = {}
    for dl_idx in range(num_dataloaders):
        metrics = {}
        for seed_results in per_seed_results:
            for key, val in seed_results[dl_idx].items():
                metrics.setdefault(key, []).append(val)
        aggregated[dl_idx] = {
            key: {
                'mean': float(np.array(vals).mean()),
                'std': float(np.array(vals).std()),
                'raw': [float(v) for v in vals],
            }
            for key, vals in metrics.items()
        }
    return aggregated


def print_per_lambda_block(key, aggregated, n_seeds):
    lam_e, lam_t = key
    print(f"\n{'='*60}")
    print(f"lambda_e = {lam_e}, lambda_t = {lam_t}  (averaged over {n_seeds} seed(s))")
    print(f"{'='*60}")
    for dl_idx, metric_map in aggregated.items():
        if len(aggregated) > 1:
            print(f"\n--- Dataloader {dl_idx} ---")
        for key in sorted(metric_map.keys()):
            stats = metric_map[key]
            print(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")


def print_comparison_table(sweep_results):
    """Compact cross-lambda_e summary for headline metrics."""
    print(f"\n{'='*60}")
    print("SWEEP COMPARISON")
    print(f"{'='*60}")
    # Collect every (dl_idx, metric) pair across all lambda_e values
    pairs = set()
    for agg in sweep_results.values():
        for dl_idx, metric_map in agg.items():
            for metric in metric_map:
                pairs.add((dl_idx, metric))
    for dl_idx, metric in sorted(pairs):
        print(f"\n[dl={dl_idx}] {metric}")
        for key in sorted(sweep_results.keys()):
            lam_e, lam_t = key
            stats = sweep_results[key].get(dl_idx, {}).get(metric)
            if stats is None:
                continue
            print(f"  lambda_e={lam_e:<8g} lambda_t={lam_t:<8g}  {stats['mean']:.4f} +/- {stats['std']:.4f}")


def write_csv(sweep_results, out_path, n_seeds):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda_e', 'lambda_t', 'dataloader_idx', 'metric', 'mean', 'std', 'n_seeds', 'raw_values'])
        for key in sorted(sweep_results.keys()):
            lam_e, lam_t = key
            for dl_idx in sorted(sweep_results[key].keys()):
                for metric in sorted(sweep_results[key][dl_idx].keys()):
                    stats = sweep_results[key][dl_idx][metric]
                    writer.writerow([
                        lam_e,
                        lam_t,
                        dl_idx,
                        metric,
                        f"{stats['mean']:.6f}",
                        f"{stats['std']:.6f}",
                        n_seeds,
                        ';'.join(f"{v:.6f}" for v in stats['raw']),
                    ])


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--lambda_e_values', type=float, nargs='+', required=True,
                     help='List of lambda_e values to sweep over.')
    pre.add_argument('--lambda_t_values', type=float, nargs='+', default=None,
                     help='List of lambda_t values to sweep over. '
                          'If omitted, uses the single args.lambda_t value.')
    sweep_args, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining # sys.argv is what is seen by the system (as args of python, basically)

    args = get_args()
    seeds = args.seed

    lambda_t_values = sweep_args.lambda_t_values
    if lambda_t_values is None:
        lambda_t_values = [args.lambda_t]

    print(f"\n{'#'*60}")
    print(f"# lambda_e sweep over {sweep_args.lambda_e_values}")
    print(f"# lambda_t sweep over {lambda_t_values}")
    print(f"# seeds: {seeds}")
    print(f"{'#'*60}")

    sweep_results = {}
    for lam_e in sweep_args.lambda_e_values:
        for lam_t in lambda_t_values:
            per_seed = []
            for seed in seeds:
                a = copy.deepcopy(args)
                a.lambda_e = lam_e
                a.lambda_t = lam_t
                a.seed = seed
                a.equivariant_layer_id = [12] #TODO hardcoded.
                a.fast = True
                per_seed.append(run(a))
            aggregated = aggregate_per_lambda(per_seed)
            sweep_results[(lam_e, lam_t)] = aggregated
            print_per_lambda_block((lam_e, lam_t), aggregated, n_seeds=len(seeds))

    print_comparison_table(sweep_results)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(args.output_root, 'sweeps', f'lambda_e_sweep_{timestamp}.csv')
    write_csv(sweep_results, csv_path, n_seeds=len(seeds))
    print(f"\nCSV written to: {csv_path}")


if __name__ == '__main__':
    main()
