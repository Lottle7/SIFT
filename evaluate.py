"""Evaluate frozen embeddings with ordered five-fold Ridge regression."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from args import ROOT
import utils


def evaluate(embedding_path, city, data_root=ROOT, task="all"):
    embedding = np.load(embedding_path, allow_pickle=False)
    if embedding.ndim != 2 or len(embedding) < 5 or not np.isfinite(embedding).all():
        raise ValueError("Embeddings must be a finite [N, D] matrix with N >= 5")
    tasks = utils.TASKS if task == "all" else (task,)
    results = {}
    for name in tasks:
        path = Path(data_root) / "tasks" / city / utils.TASK_FILES[name]
        label = np.load(path, allow_pickle=False).astype(np.float32).reshape(-1)
        if len(label) != len(embedding) or not np.isfinite(label).all():
            raise ValueError("Invalid or mismatched labels: " + str(path))
        results[name] = utils.downstream_metrics(embedding, label)
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--city", choices=utils.CITIES, required=True)
    parser.add_argument("--data_root", default=str(ROOT))
    parser.add_argument("--task", choices=("all",) + utils.TASKS, default="all")
    parser.add_argument("--output", help="Optional metrics JSON file")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    with threadpool_limits(limits=args.threads):
        metrics = evaluate(args.embedding, args.city, args.data_root, args.task)
    text = json.dumps(metrics, indent=2, allow_nan=False)
    print(text)
    if args.output:
        utils.atomic_json(args.output, metrics)


if __name__ == "__main__":
    main()
