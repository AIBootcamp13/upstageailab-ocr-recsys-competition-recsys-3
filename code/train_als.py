
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from implicit.als import AlternatingLeastSquares

from utils import *


def _resolve_path(path_str, base_candidates, must_exist=False):
    path = Path(path_str)
    if path.is_absolute():
        return path
    for base in base_candidates:
        candidate = (base / path).resolve()
        if candidate.exists() or not must_exist:
            return candidate
    return (base_candidates[0] / path).resolve()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="train_processed.parquet", type=str)
    parser.add_argument("--dir_path", default="../data/processed/", type=str)
    parser.add_argument("--output_dir", default="../output/", type=str)
    parser.add_argument("--submission_template", default="../data/sample_submission.csv", type=str)

    parser.add_argument("--num_factor", help="The number of latent factors to compute", type=int, default=256)
    parser.add_argument(
        "--regularization", type=float, default=0.001, help="The regularization factor to use"
    )
    parser.add_argument(
        "--alpha", type=float, default=10.0, help="Governs the baseline confidence in preference observations"
    )
    parser.add_argument("--iterations", type=int, default=30, help="Number of training iterations")

    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    original_cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    base_candidates = [original_cwd, script_dir]

    data_root = _resolve_path(args.dir_path, base_candidates, must_exist=True)
    train_path = _resolve_path(args.data_dir, [data_root], must_exist=True)
    submission_path = _resolve_path(args.submission_template, base_candidates + [data_root], must_exist=True)
    output_dir = _resolve_path(args.output_dir, base_candidates, must_exist=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(train_path)

    user_codes, unique_users = pd.factorize(train_df['user_id'], sort=True)
    item_codes, unique_items = pd.factorize(train_df['item_id'], sort=True)
    train_df['user_idx'] = user_codes.astype(np.int32)
    train_df['item_idx'] = item_codes.astype(np.int32)

    train_df['label'] = 1.0
    confidence = sparse.csr_matrix(
        (
            train_df['label'].astype(np.float32),
            (train_df['user_idx'].to_numpy(), train_df['item_idx'].to_numpy()),
        ),
        shape=(len(unique_users), len(unique_items)),
        dtype=np.float32,
    )

    model = AlternatingLeastSquares(
        factors=args.num_factor,
        regularization=args.regularization,
        alpha=args.alpha,
        iterations=args.iterations,
        use_gpu=False,
    )

    model.fit(confidence)

    user_indices = np.arange(confidence.shape[0])
    recommended_items, _ = model.recommend(
        user_indices,
        confidence,
        N=10,
        filter_already_liked_items=True,
    )

    idx2item = np.asarray(unique_items)
    idx2user = np.asarray(unique_users)

    user_recommendations = {
        idx2user[user_idx]: [idx2item[item_idx] for item_idx in recommended_items[row_idx]]
        for row_idx, user_idx in enumerate(user_indices)
    }

    item_popularity = train_df.groupby('item_idx')['label'].sum().sort_values(ascending=False)
    popular_item_ids = [idx2item[item_idx] for item_idx in item_popularity.index[:10]]

    submission_df = pd.read_csv(submission_path)

    user_row_counts = defaultdict(int)
    filled_items = []
    for user_id in submission_df['user_id']:
        preds = user_recommendations.get(user_id, popular_item_ids)
        idx = user_row_counts[user_id]
        if not preds:
            preds = popular_item_ids
        filled_items.append(preds[idx % len(preds)])
        user_row_counts[user_id] += 1

    submission_df['item_id'] = filled_items

    timestamp = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    output_name = f"ALS-{timestamp}-f{args.num_factor}.csv"
    submission_df.to_csv(output_dir / output_name, index=False)


if __name__ == "__main__":
    main()
