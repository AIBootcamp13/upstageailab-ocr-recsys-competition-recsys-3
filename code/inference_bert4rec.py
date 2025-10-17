from pathlib import Path
from collections import defaultdict
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
from tqdm import tqdm

from utils import *

MODEL_NAME = 'BERT4Rec'
POPULARITY_BLEND = 0.05


def _resolve_path(base_paths, relative_path):
    path = Path(relative_path)
    if path.is_absolute():
        return path
    for base in base_paths:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (base_paths[0] / path).resolve()


def _find_latest_checkpoint(default_dirs):
    pattern = re.compile(rf"{MODEL_NAME}-.*\.pth$")
    for directory in default_dirs:
        if not directory.exists():
            continue
        candidates = [p for p in directory.glob('*.pth') if pattern.match(p.name)]
        if candidates:
            return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", default="train_processed.parquet", type=str)
    parser.add_argument("--data_dir", default="../data/processed/", type=str)
    parser.add_argument("--output_dir", default="../output/", type=str)
    parser.add_argument("--submission_template", default="../data/sample_submission.csv", type=str)
    parser.add_argument("--model_file", default=None, type=str, help="Path to trained BERT4Rec checkpoint (.pth). Defaults to latest BERT4Rec file in ./check_point.")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    original_cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    data_dir = _resolve_path([original_cwd, script_dir], args.data_dir)
    train_path = data_dir / args.train_dataset
    train = pd.read_parquet(train_path).sort_values(by=['user_session', 'event_time'])

    with open(data_dir / 'user2idx.json', 'r') as f_user:
        user2idx = json.load(f_user)
    with open(data_dir / 'item2idx.json', 'r') as f_item:
        item2idx = json.load(f_item)

    idx2user = {idx: uid for uid, idx in user2idx.items()}
    idx2item = {idx: iid for iid, idx in item2idx.items()}

    train['user_idx'] = train['user_id'].map(user2idx)
    train['item_idx'] = train['item_id'].map(item2idx)
    train = train.dropna(subset=['user_idx', 'item_idx'])
    train['user_idx'] = train['user_idx'].astype(int)
    train['item_idx'] = train['item_idx'].astype(int)

    users = defaultdict(list)
    for u, i in zip(train['user_idx'], train['item_idx']):
        users[u].append(i)

    model_path = None
    if args.model_file:
        model_path = _resolve_path([original_cwd, script_dir], args.model_file)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    else:
        default_dirs = [
            (original_cwd / 'check_point').resolve(),
            (script_dir / 'check_point').resolve(),
        ]
        model_path = _find_latest_checkpoint(default_dirs)
        if model_path is None:
            raise FileNotFoundError(
                "No checkpoint found. Provide --model_file pointing to a trained BERT4Rec .pth file."
            )

    print(f"Using checkpoint: {model_path}")
    os.chdir(script_dir)
    try:
        config, model, dataset, _, _, test_data = load_data_and_model(
            model_file=str(model_path),
        )
    finally:
        os.chdir(original_cwd)
    print('Data and model load complete')

    if 'timestamp' in train.columns:
        max_ts = float(train['timestamp'].max())
        decay = 7 * 24 * 3600
        time_gap = (max_ts - train['timestamp'].astype(float)).clip(lower=0)
        train['recency_weight'] = np.exp(-time_gap / decay)
        popularity_scores = train.groupby('item_idx')['recency_weight'].sum()
    else:
        popularity_scores = train.groupby('item_idx').size()

    popularity_scores = popularity_scores.astype(float)
    if len(popularity_scores) > 0:
        pop_min = float(popularity_scores.min())
        pop_max = float(popularity_scores.max())
        if pop_max == pop_min:
            popularity_norm = popularity_scores / (abs(pop_max) + 1e-8)
        else:
            popularity_norm = (popularity_scores - pop_min) / (pop_max - pop_min)
    else:
        popularity_norm = popularity_scores.copy()
    popularity_map = popularity_norm.to_dict()
    popular_top_10 = popularity_scores.sort_values(ascending=False).head(10).index
    popular_idx_sequence = list(popular_top_10)
    popular_item_ids = [idx2item[i] for i in popular_top_10]
    user_predictions = {}

    for uid in tqdm(users, desc=f"Predicting with {MODEL_NAME}"):
        if str(uid) in dataset.field2token_id['user_idx']:
            recbole_id = dataset.token2id(dataset.uid_field, str(uid))
            topk_score, topk_iid_list = full_sort_topk(
                [recbole_id], model, test_data, k=10, device=config['device']
            )
            topk_scores = topk_score.squeeze(0).detach().cpu().numpy().tolist()
            token_matrix = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
            predicted_item_list = list(map(int, token_matrix[-1]))
            if 0.0 < POPULARITY_BLEND < 1.0:
                blended = []
                for iid, score in zip(predicted_item_list, topk_scores):
                    pop_score = popularity_map.get(iid, 0.0)
                    final_score = score * (1.0 - POPULARITY_BLEND) + pop_score * POPULARITY_BLEND
                    blended.append((final_score, iid))
                blended.sort(key=lambda x: x[0], reverse=True)
                predicted_item_list = [iid for _, iid in blended[:10]]
        else:
            predicted_item_list = list(popular_idx_sequence)

        original_user_id = idx2user[uid]
        predicted_item_ids = [idx2item[iid] for iid in predicted_item_list]
        user_predictions[original_user_id] = predicted_item_ids

    output_dir = _resolve_path([original_cwd, script_dir], args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = _resolve_path([original_cwd, script_dir], args.submission_template)
    submission_df = pd.read_csv(submission_path)

    user_row_counts = defaultdict(int)
    filled_items = []
    for user_id in submission_df['user_id']:
        preds = user_predictions.get(user_id, popular_item_ids)
        idx = user_row_counts[user_id]
        if len(preds) == 0:
            preds = popular_item_ids
        item_id = preds[idx % len(preds)]
        user_row_counts[user_id] += 1
        filled_items.append(item_id)

    submission_df['item_id'] = filled_items

    model_name = Path(model_path).stem if model_path is not None else MODEL_NAME
    sanitized_name = ''.join(ch for ch in model_name if ch.isalnum() or ch in ('-', '_')) or MODEL_NAME

    suffix = 1
    while True:
        output_filename = f"{sanitized_name}{suffix}.csv"
        output_path = output_dir / output_filename
        if not output_path.exists():
            break
        suffix += 1

    submission_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
