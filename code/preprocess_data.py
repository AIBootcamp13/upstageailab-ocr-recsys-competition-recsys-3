"""Data preprocessing utilities for recommendation models.

This script converts the raw clickstream parquet file into
numerically encoded interaction data that can be reused across
models (e.g., SASRec, ALS).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df['event_time'] = pd.to_datetime(df['event_time'])
    # Ensure chronological order within each session for reproducibility
    df = df.sort_values(['user_session', 'event_time'])
    return df.reset_index(drop=True)


def filter_event_types(df: pd.DataFrame, event_types: Optional[List[str]]) -> pd.DataFrame:
    if not event_types:
        return df
    event_types = set(event_types)
    filtered = df[df['event_type'].isin(event_types)].copy()
    return filtered


def enforce_min_interactions(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    if min_user > 1:
        user_counts = df['user_id'].value_counts()
        keep_users = user_counts[user_counts >= min_user].index
        df = df[df['user_id'].isin(keep_users)]
    if min_item > 1:
        item_counts = df['item_id'].value_counts()
        keep_items = item_counts[item_counts >= min_item].index
        df = df[df['item_id'].isin(keep_items)]
    return df.reset_index(drop=True)


def encode_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    user2idx = {uid: idx for idx, uid in enumerate(sorted(df['user_id'].unique()))}
    item2idx = {iid: idx for idx, iid in enumerate(sorted(df['item_id'].unique()))}

    df['user_idx'] = df['user_id'].map(user2idx)
    df['item_idx'] = df['item_id'].map(item2idx)
    df['timestamp'] = df['event_time'].astype('int64') // 10**9

    return df, user2idx, item2idx


def prepare_side_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int], float]:
    df['category_code'] = df['category_code'].fillna('unknown').astype(str)
    df['brand'] = df['brand'].fillna('unknown').astype(str)

    category2idx = {cat: idx for idx, cat in enumerate(sorted(df['category_code'].unique()))}
    brand2idx = {brand: idx for idx, brand in enumerate(sorted(df['brand'].unique()))}

    df['category_idx'] = df['category_code'].map(category2idx)
    df['brand_idx'] = df['brand'].map(brand2idx)

    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    default_price = df['price'].median()
    if pd.isna(default_price):
        default_price = 0.0
    df['price'] = df['price'].fillna(default_price).astype(float)

    return df, category2idx, brand2idx, float(default_price)


def build_item_features(df: pd.DataFrame, default_price: float) -> pd.DataFrame:
    item_features = (
        df.groupby('item_idx')
        .agg(
            category_idx=('category_idx', 'first'),
            brand_idx=('brand_idx', 'first'),
            price=('price', 'median'),
        )
        .reset_index()
    )
    item_features['price'] = item_features['price'].fillna(default_price).astype(float)
    return item_features.sort_values('item_idx').reset_index(drop=True)


def save_mappings(output_dir: Path, user2idx: Dict[str, int], item2idx: Dict[str, int], category2idx: Dict[str, int], brand2idx: Dict[str, int]) -> None:
    with (output_dir / 'user2idx.json').open('w') as f:
        json.dump(user2idx, f)
    with (output_dir / 'item2idx.json').open('w') as f:
        json.dump(item2idx, f)
    with (output_dir / 'category2idx.json').open('w') as f:
        json.dump(category2idx, f)
    with (output_dir / 'brand2idx.json').open('w') as f:
        json.dump(brand2idx, f)


def save_item_features(df: pd.DataFrame, sasrec_dir: Path, default_price: float) -> None:
    item_features = build_item_features(df, default_price)
    item_features = item_features.rename(
        columns={
            'item_idx': 'item_idx:token',
            'category_idx': 'category_idx:token',
            'brand_idx': 'brand_idx:token',
            'price': 'price:float',
        }
    )
    item_features.to_csv(sasrec_dir / 'SASRec_dataset.item', sep='	', index=False)


def save_processed(df: pd.DataFrame, output_dir: Path, default_price: float) -> None:
    df_out = df[
        [
            'user_id',
            'item_id',
            'user_session',
            'event_time',
            'event_type',
            'category_code',
            'brand',
            'price',
            'category_idx',
            'brand_idx',
            'user_idx',
            'item_idx',
            'timestamp',
        ]
    ]
    df_out.to_parquet(output_dir / 'train_processed.parquet', index=False)

    sasrec_dir = output_dir / 'SASRec_dataset'
    sasrec_dir.mkdir(parents=True, exist_ok=True)
    sasrec_df = df[['user_idx', 'item_idx', 'timestamp']].copy()
    sasrec_df.rename(
        columns={
            'user_idx': 'user_idx:token',
            'item_idx': 'item_idx:token',
            'timestamp': 'event_time:float',
        },
        inplace=True,
    )
    sasrec_df.to_csv(sasrec_dir / 'SASRec_dataset.inter', sep='	', index=False)
    save_item_features(df, sasrec_dir, default_price)


def preprocess(
    input_path: Path,
    output_dir: Path,
    min_user_interactions: int,
    min_item_interactions: int,
    event_types: Optional[List[str]],
) -> None:
    df = load_data(input_path)
    if event_types:
        df = filter_event_types(df, event_types)
    df = df.drop_duplicates(['user_id', 'item_id', 'event_time'])
    df = enforce_min_interactions(df, min_user_interactions, min_item_interactions)
    df, user2idx, item2idx = encode_ids(df)
    df, category2idx, brand2idx, default_price = prepare_side_features(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_mappings(output_dir, user2idx, item2idx, category2idx, brand2idx)
    save_processed(df, output_dir, default_price)

    print('Preprocessing complete:')
    print(f"  Users: {len(user2idx)} | Items: {len(item2idx)} | Interactions: {len(df)}")
    print(f"  Categories: {len(category2idx)} | Brands: {len(brand2idx)}")
    print(f"  Output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Preprocess raw parquet interactions dataset.')
    parser.add_argument('--input_path', type=Path, default=Path('../data/train.parquet'))
    parser.add_argument('--output_dir', type=Path, default=Path('../data/processed'))
    parser.add_argument('--min_user_interactions', type=int, default=1)
    parser.add_argument('--min_item_interactions', type=int, default=1)
    parser.add_argument(
        '--event_types',
        type=str,
        nargs='*',
        default=None,
        help='Optional list of event types to retain (e.g., view cart purchase).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess(
        input_path=args.input_path,
        output_dir=args.output_dir,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        event_types=args.event_types,
    )


if __name__ == '__main__':
    main()
