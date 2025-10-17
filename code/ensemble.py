import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

# Default configuration for each model:
#  - pattern: glob used to locate the latest submission file
#  - weight: contribution applied during ensemble scoring
MODEL_CONFIG = {
    "BERT4Rec": {"pattern": "BERT4Rec-*.csv", "weight": 0.33},
    "GRU4Rec": {"pattern": "GRU4Rec-*.csv", "weight": 0.10},
    "SASRec": {"pattern": "SASRec-*.csv", "weight": 0.36},
    "ALS": {"pattern": "ALS-*-f256.csv", "weight": 0.21},
}


def parse_weight_overrides(values: Iterable[str]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid weight override '{raw}'. Use Model=Weight format.")
        name, weight_str = raw.split("=", 1)
        name = name.strip()
        if name not in MODEL_CONFIG:
            raise KeyError(f"Unknown model '{name}' in weight override.")
        overrides[name] = float(weight_str)
    return overrides


def locate_latest_file(base_dir: Path, pattern: str) -> Path:
    matches = sorted(base_dir.glob(pattern), key=lambda p: (p.stat().st_mtime, p.name))
    if not matches:
        raise FileNotFoundError(f"No files found in {base_dir} matching pattern '{pattern}'.")
    return matches[-1]


def load_predictions(path: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(path, dtype={"user_id": str, "item_id": str})
    grouped = df.groupby("user_id")["item_id"].apply(list)
    return grouped.to_dict()


def build_scores(
    predictions: Dict[str, List[str]], weight: float, max_rank: int = 10
) -> Tuple[Dict[str, Dict[str, float]], Dict[Tuple[str, str], int], Dict[str, float]]:
    user_item_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    best_rank: Dict[Tuple[str, str], int] = {}
    global_counts: Dict[str, float] = defaultdict(float)

    for user_id, items in predictions.items():
        for rank, item_id in enumerate(items[:max_rank]):
            contribution = weight * (max_rank - rank)
            user_item_scores[user_id][item_id] += contribution
            global_counts[item_id] += contribution
            key = (user_id, item_id)
            if key not in best_rank or rank < best_rank[key]:
                best_rank[key] = rank
    return user_item_scores, best_rank, global_counts


def merge_scores(
    aggregate_scores: Dict[str, Dict[str, float]],
    aggregate_rank: Dict[Tuple[str, str], int],
    aggregate_popularity: Dict[str, float],
    new_scores: Dict[str, Dict[str, float]],
    new_rank: Dict[Tuple[str, str], int],
    new_popularity: Dict[str, float],
) -> None:
    for user_id, item_scores in new_scores.items():
        user_bucket = aggregate_scores[user_id]
        for item_id, score in item_scores.items():
            user_bucket[item_id] += score
            key = (user_id, item_id)
            if key not in aggregate_rank or new_rank[key] < aggregate_rank[key]:
                aggregate_rank[key] = new_rank[key]

    for item_id, score in new_popularity.items():
        aggregate_popularity[item_id] += score


def finalise_predictions(
    scores: Dict[str, Dict[str, float]],
    best_rank: Dict[Tuple[str, str], int],
    global_popularity: Dict[str, float],
    template_users: Iterable[str],
    top_k: int = 10,
) -> pd.DataFrame:
    global_fallback = [
        item_id
        for item_id, _ in sorted(
            global_popularity.items(), key=lambda pair: (-pair[1], pair[0])
        )
    ]

    predictions: Dict[str, List[str]] = {}
    for user_id, item_scores in scores.items():
        ranked = sorted(
            item_scores.items(),
            key=lambda pair: (
                -pair[1],
                best_rank.get((user_id, pair[0]), top_k + 1),
                pair[0],
            ),
        )
        predictions[user_id] = [item_id for item_id, _ in ranked[:top_k]]

    filled_items: List[str] = []
    user_counters: Dict[str, int] = defaultdict(int)

    for user_id in template_users:
        user_preds = predictions.get(user_id, [])
        local_set = set(user_preds)
        idx = user_counters[user_id]

        while len(user_preds) < top_k and len(global_fallback) > 0:
            for fallback_item in global_fallback:
                if fallback_item not in local_set:
                    user_preds.append(fallback_item)
                    local_set.add(fallback_item)
                if len(user_preds) >= top_k:
                    break
            if len(user_preds) < top_k:
                break

        if not user_preds:
            raise ValueError(f"No predictions available for user '{user_id}'.")

        selection = user_preds[idx % len(user_preds)]
        filled_items.append(selection)
        user_counters[user_id] += 1

    return pd.DataFrame({"user_id": list(template_users), "item_id": filled_items})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", default="../output", type=str, help="Directory containing model submissions."
    )
    parser.add_argument(
        "--output_dir", default="../output", type=str, help="Directory to write the ensemble submission."
    )
    parser.add_argument(
        "--submission_template",
        default="../data/sample_submission.csv",
        type=str,
        help="Path to submission template CSV.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help="Override model weight, e.g. --weight BERT4Rec=1.2",
    )
    parser.add_argument(
        "--tag",
        default="ensemble",
        type=str,
        help="Custom tag inserted into the ensemble filename.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    template_path = Path(args.submission_template).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    if not template_path.exists():
        raise FileNotFoundError(f"Submission template '{template_path}' does not exist.")
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_overrides = parse_weight_overrides(args.weight)

    aggregate_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    aggregate_rank: Dict[Tuple[str, str], int] = {}
    aggregate_popularity: Dict[str, float] = defaultdict(float)

    resolved_files = {}
    for model_name, config in MODEL_CONFIG.items():
        pattern = config["pattern"]
        weight = weight_overrides.get(model_name, config["weight"])
        model_file = locate_latest_file(input_dir, pattern)
        resolved_files[model_name] = (model_file, weight)

        predictions = load_predictions(model_file)
        scores, ranks, popularity = build_scores(predictions, weight)
        merge_scores(aggregate_scores, aggregate_rank, aggregate_popularity, scores, ranks, popularity)

    template_df = pd.read_csv(template_path, dtype={"user_id": str})
    ensemble_df = finalise_predictions(
        aggregate_scores,
        aggregate_rank,
        aggregate_popularity,
        template_df["user_id"],
        top_k=10,
    )

    timestamp = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    output_path = output_dir / f"{args.tag}-{timestamp}.csv"
    ensemble_df.to_csv(output_path, index=False)

    print("Ensemble generated at:", output_path)
    print("Models used:")
    for name, (path, weight) in resolved_files.items():
        print(f"  {name}: {path.name} (weight={weight})")


if __name__ == "__main__":
    main()
