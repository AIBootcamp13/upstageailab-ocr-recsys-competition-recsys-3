import argparse
import csv
from datetime import datetime
from pathlib import Path
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import LightGCN
from recbole.trainer import Trainer
from recbole.utils import init_seed

from utils import *


def log_training_summary(config_path, dataset, best_score, best_result, trainer):
    log_dir = Path(__file__).resolve().parent / 'log'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'lightgcn_metrics.csv'
    best_epoch = getattr(trainer, 'best_valid_epoch', None)
    timestamp = datetime.now().isoformat(timespec='seconds')
    metrics = ';'.join(
        f"{key}={value:.6f}" if isinstance(value, (int, float)) else f"{key}={value}"
        for key, value in sorted(best_result.items())
    ) if best_result else ''
    row = [timestamp, config_path, dataset, best_epoch, f"{best_score:.6f}" if isinstance(best_score, (int, float)) else best_score, metrics]
    header = ['timestamp', 'config_file', 'dataset', 'best_epoch', 'best_valid_score', 'metrics']
    write_header = not log_path.exists()
    with log_path.open('a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./yaml/lightgcn.yaml")
    parser.add_argument("--dataset", default="SASRec_dataset", type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    config = Config(model='LightGCN',
                    config_file_list=[args.config_file],
                    dataset=args.dataset,
                    )
    print('Config loaded')
    init_seed(config['seed'], config['reproducibility'])

    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)

    model = LightGCN(config, train_data.dataset).to(config['device'])
    print("model information : ", model)

    trainer = Trainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    log_training_summary(args.config_file, args.dataset, best_valid_score, best_valid_result, trainer)


if __name__ == "__main__":
    main()
