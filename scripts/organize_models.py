import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

MODELS = Path('models')
summary_path = MODELS / 'multi_seed_training_summary.json'
if not summary_path.exists():
    raise FileNotFoundError(f'Missing {summary_path}')
summary = json.loads(summary_path.read_text(encoding='utf-8'))
per_seed = {int(r['seed']): r for r in summary.get('per_seed', []) if 'seed' in r}

seed_dirs = sorted([p for p in MODELS.iterdir() if p.is_dir() and p.name.startswith('seed_')], key=lambda p: int(p.name.split('_')[1]))
index_rows = []

for seed_dir in seed_dirs:
    seed = int(seed_dir.name.split('_')[1])
    best_ckpt = seed_dir / 'densenet_best.pth'
    latest_ckpt = seed_dir / 'densenet_latest.pth'
    history_csv = seed_dir / 'history.csv'
    metrics_json = seed_dir / 'test_metrics.json'
    curves_png = seed_dir / 'training_curves.png'

    history = []
    if latest_ckpt.exists():
        latest_obj = torch.load(latest_ckpt, map_location='cpu')
        history = list(latest_obj.get('history', []))
        if history:
            fieldnames = [
                'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'epoch_time_sec'
            ]
            with history_csv.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in history:
                    writer.writerow({k: row.get(k) for k in fieldnames})

            epochs = [h.get('epoch') for h in history]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(epochs, [h.get('train_loss') for h in history], label='train')
            axes[0].plot(epochs, [h.get('val_loss') for h in history], label='val')
            axes[0].set_title(f'Seed {seed} Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].legend()

            axes[1].plot(epochs, [h.get('train_acc') for h in history], label='train')
            axes[1].plot(epochs, [h.get('val_acc') for h in history], label='val')
            axes[1].set_title(f'Seed {seed} Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(curves_png, dpi=250)
            plt.close(fig)

    seed_metrics = dict(per_seed.get(seed, {}))
    if best_ckpt.exists():
        best_obj = torch.load(best_ckpt, map_location='cpu')
        seed_metrics.setdefault('seed', seed)
        seed_metrics.setdefault('best_epoch', int(best_obj.get('epoch', -1)))
        seed_metrics.setdefault('best_val_loss', float(best_obj.get('best_val_loss', np.nan)))
        seed_metrics.setdefault('checkpoint', str(best_ckpt))
    if seed_metrics:
        metrics_json.write_text(json.dumps(seed_metrics, indent=2), encoding='utf-8')

    index_rows.append({
        'seed': seed,
        'best_checkpoint': int(best_ckpt.exists()),
        'latest_checkpoint': int(latest_ckpt.exists()),
        'history_csv': int(history_csv.exists()),
        'test_metrics_json': int(metrics_json.exists()),
        'training_curves_png': int(curves_png.exists()),
    })

index_csv = MODELS / 'artifacts_index.csv'
with index_csv.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=['seed', 'best_checkpoint', 'latest_checkpoint', 'history_csv', 'test_metrics_json', 'training_curves_png']
    )
    writer.writeheader()
    writer.writerows(index_rows)

readme = MODELS / 'README.md'
readme.write_text(
    "\n".join([
        '# Models Directory',
        '',
        '- `class_names.txt`, `split_indices.json`, `dataset_manifest.csv`, `data_prep_metadata.json`: dataset/split metadata used by training and evaluation.',
        '- `multi_seed_training_summary.json|csv`: aggregate metrics across seeds.',
        '- `artifacts_index.csv`: quick inventory of per-seed artifacts.',
        '- `seed_<N>/`: per-seed checkpoints and derived artifacts.',
        '',
        'Per seed folder standard files:',
        '- `densenet_best.pth`',
        '- `densenet_latest.pth`',
        '- `history.csv`',
        '- `test_metrics.json`',
        '- `training_curves.png`',
    ]),
    encoding='utf-8'
)

print(f'Updated seed folders: {[p.name for p in seed_dirs]}')
print(f'Wrote: {index_csv}')
print(f'Wrote: {readme}')
