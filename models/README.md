# Models Directory

- `class_names.txt`, `split_indices.json`, `dataset_manifest.csv`, `data_prep_metadata.json`: dataset/split metadata used by training and evaluation.
- `multi_seed_training_summary.json|csv`: aggregate metrics across seeds.
- `artifacts_index.csv`: quick inventory of per-seed artifacts.
- `seed_<N>/`: per-seed checkpoints and derived artifacts.

Per seed folder standard files:
- `densenet_best.pth`
- `densenet_latest.pth`
- `history.csv`
- `test_metrics.json`
- `training_curves.png`