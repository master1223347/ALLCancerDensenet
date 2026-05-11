# Research-Grade Results Summary

## 1) Overall Classification Metrics
- Accuracy = 0.9668 ± 0.0014
- Balanced accuracy = 0.9668 ± 0.0014
- Macro F1 = 0.9667 ± 0.0015
- Macro Recall = 0.9668 ± 0.0014
- Micro F1 = 0.9668 ± 0.0014

## 2) Per-Class Classification Metrics
| Class | Precision | Recall (Sensitivity) | F1-score | Specificity | Support |
|---|---:|---:|---:|---:|---:|
| all_benign | 0.9582 | 0.9467 | 0.9524 | 0.9862 | 750 |
| all_early | 0.9449 | 0.9600 | 0.9524 | 0.9813 | 750 |
| all_pre | 0.9865 | 0.9733 | 0.9799 | 0.9956 | 750 |
| all_pro | 0.9841 | 0.9933 | 0.9887 | 0.9947 | 750 |

- Confusion matrix (normalized row-wise + absolute counts):
  - results/comprehensive_eval/figures/confusion_matrices.png
  - results/comprehensive_eval/tables/confusion_matrix_normalized.csv
  - results/comprehensive_eval/tables/confusion_matrix_counts.csv

## 3) ROC / PR / Calibration
- ROC curves:
  - results/comprehensive_eval/figures/roc_curves.png
| Class | AUC |
|---|---:|
| all_benign | 0.9977 |
| all_early | 0.9969 |
| all_pre | 0.9980 |
| all_pro | 0.9986 |

- PR curves:
  - results/comprehensive_eval/figures/pr_curves.png
| Class | AP |
|---|---:|
| all_benign | 0.9931 |
| all_early | 0.9909 |
| all_pre | 0.9955 |
| all_pro | 0.9969 |

- ECE = 0.2031
- Reliability diagram: results/comprehensive_eval/figures/reliability_diagram.png

## 4) Robustness / Generalization Checks
- External test split: NOT AVAILABLE (no external dataset directory provided).
- Staining/scanner domain robustness: NOT AVAILABLE (manifest has only one domain `default`).

## 5) Repeatability / Statistical Confidence
- Seeds: [42, 52, 62]
- Epoch values for evaluated checkpoints: [2, 2, 2]
- Hardware (from checkpoints): {'platform': 'macOS-26.2-arm64-arm-64bit', 'python': '3.12.4', 'device': 'mps', 'cuda_available': False, 'mps_available': True}
- 95% bootstrap CI (1,000 resamples):
| Metric | Value | 95% CI Low | 95% CI High |
|---|---:|---:|---:|
| accuracy | 0.9683 | 0.9620 | 0.9747 |
| balanced_accuracy | 0.9683 | 0.9619 | 0.9744 |
| macro_f1 | 0.9683 | 0.9620 | 0.9745 |
| micro_f1 | 0.9683 | 0.9620 | 0.9747 |
- Baseline p-values: NOT AVAILABLE (no baseline checkpoints provided).

## 6) Explainability / Grad-CAM Metrics
- Representative Grad-CAM panels (correct + misclassified): results/comprehensive_eval/gradcam
- Quantitative mask-based Grad-CAM metrics (pointing game, IoU, CoM distance, AIC, deletion/insertion, localization ROC/PR, sanity checks, expert correlation): NOT AVAILABLE (MASK_ROOT / EXPERT_MAP_ROOT not provided).

## 7) Failure Analysis
- Low-confidence (< 0.60) = 13.03%
- Errors = 95 / 3000
- Misclassification heatmap: results/comprehensive_eval/figures/misclassification_pairs_heatmap.png
| True Class | Predicted Class | Count |
|---|---|---:|
| all_benign | all_early | 33 |
| all_early | all_benign | 29 |
| all_pre | all_pro | 9 |
| all_pre | all_early | 9 |
| all_pro | all_pre | 5 |
| all_benign | all_pre | 5 |
| all_benign | all_pro | 2 |
| all_pre | all_benign | 2 |
| all_early | all_pro | 1 |
- Grad-CAM correct-vs-misclassified quantitative comparison: NOT AVAILABLE (requires mask-based Grad-CAM metrics).

## Artifact Index
- Training summary: models/multi_seed_training_summary.json
- Evaluation summary: results/comprehensive_eval/tables/evaluation_master_summary.json
- Per-class table: results/comprehensive_eval/tables/per_class_metrics.csv
- CI table: results/comprehensive_eval/tables/bootstrap_ci_metrics.csv
- Run logs: results/run_logs