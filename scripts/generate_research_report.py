import csv
import json
from pathlib import Path

ROOT = Path('.')
MODELS = ROOT / 'models'
RESULTS = ROOT / 'results' / 'comprehensive_eval'
TABLES = RESULTS / 'tables'
FIGS = RESULTS / 'figures'
GRADCAM = RESULTS / 'gradcam'

ms = json.loads((MODELS / 'multi_seed_training_summary.json').read_text())
master = json.loads((TABLES / 'evaluation_master_summary.json').read_text())
per_class = list(csv.DictReader((TABLES / 'per_class_metrics.csv').open()))
auc_rows = list(csv.DictReader((TABLES / 'auc_per_class.csv').open()))
ap_rows = list(csv.DictReader((TABLES / 'ap_per_class.csv').open()))
ci_rows = list(csv.DictReader((TABLES / 'bootstrap_ci_metrics.csv').open()))
failure = json.loads((TABLES / 'failure_summary.json').read_text())
mis = list(csv.DictReader((TABLES / 'misclassification_pairs.csv').open()))
cal = json.loads((TABLES / 'calibration_metrics.json').read_text())

seed_values = master.get('seed_values', [])
epoch_values = master.get('epoch_values', [])
hw = master.get('hardware', [])

m = ms['metrics']
def pm(key):
    return f"{m[key]['mean']:.4f} ± {m[key]['std']:.4f}"

lines = []
lines.append('# Research-Grade Results Summary')
lines.append('')
lines.append('## 1) Overall Classification Metrics')
lines.append(f"- Accuracy = {pm('accuracy')}")
lines.append(f"- Balanced accuracy = {pm('balanced_accuracy')}")
lines.append(f"- Macro F1 = {pm('macro_f1')}")
lines.append(f"- Macro Recall = {pm('macro_recall')}")
lines.append(f"- Micro F1 = {pm('micro_f1')}")
lines.append('')

lines.append('## 2) Per-Class Classification Metrics')
lines.append('| Class | Precision | Recall (Sensitivity) | F1-score | Specificity | Support |')
lines.append('|---|---:|---:|---:|---:|---:|')
for r in per_class:
    lines.append(f"| {r['class']} | {float(r['precision']):.4f} | {float(r['recall_sensitivity']):.4f} | {float(r['f1_score']):.4f} | {float(r['specificity']):.4f} | {int(r['support'])} |")
lines.append('')
lines.append('- Confusion matrix (normalized row-wise + absolute counts):')
lines.append(f"  - {FIGS / 'confusion_matrices.png'}")
lines.append(f"  - {TABLES / 'confusion_matrix_normalized.csv'}")
lines.append(f"  - {TABLES / 'confusion_matrix_counts.csv'}")
lines.append('')

lines.append('## 3) ROC / PR / Calibration')
lines.append('- ROC curves:')
lines.append(f"  - {FIGS / 'roc_curves.png'}")
lines.append('| Class | AUC |')
lines.append('|---|---:|')
for r in auc_rows:
    lines.append(f"| {r['class']} | {float(r['auc']):.4f} |")
lines.append('')
lines.append('- PR curves:')
lines.append(f"  - {FIGS / 'pr_curves.png'}")
lines.append('| Class | AP |')
lines.append('|---|---:|')
for r in ap_rows:
    lines.append(f"| {r['class']} | {float(r['ap']):.4f} |")
lines.append('')
lines.append(f"- ECE = {float(cal['ece']):.4f}")
lines.append(f"- Reliability diagram: {FIGS / 'reliability_diagram.png'}")
lines.append('')

lines.append('## 4) Robustness / Generalization Checks')
lines.append('- External test split: NOT AVAILABLE (no external dataset directory provided).')
lines.append('- Staining/scanner domain robustness: NOT AVAILABLE (manifest has only one domain `default`).')
lines.append('')

lines.append('## 5) Repeatability / Statistical Confidence')
lines.append(f"- Seeds: {seed_values}")
lines.append(f"- Epoch values for evaluated checkpoints: {epoch_values}")
if hw:
    lines.append(f"- Hardware (from checkpoints): {hw[0]}")
lines.append('- 95% bootstrap CI (1,000 resamples):')
lines.append('| Metric | Value | 95% CI Low | 95% CI High |')
lines.append('|---|---:|---:|---:|')
for r in ci_rows:
    lines.append(f"| {r['metric']} | {float(r['value']):.4f} | {float(r['ci95_low']):.4f} | {float(r['ci95_high']):.4f} |")
lines.append('- Baseline p-values: NOT AVAILABLE (no baseline checkpoints provided).')
lines.append('')

lines.append('## 6) Explainability / Grad-CAM Metrics')
lines.append(f"- Representative Grad-CAM panels (correct + misclassified): {GRADCAM}")
lines.append('- Quantitative mask-based Grad-CAM metrics (pointing game, IoU, CoM distance, AIC, deletion/insertion, localization ROC/PR, sanity checks, expert correlation): NOT AVAILABLE (MASK_ROOT / EXPERT_MAP_ROOT not provided).')
lines.append('')

lines.append('## 7) Failure Analysis')
lines.append(f"- Low-confidence (< {failure['low_confidence_threshold']:.2f}) = {failure['low_confidence_percent']:.2f}%")
lines.append(f"- Errors = {failure['num_errors']} / {failure['num_samples']}")
lines.append(f"- Misclassification heatmap: {FIGS / 'misclassification_pairs_heatmap.png'}")
lines.append('| True Class | Predicted Class | Count |')
lines.append('|---|---|---:|')
for r in mis:
    lines.append(f"| {r['true_class']} | {r['predicted_class']} | {int(r['count'])} |")
lines.append('- Grad-CAM correct-vs-misclassified quantitative comparison: NOT AVAILABLE (requires mask-based Grad-CAM metrics).')
lines.append('')

lines.append('## Artifact Index')
lines.append(f"- Training summary: {MODELS / 'multi_seed_training_summary.json'}")
lines.append(f"- Evaluation summary: {TABLES / 'evaluation_master_summary.json'}")
lines.append(f"- Per-class table: {TABLES / 'per_class_metrics.csv'}")
lines.append(f"- CI table: {TABLES / 'bootstrap_ci_metrics.csv'}")
lines.append(f"- Run logs: {ROOT / 'results' / 'run_logs'}")

out = ROOT / 'results' / 'research_grade_report.md'
out.write_text('\n'.join(lines), encoding='utf-8')
print(out)
