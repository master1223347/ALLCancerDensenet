import copy
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.transforms import InterpolationMode

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Required core paths
DATA_DIR = Path('data')
MODELS_DIR = Path('models')
RESULTS_DIR = Path('results/comprehensive_eval')
SPLIT_FILE = MODELS_DIR / 'split_indices.json'
CLASS_NAMES_FILE = MODELS_DIR / 'class_names.txt'
MANIFEST_FILE = MODELS_DIR / 'dataset_manifest.csv'

def env_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)

def env_float(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)

def env_bool(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

def env_path(name, default=None):
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    return Path(raw)

def env_csv_paths(name, default=None):
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default if default is not None else []
    return [x.strip() for x in raw.split(',') if x.strip()]

# Optional paths for advanced metrics
EXTERNAL_DATA_DIR = env_path('EXTERNAL_DATA_DIR', None)   # e.g. Path('data_external')
MASK_ROOT = env_path('MASK_ROOT', None)                   # e.g. Path('masks')
EXPERT_MAP_ROOT = env_path('EXPERT_MAP_ROOT', None)       # e.g. Path('expert_maps')

# Optional baseline checkpoints to compute p-values vs baseline
BASELINE_CHECKPOINTS = env_csv_paths('BASELINE_CHECKPOINTS', [])  # csv list of paths

# Evaluation config
IMAGE_SIZE = env_int('IMAGE_SIZE', 224)
BATCH_SIZE = env_int('BATCH_SIZE', 32)
NUM_WORKERS = env_int('NUM_WORKERS', 0)
SEED = env_int('SEED', 42)

CONF_THRESHOLD_LOW = env_float('CONF_THRESHOLD_LOW', 0.60)
BOOTSTRAP_ROUNDS = env_int('BOOTSTRAP_ROUNDS', 1000)

ENABLE_GRADCAM = env_bool('ENABLE_GRADCAM', True)
MAX_GRADCAM_VIS_PER_GROUP = env_int('MAX_GRADCAM_VIS_PER_GROUP', 8)
MAX_GRADCAM_METRIC_SAMPLES = env_int('MAX_GRADCAM_METRIC_SAMPLES', 300)
DELETION_INSERTION_STEPS = env_int('DELETION_INSERTION_STEPS', 11)

(RESULTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'gradcam').mkdir(parents=True, exist_ok=True)


try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    skimage_ssim = None

gradcam_available = False
if ENABLE_GRADCAM:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        gradcam_available = True
    except ImportError:
        print('Grad-CAM package not installed. Run: pip install grad-cam')

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
print(f'Using device: {device}')
print(f'SciPy available: {SCIPY_AVAILABLE}')
print(f'scikit-image available: {SKIMAGE_AVAILABLE}')
print(f'Grad-CAM available: {gradcam_available}')
print(
    f'Config: batch_size={BATCH_SIZE}, bootstrap_rounds={BOOTSTRAP_ROUNDS}, '
    f'low_conf_threshold={CONF_THRESHOLD_LOW}, mask_root={MASK_ROOT}, external_data_dir={EXTERNAL_DATA_DIR}'
)


class EvalSubset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        path, target = self.base_dataset.samples[sample_idx]
        image = self.base_dataset.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target, path, sample_idx

if not DATA_DIR.exists():
    raise FileNotFoundError(f'Data directory not found: {DATA_DIR.resolve()}')
if not SPLIT_FILE.exists():
    raise FileNotFoundError(f'Split file missing: {SPLIT_FILE}. Run 01_data_prep.ipynb first.')

split_data = json.loads(SPLIT_FILE.read_text(encoding='utf-8'))

base_dataset = datasets.ImageFolder(root=str(DATA_DIR))
class_names = base_dataset.classes
num_classes = len(class_names)

if CLASS_NAMES_FILE.exists():
    loaded_classes = [x.strip() for x in CLASS_NAMES_FILE.read_text(encoding='utf-8').splitlines() if x.strip()]
    if loaded_classes and len(loaded_classes) == len(class_names):
        class_names = loaded_classes

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = EvalSubset(base_dataset, split_data['test_indices'], eval_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f'Classes ({num_classes}): {class_names}')
print(f'Test set size: {len(test_dataset)}')

checkpoint_paths = sorted(MODELS_DIR.glob('seed_*/densenet_best.pth'))
if not checkpoint_paths:
    legacy_best = MODELS_DIR / 'densenetAllBest.pth'
    if legacy_best.exists():
        checkpoint_paths = [legacy_best]

if not checkpoint_paths:
    raise FileNotFoundError('No best checkpoints found. Train models first.')

print('Checkpoints:')
for p in checkpoint_paths:
    print('-', p)

primary_checkpoint = checkpoint_paths[0]


def build_model(num_classes: int):
    try:
        model = models.densenet121(weights=None)
    except TypeError:
        model = models.densenet121(pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

def load_model_from_checkpoint(ckpt_path: Path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_model(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

def evaluate_checkpoint(ckpt_path: Path):
    model, checkpoint = load_model_from_checkpoint(ckpt_path)

    all_labels, all_preds, all_probs, all_paths, all_indices = [], [], [], [], []

    with torch.no_grad():
        for images, labels, paths, sample_indices in tqdm(test_loader, desc=f'eval {ckpt_path.parent.name}', leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_paths.extend(list(paths))
            all_indices.extend(sample_indices.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    prob = np.array(all_probs)

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'checkpoint': str(ckpt_path),
        'seed': checkpoint.get('seed', None),
        'epoch': checkpoint.get('epoch', None),
        'best_val_loss': checkpoint.get('best_val_loss', checkpoint.get('val_loss', None)),
        'hardware': checkpoint.get('hardware', None),
        'config': checkpoint.get('config', None),
    }

    return {
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred,
        'probs': prob,
        'paths': all_paths,
        'indices': all_indices,
        'checkpoint': checkpoint,
        'model': model,
        'ckpt_path': str(ckpt_path),
    }

run_results = [evaluate_checkpoint(p) for p in checkpoint_paths]
primary_run = run_results[0]

print('Run metrics:')
for r in run_results:
    m = r['metrics']
    print(f"{Path(m['checkpoint']).name}: acc={m['accuracy']:.4f}, bal_acc={m['balanced_accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}")


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)

def format_mean_std(values):
    m, s = mean_std(values)
    return f'{m:.4f} ± {s:.4f}'

def compute_per_class_table(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    rows = []
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        support = int(cm[i, :].sum())

        rows.append({
            'class': cls,
            'precision': precision,
            'recall_sensitivity': recall,
            'f1_score': f1,
            'specificity': specificity,
            'support': support,
        })
    return rows, cm

metric_keys = ['accuracy', 'balanced_accuracy', 'macro_f1', 'micro_f1', 'macro_recall']
aggregated_metrics = {}
for k in metric_keys:
    aggregated_metrics[k] = {
        'values': [r['metrics'][k] for r in run_results],
        'formatted': format_mean_std([r['metrics'][k] for r in run_results]),
    }

print('Overall classification metrics')
print(f"Accuracy = {aggregated_metrics['accuracy']['formatted']}")
print(f"Balanced accuracy = {aggregated_metrics['balanced_accuracy']['formatted']}")
print(f"Macro F1 = {aggregated_metrics['macro_f1']['formatted']}")
print(f"Micro F1 = {aggregated_metrics['micro_f1']['formatted']}")
print(f"Macro Recall = {aggregated_metrics['macro_recall']['formatted']}")

per_class_rows, cm_counts = compute_per_class_table(primary_run['y_true'], primary_run['y_pred'], class_names)

per_class_csv = RESULTS_DIR / 'tables' / 'per_class_metrics.csv'
with open(per_class_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['class', 'precision', 'recall_sensitivity', 'f1_score', 'specificity', 'support'])
    writer.writeheader()
    writer.writerows(per_class_rows)

print(f'Saved per-class metrics: {per_class_csv}')


y_true = primary_run['y_true']
y_pred = primary_run['y_pred']
probs = primary_run['probs']

report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
(RESULTS_DIR / 'tables' / 'classification_report.txt').write_text(report_txt, encoding='utf-8')
print(report_txt)

cm_norm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes), normalize='true')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(ax=axes[0], cmap='Blues', colorbar=False, xticks_rotation=45, values_format='.2f')
axes[0].set_title('Confusion Matrix (Normalized, Row-wise)')

ConfusionMatrixDisplay(cm_counts, display_labels=class_names).plot(ax=axes[1], cmap='Blues', colorbar=False, xticks_rotation=45)
axes[1].set_title('Confusion Matrix (Absolute Counts)')

plt.tight_layout()
cm_fig = RESULTS_DIR / 'figures' / 'confusion_matrices.png'
plt.savefig(cm_fig, dpi=300)
plt.show()
plt.close(fig)

np.savetxt(RESULTS_DIR / 'tables' / 'confusion_matrix_counts.csv', cm_counts, delimiter=',', fmt='%d')
np.savetxt(RESULTS_DIR / 'tables' / 'confusion_matrix_normalized.csv', cm_norm, delimiter=',', fmt='%.6f')

print(f'Saved confusion matrices: {cm_fig}')


# ROC and Precision-Recall (one-vs-rest)
y_bin = label_binarize(y_true, classes=np.arange(num_classes))

roc_rows = []
pr_rows = []

fig_roc, ax_roc = plt.subplots(figsize=(8, 7))
fig_pr, ax_pr = plt.subplots(figsize=(8, 7))

for i, cls in enumerate(class_names):
    y_i = y_bin[:, i]
    score_i = probs[:, i]

    if len(np.unique(y_i)) < 2:
        roc_auc = np.nan
        ap = np.nan
        continue

    fpr, tpr, _ = roc_curve(y_i, score_i)
    precision_curve, recall_curve, _ = precision_recall_curve(y_i, score_i)
    roc_auc = roc_auc_score(y_i, score_i)
    ap = average_precision_score(y_i, score_i)

    roc_rows.append({'class': cls, 'auc': roc_auc})
    pr_rows.append({'class': cls, 'ap': ap})

    ax_roc.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.3f})')
    ax_pr.plot(recall_curve, precision_curve, label=f'{cls} (AP={ap:.3f})')

ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax_roc.set_title('ROC Curves (One-vs-Rest)')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.legend(loc='lower right')

ax_pr.set_title('Precision-Recall Curves (One-vs-Rest)')
ax_pr.set_xlabel('Recall')
ax_pr.set_ylabel('Precision')
ax_pr.legend(loc='lower left')

fig_roc.tight_layout()
fig_pr.tight_layout()

roc_fig_path = RESULTS_DIR / 'figures' / 'roc_curves.png'
pr_fig_path = RESULTS_DIR / 'figures' / 'pr_curves.png'
fig_roc.savefig(roc_fig_path, dpi=300)
fig_pr.savefig(pr_fig_path, dpi=300)
plt.show()
plt.close(fig_roc)
plt.close(fig_pr)

with open(RESULTS_DIR / 'tables' / 'auc_per_class.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['class', 'auc'])
    writer.writeheader()
    writer.writerows(roc_rows)

with open(RESULTS_DIR / 'tables' / 'ap_per_class.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['class', 'ap'])
    writer.writeheader()
    writer.writerows(pr_rows)

print(f'Saved ROC figure: {roc_fig_path}')
print(f'Saved PR figure: {pr_fig_path}')


def expected_calibration_error(probs, y_true, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_acc, bin_conf, bin_count = [], [], []

    for b in range(n_bins):
        left, right = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        if mask.sum() == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_count.append(0)
            continue

        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        frac = mask.mean()
        ece += abs(acc - conf) * frac

        bin_acc.append(acc)
        bin_conf.append(conf)
        bin_count.append(int(mask.sum()))

    return float(ece), bin_edges, np.array(bin_acc), np.array(bin_conf), np.array(bin_count)

ece, bin_edges, bin_acc, bin_conf, bin_count = expected_calibration_error(probs, y_true, n_bins=15)
print(f'ECE = {ece:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

valid = ~np.isnan(bin_acc)
centers = (bin_edges[:-1] + bin_edges[1:]) / 2

axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0].bar(centers[valid], bin_acc[valid], width=1/15, alpha=0.7, label='Empirical accuracy')
axes[0].plot(centers[valid], bin_conf[valid], 'o-', label='Mean confidence')
axes[0].set_title(f'Reliability Diagram (ECE={ece:.4f})')
axes[0].set_xlabel('Confidence')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].bar(centers, bin_count, width=1/15, alpha=0.7)
axes[1].set_title('Confidence Histogram')
axes[1].set_xlabel('Confidence bin')
axes[1].set_ylabel('Count')

plt.tight_layout()
rel_path = RESULTS_DIR / 'figures' / 'reliability_diagram.png'
plt.savefig(rel_path, dpi=300)
plt.show()
plt.close(fig)

with open(RESULTS_DIR / 'tables' / 'calibration_metrics.json', 'w', encoding='utf-8') as f:
    json.dump({'ece': ece, 'bin_edges': bin_edges.tolist(), 'bin_acc': bin_acc.tolist(), 'bin_conf': bin_conf.tolist(), 'bin_count': bin_count.tolist()}, f, indent=2)

print(f'Saved reliability diagram: {rel_path}')


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats_vals.append(metric_fn(y_true[idx], y_pred[idx]))
    low, high = np.percentile(stats_vals, [2.5, 97.5])
    return float(low), float(high)

ci_metrics = {
    'accuracy': lambda a, b: accuracy_score(a, b),
    'balanced_accuracy': lambda a, b: balanced_accuracy_score(a, b),
    'macro_f1': lambda a, b: f1_score(a, b, average='macro', zero_division=0),
    'micro_f1': lambda a, b: f1_score(a, b, average='micro', zero_division=0),
}

ci_table = []
for name, fn in ci_metrics.items():
    lo, hi = bootstrap_ci(y_true, y_pred, fn, n_boot=BOOTSTRAP_ROUNDS, seed=SEED)
    point = fn(y_true, y_pred)
    ci_table.append({'metric': name, 'value': point, 'ci95_low': lo, 'ci95_high': hi})
    print(f'{name}: {point:.4f} (95% CI {lo:.4f}, {hi:.4f})')

with open(RESULTS_DIR / 'tables' / 'bootstrap_ci_metrics.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['metric', 'value', 'ci95_low', 'ci95_high'])
    writer.writeheader()
    writer.writerows(ci_table)

# Optional baseline p-values (paired bootstrap difference)
def paired_bootstrap_pvalue(y_true, pred_a, pred_b, metric_fn, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(min(1.0, p)), float(diffs.mean())

baseline_rows = []
if BASELINE_CHECKPOINTS:
    for p in BASELINE_CHECKPOINTS:
        p = Path(p)
        if not p.exists():
            print(f'Baseline checkpoint missing: {p}')
            continue
        base_run = evaluate_checkpoint(p)
        p_acc, diff_acc = paired_bootstrap_pvalue(y_true, y_pred, base_run['y_pred'], ci_metrics['accuracy'])
        p_f1, diff_f1 = paired_bootstrap_pvalue(y_true, y_pred, base_run['y_pred'], ci_metrics['macro_f1'])
        baseline_rows.append({
            'baseline_checkpoint': str(p),
            'accuracy_diff': diff_acc,
            'accuracy_pvalue': p_acc,
            'macro_f1_diff': diff_f1,
            'macro_f1_pvalue': p_f1,
        })
    with open(RESULTS_DIR / 'tables' / 'baseline_comparison_pvalues.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['baseline_checkpoint', 'accuracy_diff', 'accuracy_pvalue', 'macro_f1_diff', 'macro_f1_pvalue'])
        writer.writeheader()
        writer.writerows(baseline_rows)
    print('Saved baseline p-value table')
else:
    print('BASELINE_CHECKPOINTS empty -> skipped baseline p-value comparison')


# Robustness checks: external split + domain/staining/scanner slices
def evaluate_external_split(external_dir, ckpt_path):
    external_dir = Path(external_dir)
    if not external_dir.exists():
        print(f'External data directory missing: {external_dir}')
        return None

    ext_ds = datasets.ImageFolder(root=str(external_dir), transform=eval_transform)
    ext_loader = DataLoader(ext_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model, _ = load_model_from_checkpoint(Path(ckpt_path))
    y_t, y_p = [], []
    with torch.no_grad():
        for images, labels in ext_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().numpy()
            y_p.extend(preds.tolist())
            y_t.extend(labels.numpy().tolist())

    y_t = np.array(y_t)
    y_p = np.array(y_p)
    return {
        'accuracy': float(accuracy_score(y_t, y_p)),
        'balanced_accuracy': float(balanced_accuracy_score(y_t, y_p)),
        'macro_f1': float(f1_score(y_t, y_p, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_t, y_p, average='micro', zero_division=0)),
        'support': int(len(y_t)),
    }

robustness_rows = []

if EXTERNAL_DATA_DIR is not None:
    ext_metrics = evaluate_external_split(EXTERNAL_DATA_DIR, primary_checkpoint)
    if ext_metrics is not None:
        ext_metrics['split'] = 'external'
        robustness_rows.append(ext_metrics)

# Domain-level robustness if manifest exists and has domain info
if MANIFEST_FILE.exists():
    idx_to_domain = {}
    with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('split') == 'test':
                idx_to_domain[int(row['index'])] = row.get('domain', 'default')

    domain_groups = defaultdict(list)
    for i, sample_idx in enumerate(primary_run['indices']):
        domain = idx_to_domain.get(int(sample_idx), 'default')
        domain_groups[domain].append(i)

    if len(domain_groups) > 1:
        for domain, idxs in sorted(domain_groups.items()):
            yt = y_true[idxs]
            yp = y_pred[idxs]
            robustness_rows.append({
                'split': f'domain:{domain}',
                'accuracy': float(accuracy_score(yt, yp)),
                'balanced_accuracy': float(balanced_accuracy_score(yt, yp)),
                'macro_f1': float(f1_score(yt, yp, average='macro', zero_division=0)),
                'micro_f1': float(f1_score(yt, yp, average='micro', zero_division=0)),
                'support': int(len(idxs)),
            })
    else:
        print('Domain robustness skipped: only one domain present in test manifest')
else:
    print('Domain robustness skipped: dataset_manifest.csv not found')

if robustness_rows:
    with open(RESULTS_DIR / 'tables' / 'robustness_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'accuracy', 'balanced_accuracy', 'macro_f1', 'micro_f1', 'support'])
        writer.writeheader()
        writer.writerows(robustness_rows)
    print('Saved robustness_metrics.csv')
else:
    print('No external/domain robustness rows generated with current inputs')


# Failure analysis
confidences = probs.max(axis=1)
low_conf_mask = confidences < CONF_THRESHOLD_LOW
low_conf_pct = 100.0 * low_conf_mask.mean()
print(f'Low-confidence predictions (<{CONF_THRESHOLD_LOW:.2f}): {low_conf_pct:.2f}%')

# Misclassification pair table
pair_counter = Counter()
for t, p in zip(y_true, y_pred):
    if t != p:
        pair_counter[(class_names[t], class_names[p])] += 1

misclass_rows = [
    {'true_class': t, 'predicted_class': p, 'count': c}
    for (t, p), c in pair_counter.most_common()
]

with open(RESULTS_DIR / 'tables' / 'misclassification_pairs.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['true_class', 'predicted_class', 'count'])
    writer.writeheader()
    writer.writerows(misclass_rows)

# Heatmap style matrix for misclassifications
misclass_mat = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    if t != p:
        misclass_mat[t, p] += 1

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(misclass_mat, cmap='Reds')
ax.set_xticks(np.arange(num_classes), labels=class_names, rotation=45, ha='right')
ax.set_yticks(np.arange(num_classes), labels=class_names)
ax.set_title('Misclassification Pairs (True → Pred)')
for i in range(num_classes):
    for j in range(num_classes):
        if misclass_mat[i, j] > 0:
            ax.text(j, i, str(misclass_mat[i, j]), ha='center', va='center', color='black', fontsize=9)
fig.colorbar(im, ax=ax)
plt.tight_layout()
misclass_fig = RESULTS_DIR / 'figures' / 'misclassification_pairs_heatmap.png'
plt.savefig(misclass_fig, dpi=300)
plt.show()
plt.close(fig)

failure_summary = {
    'low_confidence_threshold': CONF_THRESHOLD_LOW,
    'low_confidence_percent': float(low_conf_pct),
    'num_errors': int((y_true != y_pred).sum()),
    'num_samples': int(len(y_true)),
}
(RESULTS_DIR / 'tables' / 'failure_summary.json').write_text(json.dumps(failure_summary, indent=2), encoding='utf-8')
print('Saved failure analysis tables/figures')


# Grad-CAM visualization + quantitative explainability metrics (optional)
def unnormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)

def get_dataset_item_by_sample_index(sample_index):
    # sample_index corresponds to original base_dataset index
    path, label = base_dataset.samples[sample_index]
    image = base_dataset.loader(path)
    tensor = eval_transform(image)
    return tensor, label, path

def resolve_mask_path(image_path):
    if MASK_ROOT is None:
        return None
    mask_root = Path(MASK_ROOT)
    image_path = Path(image_path)
    rel = image_path.relative_to(DATA_DIR).as_posix()
    stem = image_path.stem
    candidates = [
        mask_root / rel,
        mask_root / f'{stem}_mask.png',
        mask_root / f'{stem}.png',
        mask_root / image_path.parent.name / f'{stem}_mask.png',
        mask_root / image_path.parent.name / f'{stem}.png',
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def load_binary_mask(mask_path):
    mask = Image.open(mask_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST)
    arr = np.array(mask)
    return (arr > 127).astype(np.uint8)

def normalized_cam_map(cam_map):
    cam_map = np.maximum(cam_map, 0)
    mx = cam_map.max()
    if mx <= 1e-12:
        return np.zeros_like(cam_map)
    return cam_map / mx

def cam_centroid(cam_map):
    h, w = cam_map.shape
    yy, xx = np.mgrid[0:h, 0:w]
    m = cam_map.sum()
    if m <= 1e-12:
        return None
    cy = float((yy * cam_map).sum() / m)
    cx = float((xx * cam_map).sum() / m)
    return cy, cx

def binary_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else np.nan

def top_fraction_mask(cam_map, frac):
    threshold = np.quantile(cam_map, 1.0 - frac)
    return cam_map >= threshold

def flatten_image_with_mask(image_tensor, pixel_mask_2d):
    x = image_tensor.clone()
    x[:, pixel_mask_2d] = 0.0
    return x

def deletion_insertion_auc(model, image_tensor, pred_class, cam_map, steps=11, random_baseline=False):
    h, w = cam_map.shape
    flat_idx = np.arange(h * w)
    if random_baseline:
        rng = np.random.default_rng(SEED)
        rng.shuffle(flat_idx)
    else:
        flat_idx = np.argsort(cam_map.reshape(-1))[::-1]

    fractions = np.linspace(0, 1, steps)
    del_scores = []
    ins_scores = []

    orig = image_tensor.clone()
    base = torch.zeros_like(orig)

    with torch.no_grad():
        for frac in fractions:
            k = int(frac * h * w)
            mask_flat = np.zeros(h * w, dtype=bool)
            if k > 0:
                mask_flat[flat_idx[:k]] = True
            mask = torch.from_numpy(mask_flat.reshape(h, w))

            del_img = orig.clone()
            del_img[:, mask] = 0.0

            ins_img = base.clone()
            ins_img[:, mask] = orig[:, mask]

            del_prob = torch.softmax(model(del_img.unsqueeze(0).to(device)), dim=1)[0, pred_class].item()
            ins_prob = torch.softmax(model(ins_img.unsqueeze(0).to(device)), dim=1)[0, pred_class].item()
            del_scores.append(del_prob)
            ins_scores.append(ins_prob)

    auc_del = float(np.trapz(del_scores, fractions))
    auc_ins = float(np.trapz(ins_scores, fractions))
    return auc_del, auc_ins, fractions, del_scores, ins_scores

def spearman_corr(a, b):
    if SCIPY_AVAILABLE:
        return float(stats.spearmanr(a, b).correlation)
    # fallback: rank-correlation via numpy ranks (ties approximated)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

gradcam_metrics_rows = []

if gradcam_available:
    primary_model = primary_run['model']
    primary_model.eval()
    cam_extractor = GradCAM(model=primary_model, target_layers=[primary_model.features])

    correct_indices = np.where(y_true == y_pred)[0].tolist()
    wrong_indices = np.where(y_true != y_pred)[0].tolist()

    def save_gradcam_examples(example_indices, prefix, max_items=8):
        saved = 0
        for pos in example_indices:
            if saved >= max_items:
                break
            sample_idx = int(primary_run['indices'][pos])
            tensor, label, image_path = get_dataset_item_by_sample_index(sample_idx)
            pred_idx = int(y_pred[pos])
            conf = float(probs[pos, pred_idx])

            input_tensor = tensor.unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(pred_idx)]
            cam_map = cam_extractor(input_tensor=input_tensor, targets=targets)[0]
            cam_map = normalized_cam_map(cam_map)

            rgb = unnormalize(tensor)
            viz = show_cam_on_image(rgb.astype(np.float32), cam_map, use_rgb=True)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(rgb)
            axes[0].set_title(f'Original\nT:{class_names[label]}')
            axes[0].axis('off')

            axes[1].imshow(viz)
            axes[1].set_title(f'Grad-CAM\nP:{class_names[pred_idx]} ({conf:.2f})')
            axes[1].axis('off')

            plt.tight_layout()
            out = RESULTS_DIR / 'gradcam' / f'{prefix}_{saved + 1}.png'
            plt.savefig(out, dpi=300)
            plt.show()
            plt.close(fig)
            saved += 1

    save_gradcam_examples(correct_indices, 'correct', max_items=MAX_GRADCAM_VIS_PER_GROUP)
    save_gradcam_examples(wrong_indices, 'misclassified', max_items=MAX_GRADCAM_VIS_PER_GROUP)

    # Quantitative metrics requiring masks
    if MASK_ROOT is not None:
        eval_positions = list(range(len(y_true)))
        random.shuffle(eval_positions)
        eval_positions = eval_positions[:MAX_GRADCAM_METRIC_SAMPLES]

        for pos in tqdm(eval_positions, desc='Grad-CAM quantitative metrics', leave=False):
            sample_idx = int(primary_run['indices'][pos])
            tensor, label, image_path = get_dataset_item_by_sample_index(sample_idx)
            pred_idx = int(y_pred[pos])
            correct = int(label == pred_idx)

            mask_path = resolve_mask_path(image_path)
            if mask_path is None:
                continue
            gt_mask = load_binary_mask(mask_path)
            if gt_mask.sum() == 0:
                continue

            input_tensor = tensor.unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(pred_idx)]
            cam_map = cam_extractor(input_tensor=input_tensor, targets=targets)[0]
            cam_map = normalized_cam_map(cam_map)

            # Pointing game
            yx = np.unravel_index(np.argmax(cam_map), cam_map.shape)
            top1_inside = int(gt_mask[yx] > 0)

            c = cam_centroid(cam_map)
            if c is None:
                continue
            cy, cx = c
            cy_i = int(np.clip(round(cy), 0, IMAGE_SIZE - 1))
            cx_i = int(np.clip(round(cx), 0, IMAGE_SIZE - 1))
            centroid_inside = int(gt_mask[cy_i, cx_i] > 0)

            cam_top10 = top_fraction_mask(cam_map, 0.10)
            cam_top20 = top_fraction_mask(cam_map, 0.20)
            iou10 = binary_iou(cam_top10, gt_mask.astype(bool))
            iou20 = binary_iou(cam_top20, gt_mask.astype(bool))

            ys, xs = np.where(gt_mask > 0)
            gt_cy, gt_cx = ys.mean(), xs.mean()
            dist = math.sqrt((cy - gt_cy) ** 2 + (cx - gt_cx) ** 2)
            dist_norm = float(dist / math.sqrt(IMAGE_SIZE ** 2 + IMAGE_SIZE ** 2))

            # Confidence drop/increase under masking top salient region
            with torch.no_grad():
                orig_prob = torch.softmax(primary_model(input_tensor), dim=1)[0, pred_idx].item()
                masked = tensor.clone()
                masked[:, cam_top20] = 0.0
                masked_prob = torch.softmax(primary_model(masked.unsqueeze(0).to(device)), dim=1)[0, pred_idx].item()

            avg_drop = float(max(0.0, orig_prob - masked_prob) / max(orig_prob, 1e-8))
            avg_increase = float(max(0.0, masked_prob - orig_prob) / max(orig_prob, 1e-8))

            auc_del, auc_ins, _, _, _ = deletion_insertion_auc(primary_model, tensor, pred_idx, cam_map, steps=DELETION_INSERTION_STEPS, random_baseline=False)
            auc_del_rand, auc_ins_rand, _, _, _ = deletion_insertion_auc(primary_model, tensor, pred_idx, cam_map, steps=DELETION_INSERTION_STEPS, random_baseline=True)

            gt_flat = gt_mask.reshape(-1)
            cam_flat = cam_map.reshape(-1)
            if len(np.unique(gt_flat)) > 1:
                loc_roc_auc = float(roc_auc_score(gt_flat, cam_flat))
                loc_pr_auc = float(average_precision_score(gt_flat, cam_flat))
            else:
                loc_roc_auc = np.nan
                loc_pr_auc = np.nan

            # Stability under transforms
            flip_tensor = torch.flip(tensor, dims=[2])
            flip_cam = cam_extractor(input_tensor=flip_tensor.unsqueeze(0).to(device), targets=targets)[0]
            flip_cam = normalized_cam_map(np.fliplr(flip_cam))

            bright_tensor = torch.clamp(tensor + 0.05, min=-3.0, max=3.0)
            bright_cam = cam_extractor(input_tensor=bright_tensor.unsqueeze(0).to(device), targets=targets)[0]
            bright_cam = normalized_cam_map(bright_cam)

            rot_tensor = TF.rotate(tensor, angle=10, interpolation=InterpolationMode.BILINEAR)
            rot_cam = cam_extractor(input_tensor=rot_tensor.unsqueeze(0).to(device), targets=targets)[0]
            rot_cam_t = torch.from_numpy(rot_cam).unsqueeze(0)
            rot_cam_back = TF.rotate(rot_cam_t, angle=-10, interpolation=InterpolationMode.BILINEAR).squeeze(0).numpy()
            rot_cam_back = normalized_cam_map(rot_cam_back)

            def stability_pair(cam_a, cam_b):
                iou = binary_iou(top_fraction_mask(cam_a, 0.20), top_fraction_mask(cam_b, 0.20))
                corr = spearman_corr(cam_a.reshape(-1), cam_b.reshape(-1))
                return iou, corr

            flip_iou, flip_corr = stability_pair(cam_map, flip_cam)
            bright_iou, bright_corr = stability_pair(cam_map, bright_cam)
            rot_iou, rot_corr = stability_pair(cam_map, rot_cam_back)

            stability_iou = float(np.nanmean([flip_iou, bright_iou, rot_iou]))
            stability_corr = float(np.nanmean([flip_corr, bright_corr, rot_corr]))

            # Optional correlation with expert map
            spearman_expert = np.nan
            ssim_expert = np.nan
            if EXPERT_MAP_ROOT is not None:
                expert_root = Path(EXPERT_MAP_ROOT)
                rel = Path(image_path).relative_to(DATA_DIR)
                expert_candidates = [
                    expert_root / rel,
                    expert_root / f'{Path(image_path).stem}.png',
                    expert_root / rel.parent / f'{Path(image_path).stem}.png',
                ]
                expert_path = None
                for c_path in expert_candidates:
                    if c_path.exists():
                        expert_path = c_path
                        break
                if expert_path is not None:
                    expert_map = Image.open(expert_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
                    expert_arr = np.array(expert_map, dtype=np.float32)
                    if expert_arr.max() > 0:
                        expert_arr /= expert_arr.max()
                    spearman_expert = spearman_corr(cam_flat, expert_arr.reshape(-1))
                    if SKIMAGE_AVAILABLE:
                        ssim_expert = float(skimage_ssim(cam_map, expert_arr, data_range=1.0))

            gradcam_metrics_rows.append({
                'sample_index': sample_idx,
                'image_path': str(image_path),
                'true_class': class_names[label],
                'pred_class': class_names[pred_idx],
                'correct': correct,
                'pointing_top1_inside': top1_inside,
                'pointing_centroid_inside': centroid_inside,
                'iou_top10': iou10,
                'iou_top20': iou20,
                'com_distance_norm': dist_norm,
                'average_drop': avg_drop,
                'average_increase': avg_increase,
                'auc_deletion': auc_del,
                'auc_insertion': auc_ins,
                'auc_deletion_random': auc_del_rand,
                'auc_insertion_random': auc_ins_rand,
                'pixel_roc_auc': loc_roc_auc,
                'pixel_pr_auc': loc_pr_auc,
                'stability_iou': stability_iou,
                'stability_corr': stability_corr,
                'spearman_expert': spearman_expert,
                'ssim_expert': ssim_expert,
            })

        if gradcam_metrics_rows:
            gradcam_csv = RESULTS_DIR / 'tables' / 'gradcam_quantitative_metrics.csv'
            with open(gradcam_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(gradcam_metrics_rows[0].keys()))
                writer.writeheader()
                writer.writerows(gradcam_metrics_rows)
            print(f'Saved Grad-CAM quantitative metrics: {gradcam_csv}')

            # Aggregate per class + overall
            def summarize(rows, key):
                vals = np.array([r[key] for r in rows if r[key] == r[key]], dtype=float)
                if len(vals) == 0:
                    return np.nan, np.nan
                return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)

            summary_rows = []
            groups = {'overall': gradcam_metrics_rows}
            for cls in class_names:
                groups[cls] = [r for r in gradcam_metrics_rows if r['true_class'] == cls]

            metrics_to_summarize = [
                'pointing_top1_inside', 'pointing_centroid_inside', 'iou_top10', 'iou_top20',
                'com_distance_norm', 'average_drop', 'average_increase', 'auc_deletion', 'auc_insertion',
                'auc_deletion_random', 'auc_insertion_random', 'pixel_roc_auc', 'pixel_pr_auc',
                'stability_iou', 'stability_corr', 'spearman_expert', 'ssim_expert'
            ]

            for group_name, rows in groups.items():
                if not rows:
                    continue
                row_out = {'group': group_name, 'support': len(rows)}
                for m in metrics_to_summarize:
                    mean_v, std_v = summarize(rows, m)
                    row_out[f'{m}_mean'] = mean_v
                    row_out[f'{m}_std'] = std_v
                summary_rows.append(row_out)

            summary_csv = RESULTS_DIR / 'tables' / 'gradcam_quantitative_summary.csv'
            with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f'Saved Grad-CAM summary: {summary_csv}')

            # Compare Grad-CAM metrics for correct vs misclassified samples
            corr_rows = [r for r in gradcam_metrics_rows if r['correct'] == 1]
            wrong_rows = [r for r in gradcam_metrics_rows if r['correct'] == 0]
            compare_metrics = ['iou_top20', 'pointing_top1_inside', 'com_distance_norm', 'pixel_roc_auc']
            comparison = []

            def permutation_pvalue(a, b, rounds=2000):
                a = np.asarray(a, dtype=float)
                b = np.asarray(b, dtype=float)
                a = a[~np.isnan(a)]
                b = b[~np.isnan(b)]
                if len(a) < 2 or len(b) < 2:
                    return np.nan
                obs = a.mean() - b.mean()
                combined = np.concatenate([a, b])
                n_a = len(a)
                rng = np.random.default_rng(SEED)
                hits = 0
                for _ in range(rounds):
                    rng.shuffle(combined)
                    diff = combined[:n_a].mean() - combined[n_a:].mean()
                    if abs(diff) >= abs(obs):
                        hits += 1
                return (hits + 1) / (rounds + 1)

            for m in compare_metrics:
                a = np.array([r[m] for r in corr_rows], dtype=float)
                b = np.array([r[m] for r in wrong_rows], dtype=float)
                a = a[~np.isnan(a)]
                b = b[~np.isnan(b)]
                if len(a) == 0 or len(b) == 0:
                    continue

                if SCIPY_AVAILABLE and len(a) > 1 and len(b) > 1:
                    p = float(stats.ttest_ind(a, b, equal_var=False, nan_policy='omit').pvalue)
                else:
                    p = float(permutation_pvalue(a, b))

                comparison.append({
                    'metric': m,
                    'correct_mean': float(a.mean()),
                    'misclassified_mean': float(b.mean()),
                    'difference': float(a.mean() - b.mean()),
                    'p_value': p,
                })

            if comparison:
                comp_csv = RESULTS_DIR / 'tables' / 'gradcam_correct_vs_misclassified_stats.csv'
                with open(comp_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
                    writer.writeheader()
                    writer.writerows(comparison)
                print(f'Saved Grad-CAM correct-vs-misclassified stats: {comp_csv}')

            # Sanity checks: parameter randomization + label permutation (requires masks)
            # Pointing accuracy drop is reported with p-values.
            if gradcam_metrics_rows:
                normal_pointing = np.array([r['pointing_top1_inside'] for r in gradcam_metrics_rows], dtype=float)

                # Parameter randomization
                randomized_model = copy.deepcopy(primary_model)
                for module in randomized_model.modules():
                    if hasattr(module, 'reset_parameters'):
                        try:
                            module.reset_parameters()
                        except Exception:
                            pass
                randomized_model = randomized_model.to(device).eval()
                rand_cam = GradCAM(model=randomized_model, target_layers=[randomized_model.features])

                # Label permutation (target class shuffled)
                perm_rng = np.random.default_rng(SEED)
                perm_targets = perm_rng.integers(0, num_classes, size=len(gradcam_metrics_rows))

                rand_pointing = []
                perm_pointing = []
                for j, row in enumerate(tqdm(gradcam_metrics_rows, desc='Sanity checks', leave=False)):
                    tensor = eval_transform(base_dataset.loader(row['image_path']))
                    gt_mask = load_binary_mask(resolve_mask_path(row['image_path']))

                    # Randomized-weights CAM
                    pred_cls = class_names.index(row['pred_class'])
                    cam_rand = rand_cam(input_tensor=tensor.unsqueeze(0).to(device), targets=[ClassifierOutputTarget(pred_cls)])[0]
                    cam_rand = normalized_cam_map(cam_rand)
                    yx_rand = np.unravel_index(np.argmax(cam_rand), cam_rand.shape)
                    rand_pointing.append(int(gt_mask[yx_rand] > 0))

                    # Label permutation CAM on original model
                    perm_cls = int(perm_targets[j])
                    cam_perm = cam_extractor(input_tensor=tensor.unsqueeze(0).to(device), targets=[ClassifierOutputTarget(perm_cls)])[0]
                    cam_perm = normalized_cam_map(cam_perm)
                    yx_perm = np.unravel_index(np.argmax(cam_perm), cam_perm.shape)
                    perm_pointing.append(int(gt_mask[yx_perm] > 0))

                def simple_p(a, b):
                    return permutation_pvalue(np.array(a, dtype=float), np.array(b, dtype=float), rounds=2000)

                sanity_rows = [
                    {
                        'test': 'parameter_randomization',
                        'normal_pointing_mean': float(np.mean(normal_pointing)),
                        'sanity_pointing_mean': float(np.mean(rand_pointing)),
                        'drop': float(np.mean(normal_pointing) - np.mean(rand_pointing)),
                        'p_value': float(simple_p(normal_pointing, rand_pointing)),
                    },
                    {
                        'test': 'label_permutation',
                        'normal_pointing_mean': float(np.mean(normal_pointing)),
                        'sanity_pointing_mean': float(np.mean(perm_pointing)),
                        'drop': float(np.mean(normal_pointing) - np.mean(perm_pointing)),
                        'p_value': float(simple_p(normal_pointing, perm_pointing)),
                    },
                ]

                with open(RESULTS_DIR / 'tables' / 'gradcam_sanity_checks.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=list(sanity_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(sanity_rows)
                print('Saved Grad-CAM sanity checks')
        else:
            print('No mask-aligned samples found. Grad-CAM quantitative mask metrics skipped.')
    else:
        print('MASK_ROOT is None. Skipped mask-based Grad-CAM quantitative metrics.')
else:
    print('Grad-CAM unavailable. Skipped explainability panels and quantitative Grad-CAM metrics.')


# Final consolidated summary
final_summary = {
    'num_checkpoints_evaluated': len(run_results),
    'checkpoints': [r['metrics']['checkpoint'] for r in run_results],
    'overall_metrics_mean_std': {k: aggregated_metrics[k]['formatted'] for k in aggregated_metrics},
    'primary_checkpoint': str(primary_checkpoint),
    'primary_metrics': primary_run['metrics'],
    'ece': ece,
    'low_confidence_percent': float(low_conf_pct),
    'seed_values': [r['metrics'].get('seed') for r in run_results],
    'epoch_values': [r['metrics'].get('epoch') for r in run_results],
    'hardware': [r['metrics'].get('hardware') for r in run_results],
    'bootstrap_rounds': BOOTSTRAP_ROUNDS,
    'gradcam_enabled': bool(gradcam_available),
    'mask_root': str(MASK_ROOT) if MASK_ROOT is not None else None,
    'external_data_dir': str(EXTERNAL_DATA_DIR) if EXTERNAL_DATA_DIR is not None else None,
}

summary_path = RESULTS_DIR / 'tables' / 'evaluation_master_summary.json'
summary_path.write_text(json.dumps(final_summary, indent=2), encoding='utf-8')
print(f'Saved final summary: {summary_path}')
print(f'All outputs saved under: {RESULTS_DIR}')
