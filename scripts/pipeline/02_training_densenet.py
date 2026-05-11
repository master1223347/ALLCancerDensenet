import csv
import json
import os
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Data/config
DATA_DIR = Path('data')
MODELS_DIR = Path('models')
SPLIT_FILE = MODELS_DIR / 'split_indices.json'
CLASS_NAMES_FILE = MODELS_DIR / 'class_names.txt'

def env_bool(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

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

def env_seeds(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return [int(x.strip()) for x in raw.split(',') if x.strip()]

IMAGE_SIZE = env_int('IMAGE_SIZE', 224)
BATCH_SIZE = env_int('BATCH_SIZE', 32)
NUM_WORKERS = env_int('NUM_WORKERS', 0)

# Repeatability setup
SEEDS = env_seeds('SEEDS', [42, 52, 62])  # >=3 seeds for mean/std reporting

# Optimization setup
MAX_EPOCHS = env_int('MAX_EPOCHS', 20)
EARLY_STOP_PATIENCE = env_int('EARLY_STOP_PATIENCE', 5)
BASE_LR = env_float('BASE_LR', 1e-4)
FINE_TUNE_LR = env_float('FINE_TUNE_LR', 5e-5)
WEIGHT_DECAY = env_float('WEIGHT_DECAY', 1e-4)
LABEL_SMOOTHING = env_float('LABEL_SMOOTHING', 0.05)

FREEZE_FEATURES_AT_START = env_bool('FREEZE_FEATURES_AT_START', True)
FREEZE_EPOCHS = env_int('FREEZE_EPOCHS', 2)

SAVE_EVERY_EPOCH = env_bool('SAVE_EVERY_EPOCH', True)
SKIP_TRAIN_IF_BEST_EXISTS = env_bool('SKIP_TRAIN_IF_BEST_EXISTS', False)
RESUME_FROM_LATEST = env_bool('RESUME_FROM_LATEST', True)


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
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def hardware_summary():
    return {
        'platform': platform.platform(),
        'python': platform.python_version(),
        'device': str(device),
        'cuda_available': bool(torch.cuda.is_available()),
        'mps_available': bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()),
    }

MODELS_DIR.mkdir(parents=True, exist_ok=True)
print(f'Using device: {device}')
print('Hardware:', hardware_summary())
print(
    f'Config: seeds={SEEDS}, max_epochs={MAX_EPOCHS}, batch_size={BATCH_SIZE}, '
    f'early_stop_patience={EARLY_STOP_PATIENCE}, num_workers={NUM_WORKERS}, '
    f'resume_from_latest={RESUME_FROM_LATEST}'
)


class TransformSubset(Dataset):
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
        return image, target

if not DATA_DIR.exists():
    raise FileNotFoundError(f'Data directory not found: {DATA_DIR.resolve()}')

if not SPLIT_FILE.exists():
    raise FileNotFoundError(
        f'Split file not found at {SPLIT_FILE}. Run notebooks/01_data_prep.ipynb first.'
    )

split_data = json.loads(SPLIT_FILE.read_text(encoding='utf-8'))

base_dataset = datasets.ImageFolder(root=str(DATA_DIR))
class_names = base_dataset.classes
num_classes = len(class_names)
CLASS_NAMES_FILE.write_text('\n'.join(class_names) + '\n', encoding='utf-8')

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = TransformSubset(base_dataset, split_data['train_indices'], train_transform)
val_dataset = TransformSubset(base_dataset, split_data['val_indices'], eval_transform)
test_dataset = TransformSubset(base_dataset, split_data['test_indices'], eval_transform)

print(f'Classes ({num_classes}): {class_names}')
print(f'Split sizes -> train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}')

def build_loaders(seed_for_shuffle: int):
    g = torch.Generator().manual_seed(seed_for_shuffle)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        generator=g,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def build_model():
    try:
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    except AttributeError:
        model = models.densenet121(pretrained=True)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

def make_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)

def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(loader, leave=False, desc='train' if is_train else 'eval'):
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        amp_ctx = torch.autocast(device_type='cuda', enabled=True) if (scaler is not None) else nullcontext()
        with amp_ctx:
            logits = model(images)
            loss = criterion(logits, labels)

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc

def collect_predictions(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)

def compute_main_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def train_one_seed(seed: int):
    set_seed(seed)
    train_loader, val_loader, test_loader = build_loaders(seed)

    seed_dir = MODELS_DIR / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = seed_dir / 'densenet_best.pth'
    latest_ckpt_path = seed_dir / 'densenet_latest.pth'
    history_csv_path = seed_dir / 'history.csv'
    curves_png_path = seed_dir / 'training_curves.png'
    test_metrics_path = seed_dir / 'test_metrics.json'

    if SKIP_TRAIN_IF_BEST_EXISTS and best_ckpt_path.exists():
        print(f'[seed {seed}] Best checkpoint exists, skipping training')
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model = build_model().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        y_true, y_pred = collect_predictions(model, test_loader)
        metrics = compute_main_metrics(y_true, y_pred)
        metrics['seed'] = seed
        return metrics

    model = build_model().to(device)

    if FREEZE_FEATURES_AT_START and FREEZE_EPOCHS > 0:
        for p in model.features.parameters():
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = make_optimizer(model, BASE_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    history = []
    epochs_without_improvement = 0
    unfrozen = not (FREEZE_FEATURES_AT_START and FREEZE_EPOCHS > 0)
    start_epoch = 1

    if RESUME_FROM_LATEST and latest_ckpt_path.exists():
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint.get('optimizer_state_dict') is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as exc:
                print(f'[seed {seed}] Could not load optimizer state, using fresh optimizer: {exc}')
        if checkpoint.get('scheduler_state_dict') is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as exc:
                print(f'[seed {seed}] Could not load scheduler state, using fresh scheduler: {exc}')
        if use_amp and checkpoint.get('scaler_state_dict') is not None:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception as exc:
                print(f'[seed {seed}] Could not load scaler state, using fresh scaler: {exc}')

        best_val_loss = float(checkpoint.get('best_val_loss', best_val_loss))
        history = list(checkpoint.get('history', []))
        start_epoch = int(checkpoint.get('epoch', 0)) + 1
        epochs_without_improvement = 0
        unfrozen = start_epoch > FREEZE_EPOCHS or (not FREEZE_FEATURES_AT_START)
        print(f'[seed {seed}] Resuming from epoch {start_epoch} using {latest_ckpt_path}')

    if FREEZE_FEATURES_AT_START and start_epoch > FREEZE_EPOCHS:
        if any(not p.requires_grad for p in model.features.parameters()):
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = make_optimizer(model, FINE_TUNE_LR)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
            )
            unfrozen = True
            print(f'[seed {seed}] Restored unfrozen feature extractor for resumed training at epoch {start_epoch}')

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_start = time.time()

        if (not unfrozen) and (epoch > FREEZE_EPOCHS):
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = make_optimizer(model, FINE_TUNE_LR)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
            )
            unfrozen = True
            print(f'[seed {seed}] Unfroze feature extractor at epoch {epoch}')

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer, scaler=scaler if use_amp else None)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, scaler=None)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr,
            'epoch_time_sec': epoch_time,
        })

        ckpt = {
            'seed': seed,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if use_amp else None,
            'best_val_loss': best_val_loss,
            'class_names': class_names,
            'history': history,
            'hardware': hardware_summary(),
            'config': {
                'max_epochs': MAX_EPOCHS,
                'batch_size': BATCH_SIZE,
                'freeze_epochs': FREEZE_EPOCHS,
                'base_lr': BASE_LR,
                'fine_tune_lr': FINE_TUNE_LR,
                'weight_decay': WEIGHT_DECAY,
                'label_smoothing': LABEL_SMOOTHING,
            },
        }

        if SAVE_EVERY_EPOCH:
            torch.save(ckpt, latest_ckpt_path)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            ckpt['best_val_loss'] = best_val_loss
            torch.save(ckpt, best_ckpt_path)
            epochs_without_improvement = 0
            marker = 'BEST'
        else:
            epochs_without_improvement += 1
            marker = '-'

        print(
            f'[seed {seed}] epoch {epoch:02d}/{MAX_EPOCHS} | '
            f'train_loss {train_loss:.4f} train_acc {train_acc:.4f} | '
            f'val_loss {val_loss:.4f} val_acc {val_acc:.4f} | '
            f'lr {lr:.2e} | {marker} | {epoch_time:.1f}s'
        )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f'[seed {seed}] Early stopping triggered after epoch {epoch}')
            break

    with open(history_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [h['train_loss'] for h in history], label='train')
    axes[0].plot(epochs, [h['val_loss'] for h in history], label='val')
    axes[0].set_title(f'Seed {seed} Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(epochs, [h['train_acc'] for h in history], label='train')
    axes[1].plot(epochs, [h['val_acc'] for h in history], label='val')
    axes[1].set_title(f'Seed {seed} Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(curves_png_path, dpi=250)
    plt.close(fig)

    best_checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    y_true, y_pred = collect_predictions(model, test_loader)
    metrics = compute_main_metrics(y_true, y_pred)
    metrics['seed'] = seed
    metrics['best_epoch'] = int(best_checkpoint.get('epoch', -1))
    metrics['best_val_loss'] = float(best_checkpoint.get('best_val_loss', np.nan))

    test_metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'[seed {seed}] Test metrics: {metrics}')
    return metrics


seed_results = []
for seed in SEEDS:
    result = train_one_seed(seed)
    seed_results.append(result)

print('Completed seeds:', [r['seed'] for r in seed_results])


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)

metric_keys = ['accuracy', 'balanced_accuracy', 'macro_f1', 'micro_f1', 'macro_recall']

summary = {
    'seeds': [int(r['seed']) for r in seed_results],
    'num_seeds': len(seed_results),
    'hardware': hardware_summary(),
    'max_epochs': MAX_EPOCHS,
    'batch_size': BATCH_SIZE,
    'metrics': {},
    'per_seed': seed_results,
}

for k in metric_keys:
    m, s = mean_std([r[k] for r in seed_results])
    summary['metrics'][k] = {'mean': m, 'std': s}

summary_path = MODELS_DIR / 'multi_seed_training_summary.json'
summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

print('\nOverall mean ± std (test)')
print(f"Accuracy = {summary['metrics']['accuracy']['mean']:.4f} ± {summary['metrics']['accuracy']['std']:.4f}")
print(f"Balanced accuracy = {summary['metrics']['balanced_accuracy']['mean']:.4f} ± {summary['metrics']['balanced_accuracy']['std']:.4f}")
print(f"Macro F1 = {summary['metrics']['macro_f1']['mean']:.4f} ± {summary['metrics']['macro_f1']['std']:.4f}")
print(f"Micro F1 = {summary['metrics']['micro_f1']['mean']:.4f} ± {summary['metrics']['micro_f1']['std']:.4f}")
print(f"Macro Recall = {summary['metrics']['macro_recall']['mean']:.4f} ± {summary['metrics']['macro_recall']['std']:.4f}")

with open(MODELS_DIR / 'multi_seed_training_summary.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'mean', 'std'])
    for k in metric_keys:
        writer.writerow([k, summary['metrics'][k]['mean'], summary['metrics'][k]['std']])

print(f'Saved: {summary_path}')
print(f'Saved: {MODELS_DIR / "multi_seed_training_summary.csv"}')
