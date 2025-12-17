#!/usr/bin/env python3
"""
Updated training script with enhanced tracking and storage capabilities:
1. Stores attention maps for all samples after training
2. Tracks comprehensive training statistics (losses, metrics, class distribution, weights)
3. Saves all results for later analysis and visualization
"""

import os
import sys
import json
import pickle
import traceback
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
from torch.utils.data import WeightedRandomSampler
import pandas as pd

# Project path
sys.path.append('.')
sys.path.append('..')
DATA_CONFIG = {
    'data_dir': r'D:\athar_code\results\data\data_processed\results0',  # Your data directory
    'gaze_json_dir': r'/kaggle/input/gazedata/results/gaze',  # Your gaze JSON directory
    'train_subdir': 'train1',  # Subdirectory for training data
    'eval_subdir': 'eval1'     # Subdirectory for evaluation data
}

# Try import project modules; fallback placeholders
try:
    from cereprocess.datasets.pipeline import general_pipeline
    from cereprocess.train.xloop import get_datanpz
    from cereprocess.train.misc import def_dev, def_hyp, EarlyStopping
except Exception:
    def def_dev():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def def_hyp(batch_size=32, lr=2e-4, epochs=50, accum_iter=1):
        return {"batch_size": batch_size, "lr": lr, "epochs": epochs, "accum_iter": accum_iter}

    def get_def_ds(mins):
        return None, ("E:/NMT_events/NMT_events/edf", "NMT", "description", "results"), None

    # REMOVED EarlyStopping class

# Import model + dataset classes
try:
    from neurogate_gaze import NeuroGATE_Gaze_MultiRes
    from eeg_gaze_fixations import EEGGazeFixationDataset
except Exception as e:
    print("Error importing NeuroGATE_Gaze_MultiRes or EEGGazeFixationDataset:", e)
    raise

# ----------------- Configuration -----------------
DEFAULT_SUFFIXES_TO_STRIP = [
    '_clean', '_interp', '_filtered', '_fix', '_fixations', 
    '_epochs', '_epoch', 
    '_p0', '_p1', '_p2', '_p3', '_p4', '_p5',
    '_p0_clean', '_p0_filtered',
    '_session1', '_session2',
    '_run1', '_run2'
]

# ----------------- Utility Functions -----------------
def list_files_recursive(dir_path, ext):
    """Return sorted list of full paths under dir_path with extension ext (case-insensitive)."""
    out = []
    if not dir_path or not os.path.exists(dir_path):
        return out
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(ext.lower()):
                out.append(os.path.join(root, f))
    out.sort()
    return out

def normalize_basename(path_or_name):
    """Return the base filename in lowercase without changing suffixes."""
    if path_or_name is None:
        return None

    # If bytes, decode
    if isinstance(path_or_name, (bytes, bytearray)):
        try:
            path_or_name = path_or_name.decode('utf-8', errors='ignore')
        except:
            path_or_name = str(path_or_name)

    # Get base name without extension, lowercase
    base = os.path.splitext(os.path.basename(str(path_or_name)))[0]
    return base.lower()

# ----------------- Filtered Dataset Class -----------------
class FilteredEEGGazeFixationDataset:
    """
    Wraps EEGGazeFixationDataset and filters it to only include indices whose
    normalized basename exists both as a .npz (under data_dir) and a .json (under gaze_json_dir).
    """

    def __init__(self, data_dir, gaze_json_dir, dataset_cls, dataset_kwargs=None, suffixes_to_strip=None):
        """
        dataset_cls: EEGGazeFixationDataset class
        dataset_kwargs: dict of kwargs to pass to dataset_cls
        suffixes_to_strip: list of suffixes to strip from basenames during normalization
        """
        self.data_dir = data_dir
        self.gaze_json_dir = gaze_json_dir
        self.dataset_cls = dataset_cls
        self.dataset_kwargs = dataset_kwargs or {}
        self.suffixes_to_strip = suffixes_to_strip or DEFAULT_SUFFIXES_TO_STRIP

        # Build on-disk normalized maps
        self._build_on_disk_maps()

        # Instantiate underlying dataset once (unfiltered)
        self.original_dataset = self.dataset_cls(data_dir=self.data_dir, gaze_json_dir=self.gaze_json_dir,
                                                **self.dataset_kwargs)
        # Probe dataset to build index -> normalized basename map
        self._build_dataset_index_map()

        # Compute list of dataset indices to keep (those whose normalized basename appears in disk_matched_basenames)
        self.filtered_indices = self._compute_filtered_indices()

        # Diagnostics printed on creation
        self._print_diagnostics()

    def _build_on_disk_maps(self):
        npz_paths = list_files_recursive(self.data_dir, '.npz')
        json_paths = list_files_recursive(self.gaze_json_dir, '.json')

        self.npz_map = defaultdict(list)
        for p in npz_paths:
            nb = normalize_basename(p)
            self.npz_map[nb].append(p)

        self.json_map = defaultdict(list)
        for p in json_paths:
            nb = normalize_basename(p)
            self.json_map[nb].append(p)

        self.disk_npz_basenames = set(self.npz_map.keys())
        self.disk_json_basenames = set(self.json_map.keys())
        self.disk_matched_basenames = self.disk_npz_basenames & self.disk_json_basenames

    def _build_dataset_index_map(self):
        self.dataset_index_to_base = {}
        self.dataset_base_to_indices = defaultdict(list)
        n = len(self.original_dataset)
        for idx in range(n):
            try:
                sample = self.original_dataset[idx]
                f = None
                if isinstance(sample, dict) and 'file' in sample:
                    f = sample['file']
                elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
                    f = sample[2]
                if isinstance(f, (bytes, bytearray)):
                    try:
                        f = f.decode('utf-8', errors='ignore')
                    except:
                        f = str(f)
                if isinstance(f, str):
                    nb = normalize_basename(f)
                else:
                    nb = None
            except Exception:
                nb = None
            self.dataset_index_to_base[idx] = nb
            if nb:
                self.dataset_base_to_indices[nb].append(idx)

    def _compute_filtered_indices(self):
        kept = []
        for nb in sorted(self.disk_matched_basenames):
            idxs = self.dataset_base_to_indices.get(nb, [])
            if idxs:
                kept.extend(idxs)
        return sorted(set(kept))

    def _print_diagnostics(self):
        print("\n" + "=" * 70)
        print("FILTEREDEEGGAZEFIXATIONDATASET DIAGNOSTICS".center(70))
        print("=" * 70)
        print(f"  data_dir: {self.data_dir}")
        print(f"  gaze_json_dir: {self.gaze_json_dir}")
        print(f"\n  Disk: {len(self.npz_map)} unique npz basenames, {len(self.json_map)} unique json basenames")
        print(f"  Disk matched basenames: {len(self.disk_matched_basenames)}")
        
        if self.disk_matched_basenames:
            print(f"    Examples (first 20): {sorted(list(self.disk_matched_basenames))[:20]}")
        
        dataset_bases = set(k for k in self.dataset_base_to_indices.keys() if k)
        print(f"\n  Dataset reported basenames: {len(dataset_bases)}")
        
        if dataset_bases:
            print(f"    Examples (first 20): {sorted(list(dataset_bases))[:20]}")
        
        on_disk_not_in_dataset = sorted(list(self.disk_matched_basenames - dataset_bases))
        in_dataset_not_on_disk = sorted(list(dataset_bases - self.disk_matched_basenames))
        
        print(f"\n  Normalized diff counts:")
        print(f"    On-disk matched but NOT in dataset: {len(on_disk_not_in_dataset)}")
        if on_disk_not_in_dataset:
            print(f"      Examples: {on_disk_not_in_dataset[:10]}")
        print(f"    In dataset but NOT matched on-disk: {len(in_dataset_not_on_disk)}")
        if in_dataset_not_on_disk:
            print(f"      Examples: {in_dataset_not_on_disk[:10]}")
        
        print(f"\n  Filtered indices kept: {len(self.filtered_indices)} (out of original {len(self.original_dataset)})")
        if self.filtered_indices:
            print(f"    Sample kept basenames (first 20): {[self.dataset_index_to_base[i] for i in self.filtered_indices[:20]]}")
        
        print("=" * 70)

    # Dataset protocol
    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        orig_idx = self.filtered_indices[idx]
        return self.original_dataset[orig_idx]

# ----------------- DataDebugger Class -----------------
class DataDebugger:
    """Data debugging helpers for comprehensive dataset analysis."""
    
    @staticmethod
    def print_header(title, width=80, char="="):
        """Print a formatted header."""
        print("\n" + char * width)
        print(title.center(width))
        print(char * width)
    
    @staticmethod
    def analyze_dataset(dataset, name="Dataset", max_samples=20):
        """Analyze dataset content and statistics."""
        DataDebugger.print_header(f"ANALYZE {name}")
        print(f"  Total samples: {len(dataset)}")
        
        if len(dataset) == 0:
            print("  WARNING: Dataset is empty!")
            return None, None
        
        sample_count = min(max_samples, len(dataset))
        all_labels = []
        all_files = []
        
        for i in range(sample_count):
            try:
                sample = dataset[i]
                if isinstance(sample, dict):
                    label = sample.get('label', None)
                    eeg = sample.get('eeg', None)
                    gaze = sample.get('gaze', None)
                    
                    f = sample.get('file', None)
                    if f is not None:
                        if isinstance(f, (bytes, bytearray)):
                            try:
                                f = f.decode('utf-8', errors='ignore')
                            except:
                                f = str(f)
                        basename = os.path.splitext(os.path.basename(str(f)))[0]
                        all_files.append(basename)
                    
                    all_labels.append(label)
                    
                    print(f"  Sample {i}: Label={label}")
                    if eeg is not None:
                        print(f"    EEG shape: {eeg.shape}, dtype: {eeg.dtype}")
                    if gaze is not None:
                        print(f"    Gaze shape: {gaze.shape}, dtype: {gaze.dtype}")
                    if f is not None:
                        print(f"    File: {basename[:50]}")
                else:
                    print(f"  Sample {i}: Not a dict, type: {type(sample)}")
                    
            except Exception as e:
                print(f"  Sample {i}: Error reading sample: {e}")
                traceback.print_exc()
        
        if all_labels:
            counter = Counter(all_labels)
            print(f"\n  Label distribution (sampled): {dict(counter)}")
            print(f"  Number of unique labels: {len(counter)}")
        
        # Check data types
        if sample_count > 0:
            try:
                sample = dataset[0]
                if isinstance(sample, dict):
                    print(f"\n  Data types in first sample:")
                    for key, value in sample.items():
                        if torch.is_tensor(value):
                            print(f"    {key}: Tensor shape={value.shape}, dtype={value.dtype}")
                        elif isinstance(value, np.ndarray):
                            print(f"    {key}: Numpy shape={value.shape}, dtype={value.dtype}")
                        elif isinstance(value, (list, tuple)):
                            print(f"    {key}: {type(value).__name__} length={len(value)}")
                        else:
                            print(f"    {key}: {type(value).__name__}")
            except:
                pass
        
        return all_labels, all_files
    
    @staticmethod
    def analyze_dataloader(dataloader, name="Dataloader", max_batches=3):
        """Analyze dataloader batches."""
        DataDebugger.print_header(f"ANALYZE {name}")
        print(f"  Total batches: {len(dataloader)}")
        
        if len(dataloader) == 0:
            print("  WARNING: Dataloader is empty!")
            return None, None
        
        all_labels = []
        all_files = []
        
        for bidx, batch in enumerate(dataloader):
            if bidx >= max_batches:
                break
            
            print(f"\n  Batch {bidx+1}:")
            
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                        if key == 'label':
                            all_labels.extend(value.numpy().tolist())
                            print(f"      labels: {value.numpy().tolist()}")
                        if key == 'file':
                            files = []
                            for f in value:
                                if isinstance(f, (bytes, bytearray)):
                                    try:
                                        f = f.decode('utf-8', errors='ignore')
                                    except:
                                        f = str(f)
                                files.append(os.path.splitext(os.path.basename(str(f)))[0])
                            all_files.extend(files)
                            print(f"      files: {files[:3]}...")
                    elif isinstance(value, list):
                        print(f"    {key}: list length={len(value)}")
                    else:
                        print(f"    {key}: {type(value)}")
            
            elif isinstance(batch, (list, tuple)):
                print(f"    Batch is {type(batch).__name__} with {len(batch)} elements")
                for i, item in enumerate(batch):
                    if torch.is_tensor(item):
                        print(f"      [{i}]: shape={item.shape}, dtype={item.dtype}")
        
        if all_labels:
            counter = Counter(all_labels)
            print(f"\n  Label distribution in seen batches: {dict(counter)}")
            print(f"  Total samples seen: {len(all_labels)}")
        
        return all_labels, all_files

# ----------------- Dataloader Builder -----------------
def get_dataloaders_fixed(data_dir, batch_size, seed, target_length=None, indexes=None,
                         gaze_json_dir=None, only_matched=True,
                         suffixes_to_strip=None, **kwargs):
    """
    Build dataloaders using FilteredEEGGazeFixationDataset
    """
    DataDebugger.print_header("BUILD DATALOADERS (FIXED)")
    
    # Use configurable subdirectories
    train_dir = os.path.join(data_dir, kwargs.get('train_subdir', 'train'))
    eval_dir = os.path.join(data_dir, kwargs.get('eval_subdir', 'eval'))
    
    print(f"  Main data_dir: {data_dir}")
    print(f"  Train directory: {train_dir}")
    print(f"  Eval directory: {eval_dir}")
    print(f"  Gaze JSON directory: {gaze_json_dir}")
    # instantiate the filtered wrapper for train and eval
    dataset_kwargs = {
        'indexes': indexes,
        'target_length': target_length,
        'eeg_sampling_rate': kwargs.get('eeg_sampling_rate', 50.0)
    }

    trainset = FilteredEEGGazeFixationDataset(
        data_dir=train_dir,
        gaze_json_dir=gaze_json_dir,
        dataset_cls=EEGGazeFixationDataset,
        dataset_kwargs=dataset_kwargs,
        suffixes_to_strip=suffixes_to_strip or DEFAULT_SUFFIXES_TO_STRIP
    )

    labels = [trainset[i]['label'] for i in range(len(trainset))]
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    total_samples = len(labels)
    print(f"\n  Training label distribution: {dict(class_counts)}")
    print(f"  Number of classes: {num_classes}, Total training samples: {total_samples}")
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Assign a weight to each sample
    sample_weights = [class_weights[label] for label in labels]

    # Create sampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    evalset = FilteredEEGGazeFixationDataset(
        data_dir=eval_dir,
        gaze_json_dir=gaze_json_dir,
        dataset_cls=EEGGazeFixationDataset,
        dataset_kwargs=dataset_kwargs,
        suffixes_to_strip=suffixes_to_strip or DEFAULT_SUFFIXES_TO_STRIP
    )

    # If not only_matched, you might want to use original_dataset instead; keep current behavior
    # Attach gaze_json_dir metadata on Subset-like wrappers (for downstream code)
    for ds in (trainset, evalset):
        # if ds is a wrapper with original_dataset attribute and not Subset, set attribute on wrapper
        if hasattr(ds, 'gaze_json_dir'):
            ds.gaze_json_dir = gaze_json_dir

    # Create DataLoaders
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=min(batch_size, len(trainset)) if len(trainset) > 0 else 1,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    eval_loader = torch.utils.data.DataLoader(
        evalset,
        batch_size=min(batch_size, len(evalset)) if len(evalset) > 0 else 1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    print("\nDATALOADER SUMMARY")
    print(f"  Train samples: {len(trainset)} | batches: {len(train_loader)}")
    print(f"  Eval samples:  {len(evalset)} | batches: {len(eval_loader)}")

    # Quick dataset diagnostics
    DataDebugger.analyze_dataset(trainset, "Filtered Train Dataset", max_samples=5)
    DataDebugger.analyze_dataset(evalset, "Filtered Eval Dataset", max_samples=5)

    return train_loader, eval_loader, {
        'train_filtered': len(trainset),
        'eval_filtered': len(evalset),
        'train_disk_matched': len(trainset.disk_matched_basenames) if hasattr(trainset, 'disk_matched_basenames') else 0,
        'eval_disk_matched': len(evalset.disk_matched_basenames) if hasattr(evalset, 'disk_matched_basenames') else 0
    }

# ----------------- Training Statistics Tracker -----------------
class TrainingStatistics:
    """Comprehensive tracker for all training metrics and statistics."""
    
    def __init__(self, output_dir='training_stats'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize storage containers
        self.epoch_stats = []
        self.batch_stats = []
        self.attention_maps = defaultdict(list)
        self.model_weights = []
        self.class_distributions = []
        self.gaze_stats = []
        self.confusion_matrices = []
        self.predictions = defaultdict(list)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
    def record_epoch(self, epoch, train_stats, eval_stats, model=None):
        """Record comprehensive epoch statistics."""
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_stats.get('loss', 0),
            'train_cls_loss': train_stats.get('cls_loss', 0),
            'train_gaze_loss': train_stats.get('gaze_loss', 0),
            'train_acc': train_stats.get('acc', 0),
            'eval_acc': eval_stats.get('acc', 0),
            'eval_loss': eval_stats.get('loss', 0),  # ADD THIS LINE
            'eval_cls_loss': eval_stats.get('cls_loss', 0),  # ADD THIS LINE
            'eval_gaze_loss': eval_stats.get('gaze_loss', 0),  # ADD THIS LINE
            'eval_macro_f1': eval_stats.get('macro_f1', 0),
            'eval_balanced_acc': eval_stats.get('balanced_acc', 0),
            'eval_weighted_f1': eval_stats.get('weighted_f1', 0),
            'eval_precision': eval_stats.get('precision', 0),
            'eval_recall': eval_stats.get('recall', 0),
            'lr': train_stats.get('lr', 0),
            'train_gaze_batches': train_stats.get('gaze_batches', 0),
            'train_gaze_samples': train_stats.get('gaze_samples', 0),
            'eval_gaze_batches': eval_stats.get('gaze_batches', 0),  # ADD THIS LINE
            'timestamp': datetime.now().isoformat()
        }
        self.epoch_stats.append(epoch_data)
        
        # Store model weights if requested
        if model is not None:
            self._record_model_weights(model, epoch)
        
        # Save intermediate results
        self._save_intermediate_results()
        
        return epoch_data
    
    def record_batch(self, batch_idx, batch_stats):
        """Record batch-level statistics."""
        batch_data = {
            'batch_idx': batch_idx,
            **batch_stats
        }
        self.batch_stats.append(batch_data)
    
    def record_attention_maps(self, batch_files, attention_maps, labels, predictions, gaze_maps=None):
        """Store attention maps for later analysis."""
        batch_data = []
        for i, file in enumerate(batch_files):
            att_map = attention_maps[i].cpu().numpy() if torch.is_tensor(attention_maps[i]) else attention_maps[i]
            gaze_map = gaze_maps[i].cpu().numpy() if gaze_maps is not None and i < len(gaze_maps) else None
            
            sample_data = {
                'file': file,
                'attention_map': att_map,
                'label': labels[i].item() if torch.is_tensor(labels[i]) else labels[i],
                'prediction': predictions[i].item() if torch.is_tensor(predictions[i]) else predictions[i],
                'gaze_map': gaze_map,
                'timestamp': datetime.now().isoformat()
            }
            batch_data.append(sample_data)
        
        self.attention_maps['samples'].extend(batch_data)
    
    def record_class_distribution(self, dataloader, name="train"):
        """Record class distribution in a dataloader."""
        all_labels = []
        for batch in dataloader:
            labels = batch['label'].numpy()
            all_labels.extend(labels.tolist())
        
        distribution = Counter(all_labels)
        self.class_distributions.append({
            'dataset': name,
            'distribution': dict(distribution),
            'total_samples': len(all_labels),
            'timestamp': datetime.now().isoformat()
        })
        
        return distribution
    
    def record_confusion_matrix(self, y_true, y_pred, name="eval"):
        """Record confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrices.append({
            'dataset': name,
            'matrix': cm.tolist(),
            'labels': np.unique(np.concatenate([y_true, y_pred])).tolist(),
            'timestamp': datetime.now().isoformat()
        })
        return cm
    
    def record_predictions(self, files, true_labels, predictions, probabilities=None, dataset="eval"):
        """Store predictions for detailed analysis."""
        for i, file in enumerate(files):
            # Handle probability conversion - it might be a list or numpy array
            prob = probabilities[i] if probabilities is not None else None
            if prob is not None:
                if hasattr(prob, 'tolist'):  # numpy array
                    prob = prob.tolist()
                # If it's already a list, keep it as is
            
            pred_data = {
                'file': file,
                'true_label': true_labels[i],
                'predicted_label': predictions[i],
                'probability': prob,
                'dataset': dataset,
                'correct': true_labels[i] == predictions[i],
                'timestamp': datetime.now().isoformat()
            }
            self.predictions[dataset].append(pred_data)
    
    def _record_model_weights(self, model, epoch):
        """Record model weights and gradients for analysis."""
        weight_data = {'epoch': epoch, 'weights': {}, 'gradients': {}}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_data['weights'][name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item()
                }
                if param.grad is not None:
                    weight_data['gradients'][name] = {
                        'mean': param.grad.mean().item(),
                        'std': param.grad.std().item(),
                        'norm': param.grad.norm().item()
                    }
        
        self.model_weights.append(weight_data)
    
    def _save_intermediate_results(self):
        """Save intermediate results to disk."""
        # Save epoch stats
        if self.epoch_stats:
            pd.DataFrame(self.epoch_stats).to_csv(self.run_dir / 'epoch_stats.csv', index=False)
        
        # Save batch stats (sampled to avoid huge files)
        if self.batch_stats and len(self.batch_stats) > 1000:
            # Save only every 10th batch
            sampled_batch_stats = self.batch_stats[::10]
            pd.DataFrame(sampled_batch_stats).to_csv(self.run_dir / 'batch_stats_sampled.csv', index=False)
        
        # Save class distributions
        if self.class_distributions:
            with open(self.run_dir / 'class_distributions.pkl', 'wb') as f:
                pickle.dump(self.class_distributions, f)
    
    def save_final_results(self, model=None, attention_maps=None):
        """Save all collected statistics to disk."""
        print(f"\nSaving training statistics to: {self.run_dir}")
        
        # 1. Save epoch statistics
        epoch_df = pd.DataFrame(self.epoch_stats)
        epoch_df.to_csv(self.run_dir / 'epoch_statistics.csv', index=False)
        
        # 2. Save training history plots
        self._create_training_plots()
        
        # 3. Save attention maps
        if attention_maps:
            with open(self.run_dir / 'attention_maps.pkl', 'wb') as f:
                pickle.dump(attention_maps, f)
            print(f"  - Saved attention maps")
        
        # 4. Save model weights history
        if self.model_weights:
            with open(self.run_dir / 'model_weights_history.pkl', 'wb') as f:
                pickle.dump(self.model_weights, f)
        
        # 5. Save class distributions
        if self.class_distributions:
            class_dist_df = pd.DataFrame(self.class_distributions)
            class_dist_df.to_csv(self.run_dir / 'class_distributions.csv', index=False)
        
        # 6. Save confusion matrices
        if self.confusion_matrices:
            with open(self.run_dir / 'confusion_matrices.pkl', 'wb') as f:
                pickle.dump(self.confusion_matrices, f)
        
        # 7. Save predictions
        for dataset_name, pred_list in self.predictions.items():
            if pred_list:
                pred_df = pd.DataFrame(pred_list)
                pred_df.to_csv(self.run_dir / f'predictions_{dataset_name}.csv', index=False)
        
        # 8. Save gaze statistics
        if self.gaze_stats:
            gaze_df = pd.DataFrame(self.gaze_stats)
            gaze_df.to_csv(self.run_dir / 'gaze_statistics.csv', index=False)
        
        # 9. Save configuration and metadata
        metadata = {
            'timestamp': self.timestamp,
            'total_epochs': len(self.epoch_stats),
            'total_batches': len(self.batch_stats),
            'total_attention_maps': len(self.attention_maps.get('samples', [])),
            'run_directory': str(self.run_dir)
        }
        with open(self.run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 10. Save model architecture and final state
        if model is not None:
            torch.save(model.state_dict(), self.run_dir / 'final_model.pth')
            # Save model summary
            with open(self.run_dir / 'model_summary.txt', 'w') as f:
                f.write(str(model))
                f.write(f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
                f.write(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        print(f"Training statistics saved successfully!")
        print(f"  - CSV files: epoch_statistics.csv, class_distributions.csv")
        print(f"  - Attention maps: attention_maps.pkl")
        print(f"  - Plots: training_*.png")
        print(f"  - Model: final_model.pth")
    
    def _create_training_plots(self):
        """Create visualization plots from training statistics."""
        if not self.epoch_stats:
            return
        
        epochs = [s['epoch'] for s in self.epoch_stats]
        
        # Plot 1: Loss curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training losses
        ax = axes[0, 0]
        ax.plot(epochs, [s['train_loss'] for s in self.epoch_stats], label='Total Loss', linewidth=2)
        ax.plot(epochs, [s['train_cls_loss'] for s in self.epoch_stats], label='Classification Loss', linestyle='--')
        ax.plot(epochs, [s['train_gaze_loss'] for s in self.epoch_stats], label='Gaze Loss', linestyle=':')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax = axes[0, 1]
        ax.plot(epochs, [s['train_acc'] for s in self.epoch_stats], label='Train Accuracy', linewidth=2)
        ax.plot(epochs, [s['eval_acc'] for s in self.epoch_stats], label='Eval Accuracy', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training vs Evaluation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 scores
        ax = axes[1, 0]
        ax.plot(epochs, [s['eval_macro_f1'] for s in self.epoch_stats], label='Macro F1', linewidth=2)
        ax.plot(epochs, [s['eval_weighted_f1'] for s in self.epoch_stats], label='Weighted F1', linestyle='--')
        ax.plot(epochs, [s['eval_balanced_acc'] for s in self.epoch_stats], label='Balanced Acc', linestyle=':')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Evaluation Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision-Recall
        ax = axes[1, 1]
        ax.plot(epochs, [s['eval_precision'] for s in self.epoch_stats], label='Precision', linewidth=2)
        ax.plot(epochs, [s['eval_recall'] for s in self.epoch_stats], label='Recall', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Precision and Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Confusion matrix from last epoch
        if self.confusion_matrices:
            last_cm = self.confusion_matrices[-1]
            cm = np.array(last_cm['matrix'])
            labels = last_cm['labels']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
            
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=labels, yticklabels=labels,
                   title='Confusion Matrix',
                   ylabel='True label',
                   xlabel='Predicted label')
            
            plt.tight_layout()
            plt.savefig(self.run_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()

# ----------------- Loss Function -----------------
def compute_gaze_attention_loss(attention_map, gaze, labels, loss_type='mse'):
    """Compute loss between attention maps and gaze maps."""
    if attention_map is None or gaze is None:
        return torch.tensor(0.0).to(attention_map.device if attention_map is not None else gaze.device)
    
    if loss_type == 'mse':
        return F.mse_loss(attention_map, gaze)
    elif loss_type == 'weighted_mse':
        weights = gaze * 2 + 0.1
        return (weights * (attention_map - gaze) ** 2).mean()
    elif loss_type == 'cosine':
        att = attention_map.view(attention_map.shape[0], -1)
        gz = gaze.view(gaze.shape[0], -1)
        return 1 - F.cosine_similarity(att, gz).mean()
    elif loss_type == 'kl':
        att_prob = F.softmax(attention_map.view(attention_map.shape[0], -1), dim=1)
        gaze_prob = F.softmax(gaze.view(gaze.shape[0], -1), dim=1)
        return F.kl_div(att_prob.log(), gaze_prob, reduction='batchmean')
    elif loss_type == 'combined':
        # Combine multiple losses
        mse = F.mse_loss(attention_map, gaze)
        att_flat = attention_map.view(attention_map.shape[0], -1)
        gaze_flat = gaze.view(gaze.shape[0], -1)
        cosine = 1 - F.cosine_similarity(att_flat, gaze_flat).mean()
        return mse + 0.5 * cosine
    else:
        raise ValueError(f"Unknown gaze loss type: {loss_type}")

# ----------------- Updated Training Function -----------------
def train_epoch_with_gaze(model, train_loader, optimizer, device, gaze_weight=1, 
                         gaze_loss_type='cosine', class_weights=None, stats_tracker=None, epoch=0):
    """Enhanced training epoch with comprehensive statistics tracking."""
    model.train()
    total_loss = total_cls = total_gaze = 0.0
    correct = total = 0
    batches_with_gaze = samples_with_gaze = 0
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in pbar:
        eeg = batch['eeg'].to(device)
        labels = batch['label'].to(device)
        
        # Get batch files for tracking
        batch_files = []
        if 'file' in batch:
            for f in batch['file']:
                if isinstance(f, (bytes, bytearray)):
                    try:
                        f = f.decode('utf-8', errors='ignore')
                    except:
                        f = str(f)
                batch_files.append(os.path.basename(str(f)))
        else:
            batch_files = ["unknown"] * eeg.shape[0]
        
        # Check for gaze data
        has_gaze = 'gaze' in batch and batch['gaze'] is not None
        if has_gaze:
            gaze = batch['gaze'].to(device)
            batches_with_gaze += 1
            samples_with_gaze += eeg.shape[0]
        
        # Forward pass
        if has_gaze:
            outputs = model(eeg, return_attention=True)
            if isinstance(outputs, tuple):
                logits, attention_map = outputs
            else:
                logits = outputs['logits']
                attention_map = outputs['attention_map']
        else:
            logits = model(eeg, return_attention=False)
            attention_map = None
        
        # Classification loss
        if class_weights is not None:
            cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            cls_loss = F.cross_entropy(logits, labels)
        
        # Gaze loss if available
        if has_gaze and attention_map is not None:
            gaze_loss = compute_gaze_attention_loss(attention_map, gaze, labels, gaze_loss_type)
            loss = cls_loss + gaze_weight * gaze_loss
        else:
            gaze_loss = torch.tensor(0.0).to(device)
            loss = cls_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_gaze += gaze_loss.item() if has_gaze else 0.0
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Record batch statistics ONLY (NO attention maps during training)
        if stats_tracker:
            batch_stats = {
                'epoch': epoch,
                'batch_loss': loss.item(),
                'batch_cls_loss': cls_loss.item(),
                'batch_gaze_loss': gaze_loss.item() if has_gaze else 0.0,
                'batch_accuracy': (preds == labels).float().mean().item(),
                'has_gaze': has_gaze,
                'lr': current_lr
            }
            stats_tracker.record_batch(batch_idx, batch_stats)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct/total*100:.1f}%",
            'gaze': 'Y' if has_gaze else 'N'
        })
    
    # Calculate epoch averages
    avg_loss = total_loss / max(len(train_loader), 1)
    avg_cls = total_cls / max(len(train_loader), 1)
    avg_gaze = total_gaze / max(len(train_loader), 1)
    acc = correct / total * 100 if total > 0 else 0.0
    
    train_stats = {
        'loss': avg_loss,
        'cls_loss': avg_cls,
        'gaze_loss': avg_gaze,
        'acc': acc,
        'gaze_batches': batches_with_gaze,
        'gaze_samples': samples_with_gaze,
        'total_batches': len(train_loader),
        'total_samples': total,
        'lr': current_lr
    }
    
    return train_stats

# ----------------- Enhanced Evaluation Function -----------------
def evaluate_model_comprehensive(model, eval_loader, device, stats_tracker=None, dataset_name="eval"):
    """Enhanced evaluation with comprehensive statistics."""
    model.eval()
    all_labels = []
    all_preds = []
    all_files = []
    all_probs = []
        # ADD THESE LINES for loss tracking
    total_loss = 0.0
    total_cls_loss = 0.0
    total_gaze_loss = 0.0
    batches_with_gaze = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating {dataset_name}"):
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)
            
            # Get files
            batch_files = []
            if 'file' in batch:
                for f in batch['file']:
                    if isinstance(f, (bytes, bytearray)):
                        try:
                            f = f.decode('utf-8', errors='ignore')
                        except:
                            f = str(f)
                    batch_files.append(os.path.basename(str(f)))
            
            # Forward pass with attention (but we won't store attention maps)
            outputs = model(eeg, return_attention=True)
            if isinstance(outputs, tuple):
                logits, attention_map = outputs
            elif isinstance(outputs, dict):
                logits = outputs['logits']
                attention_map = outputs.get('attention_map', None)
            else:
                logits = outputs
                attention_map = None
                        # ADD THESE LINES: Calculate evaluation loss
            # Classification loss
            cls_loss = F.cross_entropy(logits, labels)
            
            # Gaze loss if available
            has_gaze = 'gaze' in batch and batch['gaze'] is not None
            if has_gaze and attention_map is not None:
                gaze = batch['gaze'].to(device)
                gaze_loss = compute_gaze_attention_loss(attention_map, gaze, labels, 'mse')  # or use your preferred loss type
                loss = cls_loss + gaze_loss
                batches_with_gaze += 1
                total_gaze_loss += gaze_loss.item()
            else:
                gaze_loss = torch.tensor(0.0).to(device)
                loss = cls_loss
            
            # Accumulate losses
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            
            # Get predictions
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Store results
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_files.extend(batch_files)
            all_probs.extend(probs.tolist())
    
    # Calculate metrics
    acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    # ADD THESE LINES: Calculate average losses
    avg_loss = total_loss / max(len(eval_loader), 1)
    avg_cls_loss = total_cls_loss / max(len(eval_loader), 1)
    avg_gaze_loss = total_gaze_loss / max(len(eval_loader), 1)
    eval_stats = {
        'acc': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall,
        'balanced_acc': balanced_acc,
        'total_samples': len(all_labels),
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'gaze_loss': avg_gaze_loss,
        'gaze_batches': batches_with_gaze
    }
    
    # Record statistics if tracker provided
    if stats_tracker:
        # Record confusion matrix
        stats_tracker.record_confusion_matrix(all_labels, all_preds, dataset_name)
        
        # Record predictions
        stats_tracker.record_predictions(all_files, all_labels, all_preds, all_probs, dataset_name)
    
    return eval_stats, all_labels, all_preds, all_files

# ----------------- Function to Collect All Attention Maps -----------------
def collect_eval_attention_maps(model, eval_loader, device, stats_tracker=None):
    """Collect attention maps for evaluation set after training is complete."""
    print(f"\nCollecting attention maps for evaluation dataset...")
    model.eval()
    
    all_attention_data = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Collecting eval attention maps"):
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)
            
            # Get files
            batch_files = []
            if 'file' in batch:
                for f in batch['file']:
                    if isinstance(f, (bytes, bytearray)):
                        try:
                            f = f.decode('utf-8', errors='ignore')
                        except:
                            f = str(f)
                    batch_files.append(os.path.basename(str(f)))
            
            # Forward pass WITH attention
            outputs = model(eeg, return_attention=True)
            if isinstance(outputs, tuple):
                logits, attention_maps = outputs
            elif isinstance(outputs, dict):
                logits = outputs['logits']
                attention_maps = outputs.get('attention_map', None)
            else:
                logits = outputs
                attention_maps = None
            
            # Get predictions
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Store attention data for each sample
            for i in range(eeg.shape[0]):
                sample_data = {
                    'file': batch_files[i] if i < len(batch_files) else f"unknown_{i}",
                    'eeg': eeg[i].cpu().numpy(),
                    'attention_map': attention_maps[i].cpu().numpy() if attention_maps is not None else None,
                    'label': labels[i].cpu().item(),
                    'prediction': preds[i].cpu().item(),
                    'probability': probs[i].cpu().numpy(),
                    'logits': logits[i].cpu().numpy(),
                    'dataset': "eval_final"
                }
                
                # Add gaze data if available
                if 'gaze' in batch and batch['gaze'] is not None:
                    sample_data['gaze_map'] = batch['gaze'][i].numpy()
                
                all_attention_data.append(sample_data)
    
    print(f"Collected {len(all_attention_data)} attention maps from evaluation dataset")
    
    # Store in stats tracker if provided
    if stats_tracker:
        stats_tracker.attention_maps['eval_final'] = all_attention_data
    
    return all_attention_data

# ----------------- Updated Main Function -----------------
def main(lr=1e-4, epochs=50, batch_size=32, accum_iter=1, 
         gaze_weight=0.3, gaze_loss_type='mse', dropout=0.5):
    """
    Parameterized main training function.
    
    Args:
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size
        accum_iter: Gradient accumulation steps
        gaze_weight: Weight for gaze loss (0 = no gaze supervision)
        gaze_loss_type: Type of gaze loss ('mse', 'weighted_mse', 'cosine', 'kl', 'combined')
    """
    DataDebugger.print_header("GAZE-GUIDED ATTENTION TRAINING WITH COMPREHENSIVE TRACKING", width=80)
    
    # Print parameter values
    print(f"Training Parameters:")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Accumulation steps: {accum_iter}")
    print(f"  Gaze weight: {gaze_weight}")
    print(f"  Gaze loss type: {gaze_loss_type}")
    print(f"  Dropout rate: {dropout}")  
    
    device = def_dev()
    print(f"\nDevice: {device}")
    
    # Initialize statistics tracker
    stats_tracker = TrainingStatistics(output_dir='training_statistics')
    print(f"Statistics will be saved to: {stats_tracker.run_dir}")
    
    mins = 5
    length = mins * 3000
    input_size = (22, length)
    
    # Use provided data paths or defaults
    data_dir = r"/kaggle/input/gazedata/results/data/data_processed/results0"
    gaze_json_dir = r"/kaggle/input/gazedata/results/gaze"
    
    # Update hyperparameters with passed values
    hyps = def_hyp(batch_size=batch_size, epochs=epochs, lr=lr, accum_iter=accum_iter)
    
    # Build dataloaders
    try:
        train_loader, eval_loader, gaze_stats = get_dataloaders_fixed(
            data_dir=data_dir,
            batch_size=hyps['batch_size'],
            seed=42,
            target_length=length,
            gaze_json_dir=gaze_json_dir,
            only_matched=True,
            suffixes_to_strip=DEFAULT_SUFFIXES_TO_STRIP,
            eeg_sampling_rate=50.0
        )
    except Exception as e:
        print("Error building dataloaders:", e)
        traceback.print_exc()
        return
    
    # Record initial class distributions
    print("\nRecording initial class distributions...")
    train_dist = stats_tracker.record_class_distribution(train_loader, "train")
    eval_dist = stats_tracker.record_class_distribution(eval_loader, "eval")
    
    print(f"  Train distribution: {dict(train_dist)}")
    print(f"  Eval distribution: {dict(eval_dist)}")
    
    # Model
    try:
        sample_batch = next(iter(train_loader))
        n_chan = sample_batch['eeg'].shape[1]
        print(f"\nDetected {n_chan} channels from data")
    except:
        n_chan = 22
        print(f"\nUsing default {n_chan} channels")
    
    model = NeuroGATE_Gaze_MultiRes(
        n_chan=n_chan,
        n_outputs=2,
        original_time_length=length,
        dropout_rate=dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.1)
    
    # Test forward pass
    try:
        sample_batch = next(iter(train_loader))
        test_eeg = sample_batch['eeg'].to(device)[:2]
        model.eval()
        with torch.no_grad():
            logits = model(test_eeg, return_attention=False)
            print("\nModel forward OK, logits shape:", logits.shape)
    except Exception as e:
        print("Model forward error:", e)
        traceback.print_exc()
        return
    
    # Training loop
    best_acc = 0.0
    class_counts = [803, 306]
    total = sum(class_counts)
    class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float32)
    
    print(f"\nStarting training for {hyps['epochs']} epochs...")
    print("=" * 80)
    
    for epoch in range(hyps['epochs']):
        DataDebugger.print_header(f"EPOCH {epoch+1}/{hyps['epochs']}", width=60, char='-')
        
        # Train with parameterized gaze_weight and gaze_loss_type
        train_stats = train_epoch_with_gaze(
            model, train_loader, optimizer, device, 
            gaze_weight=gaze_weight,  # Use parameter
            gaze_loss_type=gaze_loss_type,  # Use parameter
            class_weights=class_weights,
            stats_tracker=stats_tracker,
            epoch=epoch
        )
        
        # Evaluate

        eval_stats, ev_labels, ev_preds, ev_files = evaluate_model_comprehensive(
            model, eval_loader, device, stats_tracker, "eval"
        )
        
        # Record epoch statistics (pass eval_stats with losses)
        epoch_data = stats_tracker.record_epoch(epoch, train_stats, eval_stats, model)
        # Compute metrics for scheduler
        metric_for_sched = eval_stats['balanced_acc']
        scheduler.step(metric_for_sched)
        
        # Record epoch statistics
        epoch_data = stats_tracker.record_epoch(epoch, train_stats, eval_stats, model)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train: Loss={train_stats['loss']:.4f} (CLS={train_stats['cls_loss']:.4f}, "
              f"Gaze={train_stats['gaze_loss']:.4f}) | Acc={train_stats['acc']:.2f}%")
        print(f"  Eval:  Loss={eval_stats['loss']:.4f} (CLS={eval_stats['cls_loss']:.4f}, "
              f"Gaze={eval_stats['gaze_loss']:.4f}) | Acc={eval_stats['acc']:.2f}% | "
              f"Balanced Acc={eval_stats['balanced_acc']:.4f} | "
              f"Macro F1={eval_stats['macro_f1']:.4f}")
        print(f"  Gaze:  {train_stats['gaze_samples']}/{train_stats['total_samples']} samples (train), "
              f"{eval_stats['gaze_batches']}/{len(eval_loader)} batches (eval)")
        print(f"  LR:    {train_stats['lr']:.2e}")

        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(ev_labels, ev_preds, digits=4))
        
        # Save best model based on accuracy
        if eval_stats['acc'] > best_acc:
            best_acc = eval_stats['acc']
            torch.save({
                'epoch': epoch, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'accuracy': eval_stats['acc'],
                'balanced_acc': eval_stats['balanced_acc'],
                'macro_f1': eval_stats['macro_f1'],
                'params': {
                    'lr': lr,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'gaze_weight': gaze_weight,
                    'gaze_loss_type': gaze_loss_type,
                    'dropout': dropout  # ADD THIS LINE
                }
            }, 'best_model_gaze_attention_fixed.pth')
            print(f"  Saved best model at epoch {epoch+1} (acc {eval_stats['acc']:.2f}%)")
    
    # Load best model
    try:
        if os.path.exists('best_model_gaze_attention_fixed.pth'):
            checkpoint = torch.load('best_model_gaze_attention_fixed.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
            print(f"  Best accuracy: {checkpoint['accuracy']:.2f}%")
            print(f"  Training parameters: {checkpoint.get('params', {})}")
    except Exception as e:
        print("Could not load best model:", e)
    
    # Collect attention maps for evaluation set
    print("\n" + "=" * 80)
    print("COLLECTING ATTENTION MAPS FOR EVALUATION SET (AFTER TRAINING)")
    print("=" * 80)
    
    eval_attention_maps = collect_eval_attention_maps(
        model, eval_loader, device, stats_tracker
    )
    
    # Combine all attention maps
    all_attention_maps = {
        'eval_final': eval_attention_maps,
        'metadata': {
            'total_eval_samples': len(eval_attention_maps),
            'model_epochs': len(stats_tracker.epoch_stats),
            'training_params': {
                'lr': lr,
                'epochs': epochs,
                'batch_size': batch_size,
                'accum_iter': accum_iter,
                'gaze_weight': gaze_weight,
                'gaze_loss_type': gaze_loss_type
            },
            'collection_timestamp': datetime.now().isoformat()
        }
    }
    
    # Add to stats tracker
    stats_tracker.attention_maps = all_attention_maps
    
    # Save all results
    print("\n" + "=" * 80)
    print("SAVING ALL TRAINING RESULTS")
    print("=" * 80)
    
    stats_tracker.save_final_results(model=model, attention_maps=all_attention_maps)
    
    # Save a summary report with parameters
    with open(stats_tracker.run_dir / 'training_summary.txt', 'w') as f:
        f.write("TRAINING SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total epochs: {len(stats_tracker.epoch_stats)}\n")
        f.write(f"Best evaluation accuracy: {best_acc:.2f}%\n")
        f.write(f"\nTraining Parameters:\n")
        f.write(f"  Learning rate: {lr}\n")
        f.write(f"  Epochs: {epochs}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Accumulation steps: {accum_iter}\n")
        f.write(f"  Gaze weight: {gaze_weight}\n")
        f.write(f"  Gaze loss type: {gaze_loss_type}\n")
        f.write(f"  Dropout rate: {dropout}\n")
        f.write(f"\nDataset Statistics:\n")
        f.write(f"  Train samples: {len(train_loader.dataset)}\n")
        f.write(f"  Eval samples: {len(eval_loader.dataset)}\n")
        f.write(f"  Train class distribution: {dict(train_dist)}\n")
        f.write(f"  Eval class distribution: {dict(eval_dist)}\n")
        f.write(f"\nResults saved to: {stats_tracker.run_dir}\n")
    
    print("\nTraining complete!")
    print(f"Best evaluation accuracy: {best_acc:.2f}%")
    print(f"All results saved to: {stats_tracker.run_dir}")
    
    return best_acc, stats_tracker.run_dir

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Gaze Model")

    # Match your main() parameters + defaults
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--accum-iter", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gaze-weight", type=float, default=0.3, help="Gaze loss weight")
    parser.add_argument(
        "--gaze-loss-type",
        type=str,
        default="mse",
        choices=["mse", "combined", "cosine"],
        help="Type of gaze supervision loss"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (0.0 to 0.8)")


    args = parser.parse_args()

    # Call your main()
    main(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_iter=args.accum_iter,
        gaze_weight=args.gaze_weight,
        gaze_loss_type=args.gaze_loss_type,
        dropout=args.dropout, 
    )
