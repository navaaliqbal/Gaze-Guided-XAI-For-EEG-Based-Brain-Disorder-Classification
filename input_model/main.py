#!/usr/bin/env python3
"""
Refactored training script with clean data-saving architecture:
1. Separates training statistics from analysis artifacts
2. Uses memory-efficient storage formats (NPZ for attention maps)
3. Eliminates duplicate storage
4. Single authoritative model checkpoint
5. Clear separation of concerns

Storage Strategy:
- Training stats: CSV files for metrics, JSON for metadata
- Attention maps: Compressed NPZ files with one file per sample
- Model checkpoint: Single best_model.pth based on validation loss
"""

import os
import sys
import json
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
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler, Subset
import pandas as pd

# Project path
sys.path.append('.')
sys.path.append('..')
DATA_CONFIG = {
    'data_dir': r'/kaggle/input/gazedata/results/data/data_processed/results0',  # Your data directory
    'gaze_json_dir': r'/kaggle/input/gazefixations/gaze',  # Your gaze JSON directory
    'train_subdir': 'train',  # Subdirectory for training data
    'eval_subdir': 'eval'     # Subdirectory for evaluation data
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
                         suffixes_to_strip=None, val_split=0.1, **kwargs):
    """
    Build dataloaders using FilteredEEGGazeFixationDataset
    
    Returns 3 loaders: train_loader, val_loader, test_loader
    - train_loader: Training data (after splitting out validation)
    - val_loader: Validation data split from training data (for hyperparameter tuning, LR scheduling, checkpoint saving)
    - test_loader: Separate test data from eval directory (for FINAL evaluation only)
    
    Args:
        val_split: Fraction of training data to use for validation (default: 0.1 = 10%)
    """
    DataDebugger.print_header("BUILD DATALOADERS (FIXED) - Train/Val/Test Split")
    
    # Use configurable subdirectories
    train_dir = os.path.join(data_dir, kwargs.get('train_subdir', 'train'))
    test_dir = os.path.join(data_dir, kwargs.get('eval_subdir', 'eval'))  # This is the TEST set
    
    print(f"  Main data_dir: {data_dir}")
    print(f"  Train directory: {train_dir}")
    print(f"  Test directory: {test_dir} (for FINAL evaluation only)")
    print(f"  Gaze JSON directory: {gaze_json_dir}")
    print(f"  Validation split ratio: {val_split} (split from training data)")
    
    # instantiate the filtered wrapper for train and eval
    dataset_kwargs = {
        'indexes': indexes,
        'target_length': target_length,
        'eeg_sampling_rate': kwargs.get('eeg_sampling_rate', 50.0)
    }

    # Load full training dataset
    full_trainset = FilteredEEGGazeFixationDataset(
        data_dir=train_dir,
        gaze_json_dir=gaze_json_dir,
        dataset_cls=EEGGazeFixationDataset,
        dataset_kwargs=dataset_kwargs,
        suffixes_to_strip=suffixes_to_strip or DEFAULT_SUFFIXES_TO_STRIP
    )

    # Get labels for the full training set
    full_labels = [full_trainset[i]['label'] for i in range(len(full_trainset))]
    
    # ALWAYS split training data into train and validation sets
    print(f"\n  Splitting training data: {(1-val_split)*100:.0f}% train, {val_split*100:.0f}% validation")
    
    # Create indices for stratified split
    train_indices, val_indices = train_test_split(
        range(len(full_trainset)),
        test_size=val_split,
        random_state=seed,
        stratify=full_labels
    )
    
    print(f"  Train indices: {len(train_indices)}, Validation indices: {len(val_indices)}")
    
    # Create Subset datasets
    trainset = Subset(full_trainset, train_indices)
    valset = Subset(full_trainset, val_indices)
    
    # Get labels for train and val subsets
    labels = [full_labels[i] for i in train_indices]
    val_labels = [full_labels[i] for i in val_indices]
    
    print(f"  Train label distribution: {dict(Counter(labels))}")
    print(f"  Validation label distribution: {dict(Counter(val_labels))}")
    
    # Load separate TEST dataset (from eval directory)
    print(f"\n  Loading separate TEST set from: {test_dir}")
    testset = FilteredEEGGazeFixationDataset(
        data_dir=test_dir,
        gaze_json_dir=gaze_json_dir,
        dataset_cls=EEGGazeFixationDataset,
        dataset_kwargs=dataset_kwargs,
        suffixes_to_strip=suffixes_to_strip or DEFAULT_SUFFIXES_TO_STRIP
    )
    test_labels = [testset[i]['label'] for i in range(len(testset))]
    print(f"  Test label distribution: {dict(Counter(test_labels))}")

    class_counts = Counter(labels)
    num_classes = len(class_counts)
    total_samples = len(labels)
    print(f"\n  Number of classes: {num_classes}, Total training samples: {total_samples}")
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Assign a weight to each sample
    sample_weights = [class_weights[label] for label in labels]

    # Create sampler for training data only
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

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

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=min(batch_size, len(valset)) if len(valset) > 0 else 1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=min(batch_size, len(testset)) if len(testset) > 0 else 1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    print("\nDATALOADER SUMMARY")
    print(f"  Train samples: {len(trainset)} | batches: {len(train_loader)}")
    print(f"  Validation samples: {len(valset)} | batches: {len(val_loader)}")
    print(f"  Test samples: {len(testset)} | batches: {len(test_loader)}")
    print(f"\n  IMPORTANT: ")
    print(f"    - Use val_loader for: LR scheduling, checkpoint saving, early stopping")
    print(f"    - Use test_loader ONLY for: Final evaluation at the end")

    # Quick dataset diagnostics
    DataDebugger.analyze_dataset(trainset, "Filtered Train Dataset", max_samples=5)
    DataDebugger.analyze_dataset(valset, "Filtered Validation Dataset", max_samples=5)
    DataDebugger.analyze_dataset(testset, "Filtered Test Dataset", max_samples=5)

    return train_loader, val_loader, test_loader, { 
        'train_filtered': len(trainset),
        'val_filtered': len(valset),
        'test_filtered': len(testset),
        # Subset wraps the underlying dataset under the `.dataset` attribute
        'train_disk_matched': len(trainset.dataset.disk_matched_basenames) if hasattr(trainset, 'dataset') and hasattr(trainset.dataset, 'disk_matched_basenames') else 0,
        'val_disk_matched': len(valset.dataset.disk_matched_basenames) if hasattr(valset, 'dataset') and hasattr(valset.dataset, 'disk_matched_basenames') else 0,
        'test_disk_matched': len(testset.disk_matched_basenames) if hasattr(testset, 'disk_matched_basenames') else 0
    }

# ----------------- Storage Invariants and Contracts -----------------
"""
ATTENTION MAP STORAGE INVARIANTS:
1. Shape: (n_channels, n_timepoints) where n_channels=22, n_timepoints=15000
2. Dtype: float32 (4 bytes per value)
3. Range: [0, 1] after normalization (clipped if needed)
4. Format: Compressed NPZ (numpy.savez_compressed)
5. Naming: {file_identifier}_attention.npz
6. Content keys: 'attention_map', 'file_id', 'shape', 'sampling_rate'

TRAINING STATISTICS INVARIANTS:
1. Epoch stats: CSV with one row per epoch
2. Predictions: CSV with one row per sample
3. Confusion matrices: JSON-serializable lists
4. No duplicate storage of any data
5. No pickle for large arrays

MODEL CHECKPOINT INVARIANTS:
1. Single checkpoint: best_model.pth (lowest validation loss)
2. Contains: model_state_dict, optimizer_state_dict, epoch, metrics
3. Saved only when validation loss improves
"""

# ----------------- Attention Map Storage -----------------
class AttentionMapStorage:
    """
    Memory-efficient storage for attention maps using compressed NPZ format.
    Eliminates pickle and duplicate storage.
    """
    
    def __init__(self, output_dir='attention_maps'):
        """
        Args:
            output_dir: Directory to store attention maps (one file per sample)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Metadata tracking
        self.metadata = {
            'n_samples': 0,
            'expected_shape': (22, 15000),
            'dtype': 'float32',
            'sampling_rate': 50.0,
            'created': datetime.now().isoformat()
        }
        
        self.file_manifest = []
    
    def save_attention_map(self, attention_map, file_identifier, validate=True):
        """
        Save a single attention map to disk.
        
        Args:
            attention_map: numpy array of shape (n_channels, n_timepoints)
            file_identifier: unique identifier for this sample (filename)
            validate: whether to validate shape and range
        
        Returns:
            Path to saved file
        """
        # Validate if requested
        if validate:
            if not isinstance(attention_map, np.ndarray):
                attention_map = attention_map.cpu().numpy() if torch.is_tensor(attention_map) else np.array(attention_map)
            
            # Check shape
            expected_shape = self.metadata['expected_shape']
            if attention_map.shape != expected_shape:
                raise ValueError(f"Attention map shape {attention_map.shape} doesn't match expected {expected_shape}")
            
            # Check dtype
            if attention_map.dtype != np.float32:
                attention_map = attention_map.astype(np.float32)
            
            # Ensure range [0, 1]
            attention_map = np.clip(attention_map, 0.0, 1.0)
        
        # Create filename (sanitize file_identifier)
        safe_identifier = os.path.basename(file_identifier).replace('.npz', '').replace('.json', '')
        output_file = self.output_dir / f"{safe_identifier}_attention.npz"
        
        # Save with compression
        np.savez_compressed(
            output_file,
            attention_map=attention_map,
            file_id=file_identifier,
            shape=attention_map.shape,
            sampling_rate=self.metadata['sampling_rate']
        )
        
        # Update metadata
        self.metadata['n_samples'] += 1
        self.file_manifest.append({
            'file_id': file_identifier,
            'path': str(output_file),
            'shape': attention_map.shape,
            'saved_at': datetime.now().isoformat()
        })
        
        return output_file
    
    def save_batch(self, attention_maps, file_identifiers):
        """
        Save a batch of attention maps efficiently.
        
        Args:
            attention_maps: tensor or array of shape (batch_size, n_channels, n_timepoints)
            file_identifiers: list of file identifiers for each sample in batch
        
        Returns:
            List of saved file paths
        """
        if torch.is_tensor(attention_maps):
            attention_maps = attention_maps.cpu().numpy()
        
        saved_paths = []
        for i, file_id in enumerate(file_identifiers):
            path = self.save_attention_map(attention_maps[i], file_id, validate=True)
            saved_paths.append(path)
        
        return saved_paths
    
    def save_metadata(self):
        """Save metadata and manifest to JSON."""
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        manifest_file = self.output_dir / 'manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(self.file_manifest, f, indent=2)
        
        print(f"Saved metadata: {self.metadata['n_samples']} attention maps")
        return metadata_file, manifest_file
    
    @staticmethod
    def load_attention_map(file_path):
        """
        Load a single attention map from disk.
        
        Args:
            file_path: Path to the .npz file
        
        Returns:
            dict with keys: 'attention_map', 'file_id', 'shape', 'sampling_rate'
        """
        data = np.load(file_path, allow_pickle=False)
        return {
            'attention_map': data['attention_map'],
            'file_id': str(data['file_id']),
            'shape': tuple(data['shape']),
            'sampling_rate': float(data['sampling_rate'])
        }

# ----------------- Training Statistics Tracker -----------------
class TrainingStatistics:
    """
    Tracks training metrics and statistics.
    
    DOES NOT store:
    - Attention maps (use AttentionMapStorage instead)
    - Model weights per epoch (only best checkpoint is saved)
    - Redundant data
    
    DOES store:
    - Epoch-level metrics (loss, accuracy, F1, etc.)
    - Predictions and labels for analysis
    - Confusion matrices
    - Class distributions
    """
    
    def __init__(self, output_dir='training_stats'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize storage containers (NO attention_maps, NO model_weights per epoch)
        self.epoch_stats = []
        self.class_distributions = []
        self.confusion_matrices = []
        self.predictions = defaultdict(list)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True, parents=True)
        
    def record_epoch(self, epoch, train_stats, val_stats, test_stats=None):
        """
        Record epoch-level statistics for train/val/test splits.
        
        Args:
            epoch: Current epoch number
            train_stats: Dictionary of training metrics
            val_stats: Dictionary of validation metrics
            test_stats: Optional dictionary of test metrics (only for final evaluation)
        
        Returns:
            Recorded epoch data dictionary
        """
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_stats.get('loss', 0),
            'train_cls_loss': train_stats.get('cls_loss', 0),
            'train_gaze_loss': train_stats.get('gaze_loss', 0),
            'train_acc': train_stats.get('acc', 0),
            'val_acc': val_stats.get('acc', 0),
            'val_loss': val_stats.get('loss', 0),
            'val_cls_loss': val_stats.get('cls_loss', 0),
            'val_gaze_loss': val_stats.get('gaze_loss', 0),
            'val_macro_f1': val_stats.get('macro_f1', 0),
            'val_balanced_acc': val_stats.get('balanced_acc', 0),
            'val_weighted_f1': val_stats.get('weighted_f1', 0),
            'val_precision': val_stats.get('precision', 0),
            'val_recall': val_stats.get('recall', 0),
            'lr': train_stats.get('lr', 0),
            'train_gaze_batches': train_stats.get('gaze_batches', 0),
            'train_gaze_samples': train_stats.get('gaze_samples', 0),
            'val_gaze_batches': val_stats.get('gaze_batches', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add test stats if provided (only for final evaluation)
        if test_stats:
            epoch_data['test_acc'] = test_stats.get('acc', 0)
            epoch_data['test_loss'] = test_stats.get('loss', 0)
            epoch_data['test_cls_loss'] = test_stats.get('cls_loss', 0)
            epoch_data['test_gaze_loss'] = test_stats.get('gaze_loss', 0)
            epoch_data['test_macro_f1'] = test_stats.get('macro_f1', 0)
            epoch_data['test_balanced_acc'] = test_stats.get('balanced_acc', 0)
            epoch_data['test_weighted_f1'] = test_stats.get('weighted_f1', 0)
            epoch_data['test_precision'] = test_stats.get('precision', 0)
            epoch_data['test_recall'] = test_stats.get('recall', 0)
            epoch_data['test_gaze_batches'] = test_stats.get('gaze_batches', 0)
        
        self.epoch_stats.append(epoch_data)
        
        # Save intermediate results (CSV only, no large objects)
        self._save_intermediate_results()
        
        return epoch_data
    

    
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
    

    
    def _save_intermediate_results(self):
        """Save intermediate results to disk (CSV format only)."""
        # Save epoch stats
        if self.epoch_stats:
            pd.DataFrame(self.epoch_stats).to_csv(self.run_dir / 'epoch_stats.csv', index=False)
    
    def save_final_results(self):
        """
        Save all collected statistics to disk.
        
        Saves:
        - Epoch statistics (CSV)
        - Training plots (PNG)
        - Class distributions (CSV)
        - Confusion matrices (JSON)
        - Predictions (CSV)
        - Metadata (JSON)
        
        Does NOT save:
        - Attention maps (use AttentionMapStorage)
        - Model checkpoints (saved separately as best_model.pth)
        """
        print(f"\nSaving training statistics to: {self.run_dir}")
        
        # 1. Save epoch statistics
        if self.epoch_stats:
            epoch_df = pd.DataFrame(self.epoch_stats)
            epoch_df.to_csv(self.run_dir / 'epoch_statistics.csv', index=False)
            print(f"  ✓ Saved epoch statistics ({len(self.epoch_stats)} epochs)")
        
        # 2. Save training history plots
        self._create_training_plots()
        print(f"  ✓ Saved training plots")
        
        # 3. Save class distributions (JSON, not pickle)
        if self.class_distributions:
            with open(self.run_dir / 'class_distributions.json', 'w') as f:
                json.dump(self.class_distributions, f, indent=2)
            print(f"  ✓ Saved class distributions")
        
        # 4. Save confusion matrices (JSON, not pickle)
        if self.confusion_matrices:
            # Convert numpy arrays to lists for JSON serialization
            cm_serializable = []
            for cm in self.confusion_matrices:
                cm_copy = cm.copy()
                if 'matrix' in cm_copy and isinstance(cm_copy['matrix'], np.ndarray):
                    cm_copy['matrix'] = cm_copy['matrix'].tolist()
                cm_serializable.append(cm_copy)
            
            with open(self.run_dir / 'confusion_matrices.json', 'w') as f:
                json.dump(cm_serializable, f, indent=2)
            print(f"  ✓ Saved confusion matrices")
        
        # 5. Save predictions
        for dataset_name, pred_list in self.predictions.items():
            if pred_list:
                pred_df = pd.DataFrame(pred_list)
                pred_df.to_csv(self.run_dir / f'predictions_{dataset_name}.csv', index=False)
                print(f"  ✓ Saved predictions for {dataset_name} ({len(pred_list)} samples)")
        
        # 6. Save configuration and metadata
        metadata = {
            'timestamp': self.timestamp,
            'total_epochs': len(self.epoch_stats),
            'run_directory': str(self.run_dir),
            'storage_format': 'CSV and JSON (no pickle)',
            'note': 'Attention maps stored separately in attention_maps/ directory'
        }
        with open(self.run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Saved metadata")
        
        print(f"\nTraining statistics saved successfully!")
        print(f"  Location: {self.run_dir}")
        print(f"  Files: epoch_statistics.csv, predictions_*.csv, confusion_matrices.json")
        print(f"  Plots: training_curves.png, confusion_matrix.png")
    
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
        ax.plot(epochs, [s['val_acc'] for s in self.epoch_stats], label='Val Accuracy', linewidth=2)
        # Plot test accuracy if available
        if 'test_acc' in self.epoch_stats[-1]:
            test_accs = [s.get('test_acc', None) for s in self.epoch_stats]
            if any(ta is not None for ta in test_accs):
                ax.plot(epochs, test_accs, label='Test Accuracy', linewidth=2, linestyle=':')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Train vs Validation vs Test Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 scores
        ax = axes[1, 0]
        ax.plot(epochs, [s['val_macro_f1'] for s in self.epoch_stats], label='Val Macro F1', linewidth=2)
        ax.plot(epochs, [s['val_weighted_f1'] for s in self.epoch_stats], label='Val Weighted F1', linestyle='--')
        ax.plot(epochs, [s['val_balanced_acc'] for s in self.epoch_stats], label='Val Balanced Acc', linestyle=':')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Validation Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision-Recall
        ax = axes[1, 1]
        ax.plot(epochs, [s['val_precision'] for s in self.epoch_stats], label='Val Precision', linewidth=2)
        ax.plot(epochs, [s['val_recall'] for s in self.epoch_stats], label='Val Recall', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Validation Precision and Recall')
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
                         gaze_loss_type='cosine', class_weights=None, stats_tracker=None, epoch=0,gaze_loss_scale=1.0):
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
            gaze_loss_raw = compute_gaze_attention_loss(attention_map, gaze, labels, gaze_loss_type)
            gaze_loss_scaled = gaze_loss_scale * gaze_loss_raw
            loss = cls_loss + gaze_weight * gaze_loss_scaled
        else:
            gaze_loss_raw = torch.tensor(0.0).to(device)
            gaze_loss_scaled = torch.tensor(0.0).to(device)
            loss = cls_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_gaze += gaze_loss_scaled.item() if has_gaze else 0.0

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # No batch-level storage during training (only epoch-level metrics)
        
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
        'lr': current_lr,
        'gaze_loss_scale': gaze_loss_scale
    }
    
    return train_stats

# ----------------- Enhanced Evaluation Function -----------------
def evaluate_model_comprehensive(model, eval_loader, device,gaze_weight=0.3, gaze_loss_type='mse',gaze_loss_scale=1.0,stats_tracker=None, dataset_name="eval"):
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
                gaze_loss_raw = compute_gaze_attention_loss(attention_map, gaze, labels, gaze_loss_type)
                gaze_loss_scaled = gaze_loss_raw * gaze_loss_scale  # APPLY SAME SCALING
                gaze_loss_weighted = gaze_weight * gaze_loss_scaled  # Then apply gaze_weight
                loss = cls_loss + gaze_loss_weighted
                batches_with_gaze += 1
                total_gaze_loss += gaze_loss_weighted.item()
            else:
                gaze_loss_raw = torch.tensor(0.0).to(device)
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
        'gaze_batches': batches_with_gaze,
        'gaze_loss_scale': gaze_loss_scale 
    }
    
    # Record statistics if tracker provided
    if stats_tracker:
        # Record confusion matrix
        stats_tracker.record_confusion_matrix(all_labels, all_preds, dataset_name)
        
        # Record predictions
        stats_tracker.record_predictions(all_files, all_labels, all_preds, all_probs, dataset_name)
    
    return eval_stats, all_labels, all_preds, all_files

# ----------------- Function to Collect and Save Attention Maps -----------------
def collect_and_save_attention_maps(model, eval_loader, device, output_dir='attention_maps'):
    """
    Collect attention maps from eval set and save them efficiently.
    
    Memory-efficient: Processes one batch at a time and saves to disk immediately.
    Does NOT accumulate all attention maps in memory.
    
    Args:
        model: Trained model
        eval_loader: DataLoader for evaluation set
        device: torch device
        output_dir: Directory to save attention maps
    
    Returns:
        AttentionMapStorage instance with metadata
    """
    model.eval()
    storage = AttentionMapStorage(output_dir=output_dir)
    
    print(f"\nCollecting attention maps from evaluation set...")
    print(f"Output directory: {storage.output_dir}")
    
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Saving attention maps")):
            eeg = batch['eeg'].to(device)
            
            # Get file identifiers
            batch_files = []
            if 'file' in batch:
                for f in batch['file']:
                    if isinstance(f, (bytes, bytearray)):
                        f = f.decode('utf-8', errors='ignore')
                    batch_files.append(os.path.basename(str(f)))
            else:
                # Fallback: use batch index
                batch_files = [f"sample_{batch_idx}_{i}" for i in range(eeg.size(0))]
            
            # Forward pass WITH attention
            outputs = model(eeg, return_attention=True)
            
            # Handle model's return format
            if isinstance(outputs, dict):
                attention_map = outputs['attention_map']
            elif isinstance(outputs, tuple):
                _, attention_map = outputs
            else:
                raise RuntimeError("Model must return attention map when return_attention=True")
            
            if attention_map is None:
                raise RuntimeError("Attention map was not returned by the model")
            
            # Save batch immediately (memory-efficient)
            storage.save_batch(attention_map, batch_files)
            total_samples += len(batch_files)
    
    # Save metadata and manifest
    storage.save_metadata()
    
    print(f"\n✓ Successfully saved {total_samples} attention maps")
    print(f"  Format: Compressed NPZ (numpy.savez_compressed)")
    print(f"  Shape per map: {storage.metadata['expected_shape']}")
    print(f"  Total files: {len(storage.file_manifest)}")
    print(f"  Location: {storage.output_dir}")
    
    return storage
def compute_gaze_loss_scale(model, train_loader, device, gaze_loss_type='mse'):
    """
    Compute a principled scaling factor for gaze_loss so that its gradient contribution
    is comparable to classification loss.
    
    Uses a single training batch, computes the scale once, and returns it fixed for the entire training.
    
    Args:
        model: EEG model with return_attention=True capability
        train_loader: DataLoader for training data
        device: torch.device
        gaze_loss_type: Type of gaze loss ('mse', 'cosine', etc.)
        
    Returns:
        gaze_loss_scale: Scaling factor to apply to gaze_loss
        metrics: Dictionary with diagnostic information
    """
    print("\n" + "=" * 80)
    print("COMPUTING GAZE LOSS SCALING FACTOR")
    print("=" * 80)
    
    # Get a batch with gaze data
    for batch in train_loader:
        if 'gaze' in batch and batch['gaze'] is not None:
            break
    else:
        print("  WARNING: No gaze data found in training loader!")
        return 1.0, {'error': 'No gaze data'}
    
    model.eval()  # Just for computation, not training
    
    eeg = batch['eeg'].to(device)
    labels = batch['label'].to(device)
    gaze = batch['gaze'].to(device)
    
    # Get batch files for context
    batch_files = []
    if 'file' in batch:
        for f in batch['file']:
            if isinstance(f, (bytes, bytearray)):
                f = f.decode('utf-8', errors='ignore')
            batch_files.append(os.path.basename(str(f)))
    
    with torch.no_grad():
        # Forward pass with attention
        outputs = model(eeg, return_attention=True)
        if isinstance(outputs, tuple):
            logits, attention_map = outputs
        elif isinstance(outputs, dict):
            logits = outputs['logits']
            attention_map = outputs['attention_map']
        else:
            logits = outputs
            attention_map = None
        
        if attention_map is None:
            print("  WARNING: Model does not return attention map!")
            return 1.0, {'error': 'No attention map'}
        
        # Compute raw losses
        cls_loss = F.cross_entropy(logits, labels).item()
        gaze_loss = compute_gaze_attention_loss(attention_map, gaze, labels, gaze_loss_type).item()
        
        # Compute scaling factor: cls_loss / gaze_loss
        if gaze_loss > 1e-8:
            gaze_loss_scale = cls_loss / gaze_loss
        else:
            gaze_loss_scale = 1.0
        
        # Clip to reasonable range
        gaze_loss_scale = np.clip(gaze_loss_scale, 0.1, 100.0)
        
        # Diagnostic information
        metrics = {
            'cls_loss_raw': cls_loss,
            'gaze_loss_raw': gaze_loss,
            'gaze_loss_scale': gaze_loss_scale,
            'batch_size': eeg.shape[0],
            'has_gaze': True,
            'samples_with_gaze': eeg.shape[0],
            'loss_ratio': cls_loss / gaze_loss if gaze_loss > 0 else float('inf'),
            'file_count': len(batch_files)
        }
        
        print(f"\n  Loss Scaling Analysis:")
        print(f"    Classification loss: {cls_loss:.6f}")
        print(f"    Gaze loss (raw): {gaze_loss:.6f}")
        print(f"    Loss ratio (cls/gaze): {cls_loss/gaze_loss:.2f}:1")
        print(f"    Recommended gaze_loss_scale: {gaze_loss_scale:.2f}")
        print(f"    With gaze_weight=1.0: Effective scale = {gaze_loss_scale:.2f}")
        print(f"    Processed batch: {len(batch_files)} samples")
        
        return gaze_loss_scale, metrics

# ----------------- Updated Main Function -----------------
def main(lr=1e-4, epochs=50, batch_size=32, accum_iter=1, 
         gaze_weight=0.3, gaze_loss_type='mse', val_split=0.1):
    """
    Parameterized main training function.
    
    Returns 3 loaders: train_loader, val_loader, test_loader
    - train_loader: Training data (for learning)
    - val_loader: Validation data (for hyperparameter tuning, LR scheduling, checkpoint saving)
    - test_loader: Test data (for FINAL evaluation only, no training decisions)
    
    Args:
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size
        accum_iter: Gradient accumulation steps
        gaze_weight: Weight for gaze loss (0 = no gaze supervision)
        gaze_loss_type: Type of gaze loss ('mse', 'weighted_mse', 'cosine', 'kl', 'combined')
        val_split: Fraction of training data to use for validation (default: 0.1 = 10%)
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
    print(f"  Validation split: {val_split * 100:.0f}% (from training data)")
    print(f"\n  Data Split Strategy:")
    print(f"    1. Split training directory: {(1-val_split)*100:.0f}% train, {val_split*100:.0f}% validation")
    print(f"    2. Separate test directory: Used ONLY for final evaluation")
    print(f"    3. Validation set: Used for LR scheduling, checkpoint saving, early stopping")
    print(f"    4. Test set: NOT used for any training decisions")
    
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
    gaze_json_dir = r"/kaggle/input/gazefixations/gaze"
    
    # Update hyperparameters with passed values
    hyps = def_hyp(batch_size=batch_size, epochs=epochs, lr=lr, accum_iter=accum_iter)
    
    # Build dataloaders
    try:
        train_loader, val_loader, test_loader, gaze_stats = get_dataloaders_fixed(
            data_dir=data_dir,
            batch_size=hyps['batch_size'],
            seed=42,
            target_length=length,
            gaze_json_dir=gaze_json_dir,
            only_matched=True,
            suffixes_to_strip=DEFAULT_SUFFIXES_TO_STRIP,
            eeg_sampling_rate=50.0,
            val_split=val_split
        )
    except Exception as e:
        print("Error building dataloaders:", e)
        traceback.print_exc()
        return
    
    # Record initial class distributions
    print("\nRecording initial class distributions...")
    train_dist = stats_tracker.record_class_distribution(train_loader, "train")
    val_dist = stats_tracker.record_class_distribution(val_loader, "val")
    test_dist = stats_tracker.record_class_distribution(test_loader, "test")
    
    print(f"  Train distribution: {dict(train_dist)}")
    print(f"  Validation distribution: {dict(val_dist)}")
    print(f"  Test distribution: {dict(test_dist)}")
    
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
        original_time_length=length
    ).to(device)
    gaze_loss_scale, scale_metrics = compute_gaze_loss_scale(
        model, train_loader, device, gaze_loss_type
    )
    
    print(f"\n" + "=" * 80)
    print(f"FIXED SCALING FACTOR FOR ENTIRE TRAINING: {gaze_loss_scale:.2f}")
    print(f"Effective gaze loss = gaze_weight × gaze_loss_scale × gaze_loss_raw")
    print(f"With gaze_weight={gaze_weight:.2f}: Effective scale = {gaze_weight * gaze_loss_scale:.2f}")
    print("=" * 80)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)
    
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
    best_loss = float('inf')
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
            epoch=epoch,
            gaze_loss_scale=gaze_loss_scale
        )
        
        # Evaluate on VALIDATION set (for training decisions)
        val_stats, val_labels, val_preds, val_files = evaluate_model_comprehensive(
            model, val_loader, device, gaze_weight=gaze_weight, gaze_loss_type=gaze_loss_type,
            gaze_loss_scale=gaze_loss_scale, stats_tracker=stats_tracker, dataset_name="val"
        )
        
        # Record epoch statistics (NO model weights storage)
        # epoch_data = stats_tracker.record_epoch(epoch, train_stats, val_stats)
        
        # Update learning rate scheduler based on VALIDATION loss (NOT test loss!)
        metric_for_sched = val_stats['loss']
        scheduler.step(metric_for_sched)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train: Loss={train_stats['loss']:.4f} (CLS={train_stats['cls_loss']:.4f}, "
              f"Gaze={train_stats['gaze_loss']:.4f}) | Acc={train_stats['acc']:.2f}%")
        print(f"  Val:   Loss={val_stats['loss']:.4f} (CLS={val_stats['cls_loss']:.4f}, "
              f"Gaze={val_stats['gaze_loss']:.4f}) | Acc={val_stats['acc']:.2f}% | "
              f"Balanced Acc={val_stats['balanced_acc']:.4f} | "
              f"Macro F1={val_stats['macro_f1']:.4f}")
        print(f"  Gaze:  {train_stats['gaze_samples']}/{train_stats['total_samples']} samples (train), "
              f"{val_stats['gaze_batches']}/{len(val_loader)} batches (val)")
        print(f"  LR:    {train_stats['lr']:.2e}")

        
        # Detailed classification report for validation set
        print("\nValidation Classification Report:")
        print(classification_report(val_labels, val_preds, digits=4))
        
        # Track best metrics based on VALIDATION set
        if val_stats['acc'] > best_acc:
            best_acc = val_stats['acc']
        
        # Save SINGLE authoritative checkpoint (based on VALIDATION loss, NOT test loss!)
        if val_stats['loss'] < best_loss:
            best_loss = val_stats['loss']
            checkpoint_path = stats_tracker.run_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_stats['loss'],
                'val_loss': val_stats['loss'],
                'val_acc': val_stats['acc'],
                'val_f1': val_stats['macro_f1'],
            }, checkpoint_path)
            print(f"  ✓ Saved best model checkpoint (val_loss: {best_loss:.4f}, epoch: {epoch+1})")
    
    # Load best model checkpoint
    checkpoint_path = stats_tracker.run_dir / 'best_model.pth'
    try:
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\n✓ Loaded best model checkpoint from epoch {checkpoint['epoch']+1}")
            print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
            print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")
    except Exception as e:
        print(f"Warning: Could not load best model checkpoint: {e}")
    
    # FINAL EVALUATION ON TEST SET (no training decisions based on this!)
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    print("NOTE: This is the FIRST and ONLY time we evaluate on the test set!")
    print("      The test set was NOT used for any training decisions.")
    
    test_stats, test_labels, test_preds, test_files = evaluate_model_comprehensive(
        model, test_loader, device, stats_tracker=stats_tracker, dataset_name="test"
    )
    
    print(f"\nFinal Test Results:")
    print(f"  Test Loss: {test_stats['loss']:.4f} (CLS={test_stats['cls_loss']:.4f}, "
          f"Gaze={test_stats['gaze_loss']:.4f})")
    print(f"  Test Accuracy: {test_stats['acc']:.2f}%")
    print(f"  Test Balanced Accuracy: {test_stats['balanced_acc']:.4f}")
    print(f"  Test Macro F1: {test_stats['macro_f1']:.4f}")
    print(f"  Test Weighted F1: {test_stats['weighted_f1']:.4f}")
    print(f"  Test Precision: {test_stats['precision']:.4f}")
    print(f"  Test Recall: {test_stats['recall']:.4f}")
    
    print("\nFinal Test Classification Report:")
    print(classification_report(test_labels, test_preds, digits=4))
    
    # Record final test statistics in the last epoch
    if stats_tracker.epoch_stats:
        last_epoch = stats_tracker.epoch_stats[-1]
        last_epoch['test_acc'] = test_stats['acc']
        last_epoch['test_loss'] = test_stats['loss']
        last_epoch['test_cls_loss'] = test_stats['cls_loss']
        last_epoch['test_gaze_loss'] = test_stats['gaze_loss']
        last_epoch['test_macro_f1'] = test_stats['macro_f1']
        last_epoch['test_balanced_acc'] = test_stats['balanced_acc']
        last_epoch['test_weighted_f1'] = test_stats['weighted_f1']
        last_epoch['test_precision'] = test_stats['precision']
        last_epoch['test_recall'] = test_stats['recall']
        last_epoch['test_gaze_batches'] = test_stats['gaze_batches']
    
    # Save training statistics
    print("\n" + "=" * 80)
    print("SAVING TRAINING STATISTICS")
    print("=" * 80)
    stats_tracker.save_final_results()
    
    # Collect and save attention maps SEPARATELY (memory-efficient) from TEST set
    print("\n" + "=" * 80)
    print("COLLECTING AND SAVING ATTENTION MAPS FROM TEST SET")
    print("=" * 80)
    
    attention_storage = collect_and_save_attention_maps(
        model=model,
        eval_loader=test_loader,  # Use test_loader for attention map collection
        device=device,
        output_dir=stats_tracker.run_dir / 'attention_maps'
    )
    
    # Create comprehensive summary report
    summary_path = stats_tracker.run_dir / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total epochs: {len(stats_tracker.epoch_stats)}\n")
        f.write(f"Best validation accuracy: {best_acc:.2f}%\n")
        f.write(f"Best validation loss: {best_loss:.4f}\n")
        f.write(f"Final test accuracy: {test_stats['acc']:.2f}%\n")
        f.write(f"Final test loss: {test_stats['loss']:.4f}\n\n")
        
        f.write("TRAINING PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Learning rate: {lr}\n")
        f.write(f"  Epochs: {epochs}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Accumulation steps: {accum_iter}\n")
        f.write(f"  Gaze weight: {gaze_weight}\n")
        f.write(f"  Gaze loss type: {gaze_loss_type}\n\n")
        
        f.write("DATASET STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Train samples: {len(train_loader.dataset)}\n")
        f.write(f"  Validation samples: {len(val_loader.dataset)}\n")
        f.write(f"  Test samples: {len(test_loader.dataset)}\n")
        f.write(f"  Train class distribution: {dict(train_dist)}\n")
        f.write(f"  Validation class distribution: {dict(val_dist)}\n")
        f.write(f"  Test class distribution: {dict(test_dist)}\n\n")
        f.write("DATA USAGE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  ✓ Train set: Used for model training\n")
        f.write(f"  ✓ Validation set: Used for LR scheduling, checkpoint saving, early stopping\n")
        f.write(f"  ✓ Test set: Used ONLY for final evaluation (no training decisions)\n\n")
        
        f.write("STORAGE ARCHITECTURE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Training stats: {stats_tracker.run_dir}\n")
        f.write(f"    - epoch_statistics.csv (epoch-level metrics for train/val/test)\n")
        f.write(f"    - predictions_train.csv (per-sample predictions for training)\n")
        f.write(f"    - predictions_val.csv (per-sample predictions for validation)\n")
        f.write(f"    - predictions_test.csv (per-sample predictions for test)\n")
        f.write(f"    - confusion_matrices.json (confusion matrices for train/val/test)\n")
        f.write(f"    - training_curves.png (loss/accuracy plots)\n")
        f.write(f"  Model checkpoint: {checkpoint_path}\n")
        f.write(f"    - Selected based on VALIDATION loss (not test loss!)\n")
        f.write(f"  Attention maps: {attention_storage.output_dir}\n")
        f.write(f"    - Format: Compressed NPZ (one file per sample)\n")
        f.write(f"    - Count: {attention_storage.metadata['n_samples']} maps (from test set)\n")
        f.write(f"    - Shape per map: {attention_storage.metadata['expected_shape']}\n\n")
        
        f.write("DESIGN PRINCIPLES\n")
        f.write("-" * 40 + "\n")
        f.write("  ✓ No duplicate storage (attention maps saved once)\n")
        f.write("  ✓ No pickle for large arrays (NPZ with compression)\n")
        f.write("  ✓ Separated training stats from analysis artifacts\n")
        f.write("  ✓ Single authoritative checkpoint (best validation loss)\n")
        f.write("  ✓ Memory-efficient processing (batch-by-batch)\n")
        f.write("  ✓ All artifacts are independently reloadable\n\n")
        
        f.write("INVARIANTS\n")
        f.write("-" * 40 + "\n")
        f.write("  Attention map shape: (22, 15000) [channels × timepoints]\n")
        f.write("  Attention map dtype: float32\n")
        f.write("  Attention map range: [0, 1] (normalized)\n")
        f.write("  EEG sampling rate: 50 Hz\n")
        f.write("  Temporal resolution: 15000 samples = 300 seconds\n")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Final test accuracy: {test_stats['acc']:.2f}%")
    print(f"Final test loss: {test_stats['loss']:.4f}")
    print(f"\nResults saved to: {stats_tracker.run_dir}")
    print(f"  - Training statistics: epoch_statistics.csv")
    print(f"  - Predictions: predictions_train.csv, predictions_val.csv, predictions_test.csv")
    print(f"  - Model checkpoint: best_model.pth (selected based on validation loss)")
    print(f"  - Attention maps: attention_maps/ ({attention_storage.metadata['n_samples']} files from test set)")
    print(f"  - Summary: training_summary.txt")
    print("\n" + "=" * 80)
    print("DATA SPLIT VERIFICATION")
    print("=" * 80)
    print("✓ Training data: Used for learning")
    print("✓ Validation data: Used for LR scheduling, checkpoint saving (NO TEST DATA LEAKAGE!)")
    print("✓ Test data: Used ONLY for final evaluation (NO TRAINING DECISIONS!)")
    print("=" * 80)
    
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
    parser.add_argument("--val-split", type=float, default=0.1, 
                       help="Fraction of training data to use for validation (default: 0.1 = 10%%)")

    args = parser.parse_args()

    # Call your main()
    main(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_iter=args.accum_iter,
        gaze_weight=args.gaze_weight,
        gaze_loss_type=args.gaze_loss_type,
        val_split=args.val_split,
    )