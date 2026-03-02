#!/usr/bin/env python3
"""
MNIST Linear Probe Experiment for Rotation Prediction

This script:
1. Trains a model on MNIST with invariance or equivariance loss
2. Computes residuals: r(g·x) = E(g·x) - mean_h(E(h·x))
3. Trains a linear probe to predict the rotation g from residuals
4. Compares how well rotations can be predicted from invariant vs equivariant features

The hypothesis:
- Equivariant features should encode g → high probe accuracy
- Invariant features should discard g → low probe accuracy (chance = 25%)

Usage:
    python train_linear_probe.py --mode both --num_runs 3
    python train_linear_probe.py --mode equivariance --epochs 30 --probe_epochs 20
"""

import os
import argparse
import pickle
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import random


# =============================================================================
# Configuration
# =============================================================================
DEFAULT_CONFIG = {
    'batch_size': 128,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 30,
    'lambda_equiv': 0.1,
    'val_split': 0.1,
    'hidden_dims': [512, 256, 128, 64],
    'dropout_rate': 0.3,
    'latent_dim': 64,
    'probe_learning_rate': 1e-2,
    'probe_epochs': 20,
    'probe_batch_size': 256,
    'probe_every': 5,  # Evaluate probe every N epochs during training
    'quick_probe_epochs': 10,  # Fewer epochs for checkpoint probes
}

ROTATION_NAMES = ['0°', '90°', '180°', '270°']


# =============================================================================
# Diagnostic Analysis Functions
# =============================================================================
@torch.no_grad()
def compute_linear_probe_closed_form(X: torch.Tensor, y: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor, 
                                      reg_lambda: float = 1e-4) -> Tuple[float, torch.Tensor]:
    """
    Compute optimal linear probe using closed-form solution and return test accuracy.
    
    Solves: W* = argmin_W ||XW - Y||^2 + λ||W||^2
    Solution: W* = (X^T X + λI)^{-1} X^T Y
    
    Args:
        X: Training features (N, D)
        y: Training labels (N,) - integer class labels
        X_test: Test features (M, D)
        y_test: Test labels (M,)
        reg_lambda: Ridge regularization strength
    
    Returns:
        test_accuracy: Accuracy on test set
        confusion_matrix: (num_classes, num_classes) confusion matrix
    """
    num_classes = int(y.max().item()) + 1
    
    # Convert labels to one-hot
    Y = F.one_hot(y.long(), num_classes).float()
    
    # Closed-form solution: W = (X^T X + λI)^{-1} X^T Y
    XtX = X.T @ X
    XtY = X.T @ Y
    identity = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(XtX + reg_lambda * identity, XtY)
    
    # Predict on test set
    logits = X_test @ W
    preds = logits.argmax(dim=1)
    
    # Compute accuracy
    correct = (preds == y_test).sum().item()
    accuracy = 100.0 * correct / len(y_test)
    
    # Compute confusion matrix
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for pred, true in zip(preds, y_test):
        confusion[true.long(), pred.long()] += 1
    
    return accuracy, confusion


@torch.no_grad()
def analyze_residual_separability(model: nn.Module, train_dataset: Dataset, test_dataset: Dataset,
                                   device: torch.device, latent_dim: int, verbose: bool = True) -> Dict:
    """
    Comprehensive diagnostic analysis of residual structure and linear separability.
    
    Tests multiple linear probe strategies:
    1. Residuals only: r(g·x) = E(g·x) - mean_h(E(h·x))  [current approach]
    2. Raw features: E(g·x) directly
    3. Concatenated: [E(x), E(g·x)]
    4. Residual + base: [E(x), r(g·x)]
    5. Normalized residuals: r(g·x) / ||r(g·x)||
    
    Also computes:
    - Variance decomposition (between-class vs within-class)
    - Per-rotation statistics
    - Confusion matrices
    
    Args:
        model: Trained encoder model
        train_dataset: MNIST training dataset
        test_dataset: MNIST test dataset  
        device: Compute device
        latent_dim: Dimension of latent space
        verbose: Whether to print detailed results
    
    Returns:
        Dictionary containing all diagnostic results
    """
    model.eval()
    
    if verbose:
        print("\n" + "="*70)
        print("DIAGNOSTIC ANALYSIS: Residual Separability")
        print("="*70)
    
    # -------------------------------------------------------------------------
    # Step 1: Compute features for all rotations of all test samples
    # -------------------------------------------------------------------------
    if verbose:
        print("\nStep 1: Computing features for all rotations...")
    
    # Limit to subset for efficiency (can increase if needed)
    max_samples = 2000
    
    # Collect test samples
    test_images = []
    test_digit_labels = []
    for i in range(min(len(test_dataset), max_samples)):
        img, label = test_dataset[i]
        test_images.append(img)
        test_digit_labels.append(label)
    test_images = torch.stack(test_images).to(device)
    test_digit_labels = torch.tensor(test_digit_labels, device=device)
    
    # Collect train samples (for fitting probe)
    train_images = []
    train_digit_labels = []
    train_indices = random.sample(range(len(train_dataset)), min(len(train_dataset), max_samples * 5))
    for i in train_indices:
        img, label = train_dataset[i]
        train_images.append(img)
        train_digit_labels.append(label)
    train_images = torch.stack(train_images).to(device)
    train_digit_labels = torch.tensor(train_digit_labels, device=device)
    
    # Compute features for all rotations
    def compute_all_rotation_features(images):
        """Returns features for each rotation: dict[rot_idx] = features"""
        features_by_rot = {}
        for g in range(4):
            rotated = torch.rot90(images, k=g, dims=[-2, -1])
            _, feats = model(rotated)
            features_by_rot[g] = feats
        return features_by_rot
    
    # Process in batches
    batch_size = 256
    
    def batch_compute_features(images):
        all_features_by_rot = {g: [] for g in range(4)}
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_feats = compute_all_rotation_features(batch)
            for g in range(4):
                all_features_by_rot[g].append(batch_feats[g])
        return {g: torch.cat(all_features_by_rot[g], dim=0) for g in range(4)}
    
    train_features = batch_compute_features(train_images)
    test_features = batch_compute_features(test_images)
    
    if verbose:
        print(f"  Train samples: {len(train_images)}, Test samples: {len(test_images)}")
        print(f"  Feature dimension: {latent_dim}")
    
    # -------------------------------------------------------------------------
    # Step 2: Compute orbit means and residuals
    # -------------------------------------------------------------------------
    if verbose:
        print("\nStep 2: Computing orbit means and residuals...")
    
    # Orbit mean: (1/4) * sum_h E(h·x) for each sample
    train_orbit_mean = sum(train_features[g] for g in range(4)) / 4.0
    test_orbit_mean = sum(test_features[g] for g in range(4)) / 4.0
    
    # Residuals: r(g·x) = E(g·x) - orbit_mean
    train_residuals = {g: train_features[g] - train_orbit_mean for g in range(4)}
    test_residuals = {g: test_features[g] - test_orbit_mean for g in range(4)}
    
    # -------------------------------------------------------------------------
    # Step 3: Prepare different input representations for probes
    # -------------------------------------------------------------------------
    if verbose:
        print("\nStep 3: Preparing input representations...")
    
    # Each representation will have shape (N*4, D) where N is num samples
    # and labels are rotation indices 0,1,2,3
    
    def flatten_rotations(data_dict):
        """Flatten dict of per-rotation data into single tensor with labels"""
        features = []
        labels = []
        for g in range(4):
            features.append(data_dict[g])
            labels.append(torch.full((len(data_dict[g]),), g, dtype=torch.long, device=device))
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)
    
    # 1. Residuals only (current approach)
    train_residuals_flat, train_rot_labels = flatten_rotations(train_residuals)
    test_residuals_flat, test_rot_labels = flatten_rotations(test_residuals)
    
    # 2. Raw features E(g·x)
    train_raw_flat, _ = flatten_rotations(train_features)
    test_raw_flat, _ = flatten_rotations(test_features)
    
    # 3. Concatenated [E(x), E(g·x)] - E(x) is the g=0 feature
    def concat_with_base(features_dict, base_features):
        """Concatenate base features with each rotation's features"""
        result = {}
        for g in range(4):
            result[g] = torch.cat([base_features, features_dict[g]], dim=1)
        return result
    
    train_concat = concat_with_base(train_features, train_features[0])
    test_concat = concat_with_base(test_features, test_features[0])
    train_concat_flat, _ = flatten_rotations(train_concat)
    test_concat_flat, _ = flatten_rotations(test_concat)
    
    # 4. Residual + base [E(x), r(g·x)]
    train_res_base = concat_with_base(train_residuals, train_features[0])
    test_res_base = concat_with_base(test_residuals, test_features[0])
    train_res_base_flat, _ = flatten_rotations(train_res_base)
    test_res_base_flat, _ = flatten_rotations(test_res_base)
    
    # 5. Normalized residuals r(g·x) / ||r(g·x)||
    def normalize_features(features_dict):
        result = {}
        for g in range(4):
            norms = features_dict[g].norm(dim=1, keepdim=True).clamp(min=1e-8)
            result[g] = features_dict[g] / norms
        return result
    
    train_norm_residuals = normalize_features(train_residuals)
    test_norm_residuals = normalize_features(test_residuals)
    train_norm_flat, _ = flatten_rotations(train_norm_residuals)
    test_norm_flat, _ = flatten_rotations(test_norm_residuals)
    
    # -------------------------------------------------------------------------
    # Step 4: Train and evaluate linear probes on each representation
    # -------------------------------------------------------------------------
    if verbose:
        print("\nStep 4: Training linear probes on different representations...")
    
    results = {}
    
    representations = [
        ('residuals', train_residuals_flat, test_residuals_flat, 'r(g·x) = E(g·x) - mean'),
        ('raw_features', train_raw_flat, test_raw_flat, 'E(g·x) directly'),
        ('concat_base_transformed', train_concat_flat, test_concat_flat, '[E(x), E(g·x)]'),
        ('residual_plus_base', train_res_base_flat, test_res_base_flat, '[E(x), r(g·x)]'),
        ('normalized_residuals', train_norm_flat, test_norm_flat, 'r(g·x) / ||r||'),
    ]
    
    for name, train_X, test_X, description in representations:
        acc, confusion = compute_linear_probe_closed_form(
            train_X, train_rot_labels, test_X, test_rot_labels
        )
        results[name] = {
            'accuracy': acc,
            'confusion_matrix': confusion.cpu().numpy(),
            'description': description,
        }
        if verbose:
            print(f"\n  {description}:")
            print(f"    Accuracy: {acc:.2f}%")
    
    # -------------------------------------------------------------------------
    # Step 5: Variance decomposition analysis
    # -------------------------------------------------------------------------
    if verbose:
        print("\n\nStep 5: Variance decomposition analysis...")
    
    # For residuals: decompose variance into between-class (rotation) and within-class
    # Total variance = Between-class variance + Within-class variance
    
    # Global mean
    global_mean = test_residuals_flat.mean(dim=0)
    
    # Between-class variance: variance of class means around global mean
    class_means = torch.stack([test_residuals[g].mean(dim=0) for g in range(4)])
    between_var = ((class_means - global_mean) ** 2).sum(dim=1).mean().item()
    
    # Within-class variance: average variance within each class
    within_vars = []
    for g in range(4):
        class_mean = test_residuals[g].mean(dim=0)
        within_var_g = ((test_residuals[g] - class_mean) ** 2).sum(dim=1).mean().item()
        within_vars.append(within_var_g)
    within_var = np.mean(within_vars)
    
    # Total variance
    total_var = ((test_residuals_flat - global_mean) ** 2).sum(dim=1).mean().item()
    
    # Ratio: how much of variance is "explained" by rotation
    variance_ratio = between_var / (total_var + 1e-8)
    
    results['variance_analysis'] = {
        'total_variance': total_var,
        'between_class_variance': between_var,
        'within_class_variance': within_var,
        'variance_ratio_rotation': variance_ratio,
    }
    
    if verbose:
        print(f"\n  Residual Variance Decomposition:")
        print(f"    Total variance: {total_var:.4f}")
        print(f"    Between-class (rotation) variance: {between_var:.4f}")
        print(f"    Within-class (sample) variance: {within_var:.4f}")
        print(f"    Variance ratio (rotation explains): {variance_ratio*100:.2f}%")
    
    # -------------------------------------------------------------------------
    # Step 6: Per-rotation statistics
    # -------------------------------------------------------------------------
    if verbose:
        print("\n\nStep 6: Per-rotation residual statistics...")
    
    per_rot_stats = {}
    for g in range(4):
        residuals_g = test_residuals[g]
        per_rot_stats[f'rot_{g}'] = {
            'mean_norm': residuals_g.norm(dim=1).mean().item(),
            'std_norm': residuals_g.norm(dim=1).std().item(),
            'mean': residuals_g.mean(dim=0).cpu().numpy(),
        }
        if verbose:
            print(f"    Rotation {g} (={g*90}°): mean_norm={per_rot_stats[f'rot_{g}']['mean_norm']:.4f}, "
                  f"std_norm={per_rot_stats[f'rot_{g}']['std_norm']:.4f}")
    
    results['per_rotation_stats'] = per_rot_stats
    
    # -------------------------------------------------------------------------
    # Step 7: Fisher's Linear Discriminant Analysis (for interpretability)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n\nStep 7: Fisher's discriminant ratio (class separability)...")
    
    # Fisher's criterion: ratio of between-class scatter to within-class scatter
    # Higher = better linear separability
    
    # Between-class scatter matrix
    S_b = torch.zeros(latent_dim, latent_dim, device=device)
    for g in range(4):
        mean_diff = class_means[g] - global_mean
        S_b += len(test_residuals[g]) * torch.outer(mean_diff, mean_diff)
    S_b /= len(test_residuals_flat)
    
    # Within-class scatter matrix  
    S_w = torch.zeros(latent_dim, latent_dim, device=device)
    for g in range(4):
        class_mean = test_residuals[g].mean(dim=0)
        centered = test_residuals[g] - class_mean
        S_w += centered.T @ centered
    S_w /= len(test_residuals_flat)
    
    # Fisher ratio: trace(S_b) / trace(S_w)
    fisher_ratio = torch.trace(S_b).item() / (torch.trace(S_w).item() + 1e-8)
    
    results['fisher_ratio'] = fisher_ratio
    
    if verbose:
        print(f"    Fisher's discriminant ratio: {fisher_ratio:.4f}")
        print(f"    (Higher = better linear separability of rotations)")
    
    # -------------------------------------------------------------------------
    # Step 8: Direction Consistency Analysis
    # -------------------------------------------------------------------------
    # For equivariant features: r(g·x) = M(g)·E(x), so the DIRECTION of residuals
    # for each g should be more consistent (determined by M(g)) even though magnitude varies.
    # For invariant features: residuals are "error", so directions should be more random.
    
    if verbose:
        print("\n\nStep 8: Direction consistency analysis...")
    
    # Compute mean direction for each rotation class
    mean_directions = {}
    for g in range(4):
        mean_dir = test_residuals[g].mean(dim=0)
        mean_dir = mean_dir / (mean_dir.norm() + 1e-8)  # Normalize
        mean_directions[g] = mean_dir
    
    # Measure how well individual samples align with the class mean direction
    # Cosine similarity between each sample's residual and the class mean direction
    direction_consistency = {}
    for g in range(4):
        # Normalize each residual
        residuals_g = test_residuals[g]
        norms = residuals_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = residuals_g / norms
        
        # Cosine similarity with mean direction
        cos_sims = (normalized @ mean_directions[g]).cpu().numpy()
        direction_consistency[g] = {
            'mean_cos_sim': float(np.mean(cos_sims)),
            'std_cos_sim': float(np.std(cos_sims)),
            'fraction_aligned': float(np.mean(cos_sims > 0)),  # Fraction in same half-space
        }
    
    # Overall direction consistency: average across rotations
    avg_cos_sim = np.mean([direction_consistency[g]['mean_cos_sim'] for g in range(4)])
    avg_alignment = np.mean([direction_consistency[g]['fraction_aligned'] for g in range(4)])
    
    results['direction_consistency'] = {
        'per_rotation': direction_consistency,
        'average_cos_similarity': avg_cos_sim,
        'average_alignment_fraction': avg_alignment,
    }
    
    if verbose:
        print(f"\n  Direction Consistency (higher = more structured residuals):")
        print(f"    Average cosine similarity with class mean: {avg_cos_sim:.4f}")
        print(f"    Average fraction aligned (cos > 0): {avg_alignment*100:.1f}%")
        print(f"    Per-rotation breakdown:")
        for g in range(4):
            dc = direction_consistency[g]
            print(f"      Rotation {g*90}°: cos_sim={dc['mean_cos_sim']:.4f}±{dc['std_cos_sim']:.4f}, "
                  f"aligned={dc['fraction_aligned']*100:.1f}%")
    
    # -------------------------------------------------------------------------
    # Step 9: Cross-class direction similarity (should be LOW for good separation)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n\nStep 9: Cross-class direction similarity...")
    
    # Compute cosine similarity between mean directions of different rotation classes
    cross_class_sims = torch.zeros(4, 4)
    for g1 in range(4):
        for g2 in range(4):
            cross_class_sims[g1, g2] = (mean_directions[g1] @ mean_directions[g2]).item()
    
    # Off-diagonal average (excluding self-similarity)
    off_diag_mask = ~torch.eye(4, dtype=torch.bool)
    avg_cross_sim = cross_class_sims[off_diag_mask].mean().item()
    
    results['cross_class_similarity'] = {
        'matrix': cross_class_sims.cpu().numpy(),
        'average_off_diagonal': avg_cross_sim,
    }
    
    if verbose:
        print(f"\n  Mean direction similarity matrix:")
        print("              0°     90°    180°   270°")
        for g1 in range(4):
            row = [f"{cross_class_sims[g1, g2].item():6.3f}" for g2 in range(4)]
            print(f"    {g1*90:3}°  {' '.join(row)}")
        print(f"\n  Average off-diagonal similarity: {avg_cross_sim:.4f}")
        print(f"  (Lower = more distinct rotation directions = better separation)")

    # -------------------------------------------------------------------------
    # Step 10: Per-Sample Separability Analysis (KEY DIAGNOSTIC)
    # -------------------------------------------------------------------------
    # If r(g·x) = M(g)·E(x), then for a FIXED sample x, the 4 residuals should
    # be perfectly linearly separable (they're just 4 fixed vectors in 64-dim space).
    # The limitation comes from needing ONE linear probe for ALL samples.
    
    if verbose:
        print("\n\nStep 10: Per-sample separability analysis...")
        print("  Testing if rotations are separable when we know the sample identity...")
    
    # For each test sample, check if its 4 rotations are linearly separable
    n_test_samples = min(500, len(test_images))
    
    # Method 1: Check if 4 residual vectors per sample are linearly independent
    # (they span a subspace, and if distinct, a hyperplane can separate them)
    
    # Method 2: For each sample, compute pairwise distances between its 4 residuals
    per_sample_separability = []
    per_sample_min_distances = []
    
    for i in range(n_test_samples):
        # Get the 4 residuals for this sample
        sample_residuals = torch.stack([test_residuals[g][i] for g in range(4)])  # (4, latent_dim)
        
        # Compute pairwise L2 distances
        dists = torch.cdist(sample_residuals.unsqueeze(0), sample_residuals.unsqueeze(0)).squeeze(0)
        
        # Minimum distance between different rotations (off-diagonal)
        off_diag = dists[~torch.eye(4, dtype=torch.bool, device=device)]
        min_dist = off_diag.min().item()
        mean_dist = off_diag.mean().item()
        
        per_sample_min_distances.append(min_dist)
        
        # Check linear separability: can we find a hyperplane separating each rotation?
        # Simple heuristic: compute centroid, check if 4 points are "spread out"
        centroid = sample_residuals.mean(dim=0)
        deviations = sample_residuals - centroid
        
        # Compute the rank of the deviation matrix (should be 3 for 4 distinct points)
        try:
            rank = torch.linalg.matrix_rank(deviations).item()
        except:
            rank = 0
        per_sample_separability.append(rank)
    
    avg_min_dist = np.mean(per_sample_min_distances)
    avg_rank = np.mean(per_sample_separability)
    fraction_full_rank = np.mean([r >= 3 for r in per_sample_separability])
    
    # Method 3: Oracle accuracy - what if we train a separate probe for each sample?
    # Too expensive, but we can approximate: train probe on single-sample data
    
    # Method 4: Sample-conditioned probe
    # Train probe on [E(x), r(g·x)] - this gives the probe access to sample identity
    # We already have this! It's 'residual_plus_base' - but let's check if giving
    # the model the EXACT sample identity (one-hot) helps more
    
    # Create oracle features: [sample_index_onehot, r(g·x)]
    # This explicitly tells the probe which sample it is
    n_train = len(train_images)
    n_test = len(test_images)
    
    # One-hot encode sample indices (use first n_test samples for simplicity)
    # This is expensive for large datasets, so we'll use a smaller test
    oracle_n = min(200, n_test)
    
    # For oracle test: use sample identity as input
    oracle_correct = 0
    oracle_total = 0
    
    for i in range(oracle_n):
        sample_residuals = torch.stack([test_residuals[g][i] for g in range(4)])  # (4, D)
        labels = torch.arange(4, device=device)
        
        # Simple linear classifier for just these 4 points
        # Use least squares: W = (X^T X)^{-1} X^T Y
        X = sample_residuals
        Y = F.one_hot(labels, 4).float()
        
        try:
            W = torch.linalg.lstsq(X, Y).solution
            preds = (X @ W).argmax(dim=1)
            oracle_correct += (preds == labels).sum().item()
            oracle_total += 4
        except:
            oracle_total += 4  # Count as failure
    
    oracle_accuracy = 100.0 * oracle_correct / oracle_total if oracle_total > 0 else 0.0
    
    results['per_sample_analysis'] = {
        'avg_min_pairwise_distance': avg_min_dist,
        'avg_rank': avg_rank,
        'fraction_full_rank': fraction_full_rank,
        'oracle_accuracy': oracle_accuracy,
    }
    
    if verbose:
        print(f"\n  Per-sample residual statistics:")
        print(f"    Average minimum pairwise distance: {avg_min_dist:.4f}")
        print(f"    Average rank of residual deviations: {avg_rank:.2f} (max=3)")
        print(f"    Fraction with full rank (>=3): {fraction_full_rank*100:.1f}%")
        print(f"\n  Oracle accuracy (separate classifier per sample): {oracle_accuracy:.1f}%")
        print(f"    → If ~100%: rotations ARE separable per-sample, problem is cross-sample")
        print(f"    → If <<100%: rotations not separable even within single sample")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "="*70)
        print("DIAGNOSTIC SUMMARY")
        print("="*70)
        print("\nLinear Probe Accuracies:")
        for name in ['residuals', 'raw_features', 'concat_base_transformed', 'residual_plus_base', 'normalized_residuals']:
            acc = results[name]['accuracy']
            desc = results[name]['description']
            print(f"  {desc}: {acc:.2f}%")
        print(f"\nVariance explained by rotation: {variance_ratio*100:.2f}%")
        print(f"Fisher's discriminant ratio: {fisher_ratio:.4f}")
        print(f"\n*** KEY FINDING: Per-Sample Separability ***")
        print(f"  Oracle accuracy (per-sample classifier): {oracle_accuracy:.1f}%")
        print(f"  Shared probe accuracy (residuals): {results['residuals']['accuracy']:.1f}%")
        print(f"  Gap: {oracle_accuracy - results['residuals']['accuracy']:.1f}%")
        if oracle_accuracy > 95:
            print(f"  → CONCLUSION: Rotations ARE separable per-sample!")
            print(f"    The {100 - results['residuals']['accuracy']:.0f}% error is due to cross-sample interference.")
            print(f"    A single linear probe cannot handle the sample-dependent E(x) factor.")
        print(f"\nDirection Consistency:")
        print(f"  Average cosine similarity with class mean: {avg_cos_sim:.4f}")
        print(f"  Average fraction aligned: {avg_alignment*100:.1f}%")
        print(f"  Cross-class direction similarity: {avg_cross_sim:.4f}")
        
        # Print confusion matrix for residuals
        print("\nConfusion Matrix (residuals probe):")
        print("              Predicted")
        print("              0°    90°   180°  270°")
        conf = results['residuals']['confusion_matrix']
        for i, row in enumerate(conf):
            row_total = row.sum()
            row_pct = [f"{100*v/row_total:5.1f}" if row_total > 0 else "  N/A" for v in row]
            print(f"  True {i*90:3}°  {' '.join(row_pct)}")
    
    return results


@torch.no_grad()
def analyze_global_probe_approaches(model: nn.Module, test_dataset: Dataset,
                                     device: torch.device, latent_dim: int,
                                     c4_representations: List[torch.Tensor],
                                     verbose: bool = True) -> Dict:
    """
    Test approaches to eliminate E(x) dependence for a global linear probe.
    
    The problem: r(g·x) = M(g)·E(x) mixes rotation g and sample identity E(x) multiplicatively.
    Goal: Find a representation that isolates g so a single linear probe works for all samples.
    
    Approaches tested:
    1. Normalize by ||E(x)|| - scale-invariant residuals
    2. Unit direction - normalize each residual to unit length  
    3. Pairwise cosines - use cosine similarities between rotations (4 features)
    4. Projection scores - measure alignment with theoretical M(g) subspaces
    5. Whitening - normalize by per-orbit covariance
    6. Theoretical matching - directly use M(g) matrices
    7. Ratio features - element-wise r(g·x)/r(0·x)
    """
    model.eval()
    
    if verbose:
        print("\n" + "="*70)
        print("GLOBAL PROBE ANALYSIS: Eliminating E(x) Dependence")
        print("="*70)
    
    # -------------------------------------------------------------------------
    # Step 1: Compute features and residuals
    # -------------------------------------------------------------------------
    max_samples = 2000
    batch_size = 256
    
    test_images = []
    for i in range(min(len(test_dataset), max_samples)):
        img, _ = test_dataset[i]
        test_images.append(img)
    test_images = torch.stack(test_images).to(device)
    n_samples = len(test_images)
    
    if verbose:
        print(f"\nUsing {n_samples} test samples...")
    
    # Compute features for all rotations
    features = {g: [] for g in range(4)}
    for i in range(0, n_samples, batch_size):
        batch = test_images[i:i+batch_size]
        for g in range(4):
            rotated = torch.rot90(batch, k=g, dims=[-2, -1])
            _, feats = model(rotated)
            features[g].append(feats)
    features = {g: torch.cat(features[g], dim=0) for g in range(4)}
    
    # Compute orbit mean and residuals
    orbit_mean = sum(features[g] for g in range(4)) / 4.0
    residuals = {g: features[g] - orbit_mean for g in range(4)}
    
    # -------------------------------------------------------------------------
    # Step 2: Compute theoretical M(g) matrices
    # -------------------------------------------------------------------------
    # P_inv = (1/4) * sum_h ρ(h) = projection onto invariant subspace
    P_inv = sum(c4_representations) / 4.0
    
    # M(g) = ρ(g) - P_inv
    M_matrices = {g: c4_representations[g] - P_inv for g in range(4)}
    
    if verbose:
        print(f"\nTheoretical M(g) matrices computed (rank of M(0): {torch.linalg.matrix_rank(M_matrices[0]).item()})")
    
    # -------------------------------------------------------------------------
    # Helper: Split data into train/test
    # -------------------------------------------------------------------------
    n_train = int(0.8 * n_samples)
    
    def get_train_test(X_dict):
        """Split flattened features into train/test."""
        X_flat = torch.cat([X_dict[g] for g in range(4)], dim=0)
        y = torch.cat([torch.full((n_samples,), g, device=device, dtype=torch.long) for g in range(4)])
        
        # Split: first n_train samples of each rotation for train
        X_train = torch.cat([X_dict[g][:n_train] for g in range(4)])
        X_test = torch.cat([X_dict[g][n_train:] for g in range(4)])
        y_train = torch.cat([torch.full((n_train,), g, device=device, dtype=torch.long) for g in range(4)])
        y_test = torch.cat([torch.full((n_samples - n_train,), g, device=device, dtype=torch.long) for g in range(4)])
        
        return X_train, y_train, X_test, y_test
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Baseline: Raw residuals
    # -------------------------------------------------------------------------
    X_train, y_train, X_test, y_test = get_train_test(residuals)
    acc_baseline, _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    results['baseline_residuals'] = acc_baseline
    
    if verbose:
        print(f"\n{'Approach':<40} {'Accuracy':>10}")
        print("-"*55)
        print(f"{'Baseline (raw residuals)':<40} {acc_baseline:>9.2f}%")
    
    # -------------------------------------------------------------------------
    # Approach 1: Normalize by ||E(x)||
    # -------------------------------------------------------------------------
    E_norms = features[0].norm(dim=1, keepdim=True).clamp(min=1e-8)
    norm_by_Ex = {g: residuals[g] / E_norms for g in range(4)}
    
    X_train, y_train, X_test, y_test = get_train_test(norm_by_Ex)
    acc_norm_Ex, _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    results['normalize_by_Ex'] = acc_norm_Ex
    
    if verbose:
        print(f"{'Normalize by ||E(x)||':<40} {acc_norm_Ex:>9.2f}%")
    
    # -------------------------------------------------------------------------
    # Approach 2: Unit direction (normalize each residual)
    # -------------------------------------------------------------------------
    unit_residuals = {}
    for g in range(4):
        norms = residuals[g].norm(dim=1, keepdim=True).clamp(min=1e-8)
        unit_residuals[g] = residuals[g] / norms
    
    X_train, y_train, X_test, y_test = get_train_test(unit_residuals)
    acc_unit, _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    results['unit_direction'] = acc_unit
    
    if verbose:
        print(f"{'Unit direction r/||r||':<40} {acc_unit:>9.2f}%")
    
    # -------------------------------------------------------------------------
    # Approach 3: Pairwise cosines (excluding self-similarity to avoid leakage)
    # We use a FIXED reference rotation (g=0) and compute cosines with it
    # This gives 1 feature per sample: cos(r(g·x), r(0·x))
    # -------------------------------------------------------------------------
    def compute_cosine_with_base(res_dict, query_g):
        """Cosine of r(query_g·x) with r(0·x) - excludes self for g=0."""
        query = res_dict[query_g]
        query_norm = query / query.norm(dim=1, keepdim=True).clamp(min=1e-8)
        
        # Use all OTHER rotations as references (exclude self)
        cosines = []
        for h in range(4):
            if h == query_g:
                continue  # Skip self-cosine (would always be 1.0 = data leakage)
            ref = res_dict[h]
            ref_norm = ref / ref.norm(dim=1, keepdim=True).clamp(min=1e-8)
            cos_sim = (query_norm * ref_norm).sum(dim=1)
            cosines.append(cos_sim)
        
        return torch.stack(cosines, dim=1)  # 3 features per sample
    
    cosine_features = {g: compute_cosine_with_base(residuals, g) for g in range(4)}
    X_train, y_train, X_test, y_test = get_train_test(cosine_features)
    acc_cosine, _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test, reg_lambda=1.0)
    results['pairwise_cosines'] = acc_cosine
    
    if verbose:
        print(f"{'Pairwise cosines (3 feat, no self)':<40} {acc_cosine:>9.2f}%")
    
    # -------------------------------------------------------------------------
    # Approach 4: Projection onto M(g) column spaces
    # -------------------------------------------------------------------------
    projection_features = {g: [] for g in range(4)}
    
    # Precompute SVD of M matrices
    M_col_spaces = {}
    for h in range(4):
        U, S, _ = torch.linalg.svd(M_matrices[h], full_matrices=False)
        rank = (S > 1e-6).sum().item()
        M_col_spaces[h] = U[:, :max(1, rank)]
    
    for g in range(4):
        r_g = residuals[g]
        r_g_norm = r_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        r_g_unit = r_g / r_g_norm
        
        scores = []
        for h in range(4):
            U_col = M_col_spaces[h]
            proj = r_g_unit @ U_col
            proj_norm = proj.norm(dim=1)
            scores.append(proj_norm)
        
        projection_features[g] = torch.stack(scores, dim=1)
    
    X_train, y_train, X_test, y_test = get_train_test(projection_features)
    # Use higher regularization for low-dimensional features
    acc_proj, _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test, reg_lambda=1.0)
    results['projection_scores'] = acc_proj
    
    if verbose:
        print(f"{'Projection onto M(g) spaces (4 feat)':<40} {acc_proj:>9.2f}%")
    
    # -------------------------------------------------------------------------
    # Approach 5: Whitening per orbit
    # -------------------------------------------------------------------------
    whitened = {g: [] for g in range(4)}
    
    for i in range(n_samples):
        orbit_res = torch.stack([residuals[g][i] for g in range(4)])
        mean_r = orbit_res.mean(dim=0)
        centered = orbit_res - mean_r
        cov = centered.T @ centered / 4.0 + 0.01 * torch.eye(latent_dim, device=device)
        
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = eigenvalues.clamp(min=1e-6)
            whitening = eigenvectors @ torch.diag(1.0 / eigenvalues.sqrt()) @ eigenvectors.T
            
            for g in range(4):
                whitened[g].append(whitening @ residuals[g][i])
        except:
            for g in range(4):
                whitened[g].append(residuals[g][i])
    
    whitened = {g: torch.stack(whitened[g]) for g in range(4)}
    X_train, y_train, X_test, y_test = get_train_test(whitened)
    acc_white, _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    results['whitened'] = acc_white
    
    if verbose:
        print(f"{'Whitened (per-orbit cov)':<40} {acc_white:>9.2f}%")
    
    # -------------------------------------------------------------------------
    # Approach 6: Theoretical M(g) matching
    # -------------------------------------------------------------------------
    theoretical_preds = []
    theoretical_true = []
    
    for g in range(4):
        for i in range(n_samples):
            r = residuals[g][i]
            scores = [(M_matrices[h].T @ r).norm().item() for h in range(4)]
            theoretical_preds.append(int(np.argmax(scores)))
            theoretical_true.append(g)
    
    theoretical_preds = torch.tensor(theoretical_preds, device=device)
    theoretical_true = torch.tensor(theoretical_true, device=device)
    acc_theoretical = (theoretical_preds == theoretical_true).float().mean().item() * 100
    results['theoretical_matching'] = acc_theoretical
    
    if verbose:
        print(f"{'Theoretical M(g)^T matching':<40} {acc_theoretical:>9.2f}%")
    
    # -------------------------------------------------------------------------
    # Approach 7: Ratio features r(g·x) / r(0·x)
    # -------------------------------------------------------------------------
    eps = 1e-6
    r_0 = residuals[0]
    
    ratio_features = {}
    for g in range(4):
        ratio = residuals[g] / (r_0.abs() + eps) * r_0.sign()
        # Clip extreme values
        ratio = ratio.clamp(-100, 100)
        ratio_features[g] = ratio
    
    X_train, y_train, X_test, y_test = get_train_test(ratio_features)
    acc_ratio, _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    results['ratio_features'] = acc_ratio
    
    if verbose:
        print(f"{'Ratio features r(g)/r(0)':<40} {acc_ratio:>9.2f}%")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if verbose:
        best_approach = max(results, key=results.get)
        best_acc = results[best_approach]
        
        print("\n" + "-"*55)
        print(f"Best approach: {best_approach} ({best_acc:.2f}%)")
        
        if best_acc < 80:
            print("\n⚠ No approach achieves high accuracy!")
            print("  This confirms the fundamental limitation:")
            print("  r(g·x) = M(g)·E(x) entangles g and x multiplicatively.")
            print("  A single linear W cannot separate g when E(x) varies.")
        elif best_acc > 90:
            print(f"\n✓ {best_approach} successfully isolates rotation info!")
    
    return results


@torch.no_grad()
def analyze_residual_decomposition(model: nn.Module, test_dataset: Dataset,
                                    device: torch.device, latent_dim: int,
                                    c4_representations: List[torch.Tensor],
                                    verbose: bool = True) -> Dict:
    """
    Decompose residuals into structured part and error part:
    
    E(g·x) = ρ(g)·E(x) + ε(g,x)
    
    Therefore:
    r(g·x) = E(g·x) - mean_h(E(h·x))
           = [ρ(g)·E(x) + ε(g,x)] - mean_h[ρ(h)·E(x) + ε(h,x)]
           = [ρ(g) - mean_h(ρ(h))]·E(x) + [ε(g,x) - mean_h(ε(h,x))]
           = M(g)·E(x) + ε_residual(g,x)
    
    We test whether the probe is using M(g)·E(x) or ε_residual(g,x).
    """
    model.eval()
    
    if verbose:
        print("\n" + "="*70)
        print("RESIDUAL DECOMPOSITION ANALYSIS")
        print("="*70)
        print("\nDecomposing r(g·x) = M(g)·E(x) + ε_residual(g,x)")
        print("Testing which component the probe actually uses.\n")
    
    # Get test data
    max_samples = 2000
    batch_size = 256
    
    test_images = []
    for i in range(min(len(test_dataset), max_samples)):
        img, _ = test_dataset[i]
        test_images.append(img)
    test_images = torch.stack(test_images).to(device)
    n_samples = len(test_images)
    
    # Compute features for all rotations
    features = {g: [] for g in range(4)}
    for i in range(0, n_samples, batch_size):
        batch = test_images[i:i+batch_size]
        for g in range(4):
            rotated = torch.rot90(batch, k=g, dims=[-2, -1])
            _, feats = model(rotated)
            features[g].append(feats)
    features = {g: torch.cat(features[g], dim=0) for g in range(4)}
    
    # E(x) = E(0·x) = features at identity
    E_x = features[0]  # (n_samples, latent_dim)
    
    # Compute theoretical equivariant prediction: ρ(g)·E(x)
    rho_E_x = {}
    for g in range(4):
        rho_g = c4_representations[g].to(device)
        rho_E_x[g] = E_x @ rho_g.T  # (n_samples, latent_dim)
    
    # Compute error term: ε(g,x) = E(g·x) - ρ(g)·E(x)
    epsilon = {g: features[g] - rho_E_x[g] for g in range(4)}
    
    # Compute orbit means
    orbit_mean_actual = sum(features[g] for g in range(4)) / 4.0
    orbit_mean_epsilon = sum(epsilon[g] for g in range(4)) / 4.0
    
    # Compute actual residuals
    residuals_actual = {g: features[g] - orbit_mean_actual for g in range(4)}
    
    # Decompose residuals:
    # r(g·x) = [ρ(g)·E(x) - mean_h(ρ(h)·E(x))] + [ε(g,x) - mean_h(ε(h,x))]
    #        = M(g)·E(x) + ε_residual(g,x)
    
    # Structured part: M(g)·E(x) where M(g) = ρ(g) - P_inv
    P_inv = sum(c4_representations) / 4.0  # Projection onto invariant subspace
    M_matrices = {g: c4_representations[g] - P_inv for g in range(4)}
    
    structured_part = {g: E_x @ M_matrices[g].to(device).T for g in range(4)}
    
    # Error residual part: ε(g,x) - mean_h(ε(h,x))
    error_part = {g: epsilon[g] - orbit_mean_epsilon for g in range(4)}
    
    # Verify decomposition: r(g·x) ≈ structured_part + error_part
    if verbose:
        print("1. Verifying decomposition r(g·x) = M(g)·E(x) + ε_residual(g,x):")
        for g in range(4):
            reconstructed = structured_part[g] + error_part[g]
            reconstruction_error = F.mse_loss(residuals_actual[g], reconstructed).item()
            print(f"   Rotation {g*90}°: reconstruction MSE = {reconstruction_error:.2e}")
    
    # Compute norms of each component
    if verbose:
        print("\n2. Component magnitudes (averaged over samples):")
        print(f"   {'Rotation':<10} {'||r(g·x)||':<15} {'||M(g)·E(x)||':<15} {'||ε_res||':<15} {'ratio ε/M':<15}")
        print("   " + "-"*65)
        
        for g in range(4):
            norm_total = residuals_actual[g].norm(dim=1).mean().item()
            norm_structured = structured_part[g].norm(dim=1).mean().item()
            norm_error = error_part[g].norm(dim=1).mean().item()
            ratio = norm_error / (norm_structured + 1e-8)
            print(f"   {g*90}°{'':<7} {norm_total:<15.4f} {norm_structured:<15.4f} {norm_error:<15.4f} {ratio:<15.4f}")
    
    # Now test linear probe on each component separately
    if verbose:
        print("\n3. Linear probe accuracy on each component:")
    
    # Prepare train/test split
    n_train = int(0.8 * n_samples)
    
    def get_train_test_split(data_dict):
        X_train = torch.cat([data_dict[g][:n_train] for g in range(4)])
        X_test = torch.cat([data_dict[g][n_train:] for g in range(4)])
        y_train = torch.cat([torch.full((n_train,), g, device=device, dtype=torch.long) for g in range(4)])
        y_test = torch.cat([torch.full((n_samples - n_train,), g, device=device, dtype=torch.long) for g in range(4)])
        return X_train, y_train, X_test, y_test
    
    # Test on actual residuals
    X_train, y_train, X_test, y_test = get_train_test_split(residuals_actual)
    acc_actual, _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    
    # Test on structured part only (M(g)·E(x))
    X_train, y_train, X_test, y_test = get_train_test_split(structured_part)
    acc_structured, _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    
    # Test on error part only (ε_residual)
    X_train, y_train, X_test, y_test = get_train_test_split(error_part)
    acc_error, _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    
    if verbose:
        print(f"   Actual residuals r(g·x):        {acc_actual:.2f}%")
        print(f"   Structured part M(g)·E(x):      {acc_structured:.2f}%")
        print(f"   Error part ε_residual(g,x):     {acc_error:.2f}%")
        
        print("\n4. Interpretation:")
        if acc_error > acc_structured + 5:
            print(f"   ⚠ ERROR PART has HIGHER probe accuracy!")
            print(f"   → The probe is primarily using the non-equivariant error term")
            print(f"   → This explains why lower equiv loss can HURT probe accuracy")
            print(f"   → As equivariance improves, ε shrinks, and the probe loses signal")
        elif acc_structured > acc_error + 5:
            print(f"   ✓ STRUCTURED PART has higher probe accuracy")
            print(f"   → The probe is using the equivariant structure M(g)·E(x)")
        else:
            print(f"   ~ Both components contribute similarly")
            print(f"   → The probe uses both structured and error terms")
    
    # Compute correlation between components
    if verbose:
        print("\n5. Cosine similarity between actual residual and components:")
        for g in range(4):
            corr_structured = F.cosine_similarity(
                residuals_actual[g], structured_part[g], dim=1
            ).mean().item()
            corr_error = F.cosine_similarity(
                residuals_actual[g], error_part[g], dim=1
            ).mean().item()
            print(f"   {g*90}°: cos(r, M·E)={corr_structured:.4f}, cos(r, ε)={corr_error:.4f}")
    
    # Compute variance explained
    if verbose:
        print("\n6. Variance explained by each component:")
        total_var = sum(residuals_actual[g].var().item() for g in range(4)) / 4
        structured_var = sum(structured_part[g].var().item() for g in range(4)) / 4
        error_var = sum(error_part[g].var().item() for g in range(4)) / 4
        
        print(f"   Total residual variance:     {total_var:.4f}")
        print(f"   Structured part variance:    {structured_var:.4f} ({100*structured_var/total_var:.1f}%)")
        print(f"   Error part variance:         {error_var:.4f} ({100*error_var/total_var:.1f}%)")
    
    results = {
        'acc_actual_residuals': acc_actual,
        'acc_structured_part': acc_structured,
        'acc_error_part': acc_error,
        'norm_structured': {g: structured_part[g].norm(dim=1).mean().item() for g in range(4)},
        'norm_error': {g: error_part[g].norm(dim=1).mean().item() for g in range(4)},
    }
    
    return results


@torch.no_grad()
def compute_quick_global_probes(model: nn.Module, val_dataset: Dataset, device: torch.device,
                                 latent_dim: int, c4_representations: List[torch.Tensor],
                                 max_samples: int = 500, probe_layer: int = -1) -> Dict[str, float]:
    """
    Quick version of analyze_global_probe_approaches for checkpoint evaluation.
    Returns accuracy for each of the 7 approaches.
    
    Args:
        probe_layer: Which layer to probe. -1 = final layer, 0-3 = intermediate layers
                    (0=512, 1=256, 2=128, 3=64 for default architecture)
    """
    model.eval()
    
    # Determine feature dimension and C4 representations for the probe layer
    hidden_dims = [512, 256, 128, 64]  # Default architecture hidden dims
    if probe_layer == -1:
        feat_dim = latent_dim
        layer_c4_reps = c4_representations
    elif 0 <= probe_layer < len(hidden_dims):
        feat_dim = hidden_dims[probe_layer]
        # Create C4 representations for this dimension
        layer_c4_reps = get_c4_representation_for_latent_dim(feat_dim, device)
    else:
        feat_dim = latent_dim
        layer_c4_reps = c4_representations
    
    # Collect samples
    batch_size = 256
    val_images = []
    for i in range(min(len(val_dataset), max_samples)):
        img, _ = val_dataset[i]
        val_images.append(img)
    val_images = torch.stack(val_images).to(device)
    n_samples = len(val_images)
    
    # Compute features for all rotations
    features = {g: [] for g in range(4)}
    for i in range(0, n_samples, batch_size):
        batch = val_images[i:i+batch_size]
        for g in range(4):
            rotated = torch.rot90(batch, k=g, dims=[-2, -1])
            if probe_layer == -1:
                # Use final layer
                _, feats = model(rotated)
            else:
                # Use intermediate layer
                feats = model.get_features(rotated, layer_idx=probe_layer)
            features[g].append(feats)
    features = {g: torch.cat(features[g], dim=0) for g in range(4)}
    
    # Compute P_inv (projector onto invariant subspace) using the layer-appropriate representations
    P_inv = sum(layer_c4_reps) / 4.0
    P_equiv = torch.eye(P_inv.shape[0], device=device) - P_inv  # Projector onto equivariant subspaces
    
    # Key insight: For probing, we want to identify g from E(g·x).
    # If equivariant: E(g·x) = ρ(g) · E(x), so E_eq(g·x) = ρ(g) · E_eq(x)
    # The equivariant part E_eq(g·x) for different g differs by the action of ρ(g)
    
    # Project features onto equivariant subspace: E_eq(g·x) = P_equiv · E(g·x)
    features_eq = {g: features[g] @ P_equiv.T for g in range(4)}
    
    # Compute orbit mean and residuals (on full features)
    orbit_mean = sum(features[g] for g in range(4)) / 4.0
    residuals = {g: features[g] - orbit_mean for g in range(4)}
    
    # Compute M(g) matrices using layer-appropriate representations
    M_matrices = {g: layer_c4_reps[g] - P_inv for g in range(4)}
    
    # Helper: Split data
    n_train = int(0.8 * n_samples)
    
    def get_train_test(X_dict):
        X_train = torch.cat([X_dict[g][:n_train] for g in range(4)])
        X_test = torch.cat([X_dict[g][n_train:] for g in range(4)])
        y_train = torch.cat([torch.full((n_train,), g, device=device, dtype=torch.long) for g in range(4)])
        y_test = torch.cat([torch.full((n_samples - n_train,), g, device=device, dtype=torch.long) for g in range(4)])
        return X_train, y_train, X_test, y_test
    
    results = {}
    
    # 1. Baseline (full residuals)
    X_train, y_train, X_test, y_test = get_train_test(residuals)
    results['baseline_residuals'], _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    
    # 2. E_eq directly (equivariant subspace projection)
    X_train, y_train, X_test, y_test = get_train_test(features_eq)
    results['E_eq_direct'], _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    
    # 3. E_eq normalized by ||E_eq(x)|| (removes sample-dependent magnitude)
    E_eq_x_norms = features_eq[0].norm(dim=1, keepdim=True).clamp(min=1e-8)
    E_eq_normalized = {g: features_eq[g] / E_eq_x_norms for g in range(4)}
    X_train, y_train, X_test, y_test = get_train_test(E_eq_normalized)
    results['E_eq_normalized'], _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    
    # 4. E_eq unit direction
    E_eq_unit = {}
    for g in range(4):
        norms = features_eq[g].norm(dim=1, keepdim=True).clamp(min=1e-8)
        E_eq_unit[g] = features_eq[g] / norms
    X_train, y_train, X_test, y_test = get_train_test(E_eq_unit)
    results['E_eq_unit'], _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    
    # 5. Normalize by ||E(x)|| (original approach)
    E_norms = features[0].norm(dim=1, keepdim=True).clamp(min=1e-8)
    norm_by_Ex = {g: residuals[g] / E_norms for g in range(4)}
    X_train, y_train, X_test, y_test = get_train_test(norm_by_Ex)
    results['normalize_by_Ex'], _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    
    # 6. Unit direction (original)
    unit_residuals = {}
    for g in range(4):
        norms = residuals[g].norm(dim=1, keepdim=True).clamp(min=1e-8)
        unit_residuals[g] = residuals[g] / norms
    X_train, y_train, X_test, y_test = get_train_test(unit_residuals)
    results['unit_direction'], _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    
    # 7. Pairwise cosines on E_eq (excluding self to avoid data leakage)
    def compute_cosine_features(res_dict, query_g):
        query = res_dict[query_g]
        query_norm = query / query.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cosines = []
        for h in range(4):
            if h == query_g:
                continue  # Skip self-cosine (would be 1.0 = data leakage)
            ref = res_dict[h]
            ref_norm = ref / ref.norm(dim=1, keepdim=True).clamp(min=1e-8)
            cos_sim = (query_norm * ref_norm).sum(dim=1)
            cosines.append(cos_sim)
        return torch.stack(cosines, dim=1)  # 3 features
    
    # Pairwise cosines on E_eq
    cosine_features_eq = {g: compute_cosine_features(features_eq, g) for g in range(4)}
    X_train, y_train, X_test, y_test = get_train_test(cosine_features_eq)
    results['pairwise_cosines_E_eq'], _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test, reg_lambda=1.0)
    
    # 8. Pairwise cosines on residuals (original)
    cosine_features = {g: compute_cosine_features(residuals, g) for g in range(4)}
    X_train, y_train, X_test, y_test = get_train_test(cosine_features)
    results['pairwise_cosines'], _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test, reg_lambda=1.0)
    
    # 9. Projection onto M(g) column spaces
    M_col_spaces = {}
    for h in range(4):
        U, S, _ = torch.linalg.svd(M_matrices[h], full_matrices=False)
        rank = (S > 1e-6).sum().item()
        M_col_spaces[h] = U[:, :max(1, rank)]
    
    projection_features = {g: [] for g in range(4)}
    for g in range(4):
        r_g = residuals[g]
        r_g_norm = r_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        r_g_unit = r_g / r_g_norm
        scores = []
        for h in range(4):
            U_col = M_col_spaces[h]
            proj = r_g_unit @ U_col
            proj_norm = proj.norm(dim=1)
            scores.append(proj_norm)
        projection_features[g] = torch.stack(scores, dim=1)
    X_train, y_train, X_test, y_test = get_train_test(projection_features)
    results['projection_scores'], _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test, reg_lambda=1.0)
    
    # 6. Whitening (simplified - skip for speed, use estimate)
    # Full whitening is expensive, use a faster approximation: normalize by per-sample orbit std
    orbit_stds = torch.stack([residuals[g] for g in range(4)], dim=0).std(dim=0)  # (n_samples, D)
    orbit_stds = orbit_stds.clamp(min=1e-8)
    whitened_approx = {g: residuals[g] / orbit_stds for g in range(4)}
    X_train, y_train, X_test, y_test = get_train_test(whitened_approx)
    results['whitened'], _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    
    # 7. Theoretical matching
    theoretical_preds = []
    theoretical_true = []
    for g in range(4):
        for i in range(n_samples):
            r = residuals[g][i]
            scores = [(M_matrices[h].T @ r).norm().item() for h in range(4)]
            theoretical_preds.append(int(np.argmax(scores)))
            theoretical_true.append(g)
    theoretical_preds = torch.tensor(theoretical_preds, device=device)
    theoretical_true = torch.tensor(theoretical_true, device=device)
    results['theoretical_matching'] = (theoretical_preds == theoretical_true).float().mean().item() * 100
    
    # 8. Ratio features
    eps = 1e-6
    r_0 = residuals[0]
    ratio_features = {}
    for g in range(4):
        ratio = residuals[g] / (r_0.abs() + eps) * r_0.sign()
        ratio = ratio.clamp(-100, 100)
        ratio_features[g] = ratio
    X_train, y_train, X_test, y_test = get_train_test(ratio_features)
    results['ratio_features'], _ = compute_linear_probe_closed_form(X_train, y_train, X_test, y_test)
    
    # Track residual and feature norms for diagnostic
    avg_residual_norm = sum(residuals[g].norm(dim=1).mean().item() for g in range(4)) / 4.0
    avg_feature_norm = sum(features[g].norm(dim=1).mean().item() for g in range(4)) / 4.0
    results['avg_residual_norm'] = avg_residual_norm
    results['avg_feature_norm'] = avg_feature_norm
    results['probe_layer'] = probe_layer
    results['probe_layer_dim'] = feat_dim
    
    # Properly decompose E(x) into trivial vs equivariant subspaces
    # P_inv projects onto the trivial (invariant) subspace of the representation
    # (P_inv was already computed using layer_c4_reps above)
    E_x = features[0]  # E(x) for the unrotated input
    
    # Project E(x) onto invariant and equivariant subspaces
    E_inv = E_x @ P_inv.T  # Component in trivial subspace
    E_equiv = E_x - E_inv   # Component in non-trivial (equivariant) subspaces
    
    # Compute norms
    E_inv_norm = E_inv.norm(dim=1)
    E_equiv_norm = E_equiv.norm(dim=1)
    E_total_norm = E_x.norm(dim=1).clamp(min=1e-8)
    
    # Fraction of energy in each subspace
    results['inv_subspace_frac'] = (E_inv_norm / E_total_norm).mean().item()
    results['equiv_subspace_frac'] = (E_equiv_norm / E_total_norm).mean().item()
    results['avg_E_inv_norm'] = E_inv_norm.mean().item()
    results['avg_E_equiv_norm'] = E_equiv_norm.mean().item()
    
    # Compute σ_min (smallest singular value) of each orbit
    # For each sample x, form the 4×D matrix [E(x), E(90°·x), E(180°·x), E(270°·x)]
    # and compute its smallest singular value
    orbit_matrices = torch.stack([features[g] for g in range(4)], dim=1)  # (n_samples, 4, D)
    # Compute SVD for each sample's orbit matrix
    # torch.linalg.svdvals returns singular values in descending order
    sigma_vals = torch.linalg.svdvals(orbit_matrices)  # (n_samples, min(4, D))
    sigma_min_per_sample = sigma_vals[:, -1]  # smallest singular value for each sample
    results['avg_sigma_min'] = sigma_min_per_sample.mean().item()
    results['std_sigma_min'] = sigma_min_per_sample.std().item()
    
    return results


# =============================================================================
# Rotation Functions
# =============================================================================
def rotate_90(x):
    return torch.rot90(x, k=1, dims=(-2, -1))

def rotate_180(x):
    return torch.rot90(x, k=2, dims=(-2, -1))

def rotate_270(x):
    return torch.rot90(x, k=3, dims=(-2, -1))

def identity(x):
    return x

ROTATIONS = [identity, rotate_90, rotate_180, rotate_270]


def apply_rotation_by_index(x, rot_idx):
    """Apply rotation by index (0=identity, 1=90°, 2=180°, 3=270°)."""
    return ROTATIONS[rot_idx](x)


# =============================================================================
# C4 Representations
# =============================================================================
def get_c4_regular_representation():
    """Returns the 4x4 regular representation matrices for C4 group."""
    rho_0 = torch.eye(4)
    rho_90 = torch.tensor([
        [0., 0., 0., 1.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.]
    ])
    rho_180 = rho_90 @ rho_90
    rho_270 = rho_180 @ rho_90
    return [rho_0, rho_90, rho_180, rho_270]


def get_c4_representation_for_latent_dim(latent_dim, device):
    """Create C4 representation matrices: ρ_full(g) = ρ_regular(g) ⊗ I_{latent_dim/4}"""
    assert latent_dim % 4 == 0, "latent_dim must be divisible by 4"
    num_copies = latent_dim // 4
    c4_regular = get_c4_regular_representation()
    identity_block = torch.eye(num_copies)
    representations = []
    for rho in c4_regular:
        rho_full = torch.kron(rho, identity_block).to(device)
        representations.append(rho_full)
    return representations


def apply_random_rotation(x):
    """Apply per-sample random rotations from C4 group."""
    batch_size = x.size(0)
    rotation_indices = torch.randint(0, 4, (batch_size,), device=x.device)
    x_rotated = x.clone()
    for rot_idx in range(4):
        mask = rotation_indices == rot_idx
        if mask.any():
            x_rotated[mask] = ROTATIONS[rot_idx](x[mask])
    return x_rotated, rotation_indices


# =============================================================================
# Model Definition
# =============================================================================
class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128, 64], 
                 latent_dim=64, dropout_rate=0.3, use_layernorm=False):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # Build layers as a list so we can access intermediate outputs
        self.blocks = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            block = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.blocks.append(block)
            prev_dim = hidden_dim
        
        # Final projection to latent space
        self.final_block = nn.Sequential(
            nn.Linear(prev_dim, latent_dim),
            nn.LayerNorm(latent_dim) if use_layernorm else nn.BatchNorm1d(latent_dim),
            nn.GELU()
        )
    
    def forward(self, x, return_intermediate=False):
        x = x.reshape(x.size(0), -1)
        intermediates = []
        for block in self.blocks:
            x = block(x)
            if return_intermediate:
                intermediates.append(x)
        x = self.final_block(x)
        if return_intermediate:
            intermediates.append(x)
            return x, intermediates
        return x


class ClassificationHead(nn.Module):
    def __init__(self, latent_dim=64, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)
    
    def forward(self, features):
        return self.fc(features)


class InvariantMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128, 64], 
                 latent_dim=64, num_classes=10, dropout_rate=0.3, use_layernorm=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.feature_extractor = MLPFeatureExtractor(
            input_dim, hidden_dims, latent_dim, dropout_rate, use_layernorm=use_layernorm)
        self.classifier = ClassificationHead(latent_dim, num_classes)
    
    def get_features(self, x, layer_idx=None, return_dim=False):
        """
        Get features from a specific layer.
        
        Args:
            x: Input tensor
            layer_idx: Which layer to use. None or -1 = final layer (latent_dim), 
                       0-3 = intermediate layers (hidden_dims)
            return_dim: If True, returns (features, dimension). If False, returns just features.
        
        Returns:
            features if return_dim=False, else (features, dim)
        """
        if layer_idx is None or layer_idx == -1:
            # Standard: return final latent features
            features = self.feature_extractor(x)
            if return_dim:
                return features, self.latent_dim
            return features
        else:
            # Get intermediate features
            _, intermediates = self.feature_extractor(x, return_intermediate=True)
            # intermediates[0] = 512, [1] = 256, [2] = 128, [3] = 64, [4] = latent (64)
            if layer_idx < len(intermediates):
                feat = intermediates[layer_idx]
                dim = feat.shape[-1]
            else:
                feat = intermediates[-1]
                dim = self.latent_dim
            if return_dim:
                return feat, dim
            return feat
    
    def get_layer_dim(self, layer_idx=-1):
        """Get the dimensionality of a specific layer."""
        if layer_idx == -1:
            return self.latent_dim
        elif layer_idx < len(self.hidden_dims):
            return self.hidden_dims[layer_idx]
        else:
            return self.latent_dim
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features


# =============================================================================
# Linear Probe
# =============================================================================
class LinearProbe(nn.Module):
    """Simple linear probe to predict rotation from residual features."""
    def __init__(self, latent_dim, num_rotations=4):
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_rotations)
    
    def forward(self, residual):
        return self.linear(residual)


# =============================================================================
# Residual Dataset
# =============================================================================
class ResidualDataset(Dataset):
    """
    Dataset that computes residuals r(g·x) = E(g·x) - mean_h(E(h·x)) on-the-fly.
    
    For each sample x, picks ONE random rotation g per __getitem__ call.
    This ensures different rotations of the same x don't appear in the same batch.
    """
    def __init__(self, base_dataset, model, device, latent_dim, precompute_means=True):
        """
        Args:
            base_dataset: Original MNIST dataset
            model: Trained model with get_features method
            device: torch device
            latent_dim: Dimension of feature space
            precompute_means: If True, precompute orbit means for efficiency
        """
        self.base_dataset = base_dataset
        self.model = model
        self.device = device
        self.latent_dim = latent_dim
        self.precompute_means = precompute_means
        
        if precompute_means:
            self._precompute_orbit_means()
    
    @torch.no_grad()
    def _precompute_orbit_means(self):
        """Precompute orbit means μ(x) = (1/4) Σ_h E(h·x) for all samples."""
        print("Precomputing orbit means...")
        self.model.eval()
        
        # We need to process in batches for efficiency
        batch_size = 256
        
        # Temporary loader
        temp_loader = DataLoader(self.base_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=2)
        
        all_means = []
        all_features = {i: [] for i in range(4)}  # Store features for each rotation
        
        for data, _ in tqdm(temp_loader, desc="Computing orbit means"):
            data = data.to(self.device)
            batch_means = torch.zeros(data.size(0), self.latent_dim, device=self.device)
            
            # Compute features for all rotations
            for rot_idx in range(4):
                data_rotated = apply_rotation_by_index(data, rot_idx)
                features = self.model.get_features(data_rotated)
                batch_means += features
                all_features[rot_idx].append(features.cpu())
            
            batch_means /= 4.0  # Mean over orbit
            all_means.append(batch_means.cpu())
        
        self.orbit_means = torch.cat(all_means, dim=0)
        self.all_features = {i: torch.cat(all_features[i], dim=0) for i in range(4)}
        print(f"Precomputed means for {len(self.orbit_means)} samples, latent_dim={self.latent_dim}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Pick a random rotation for this sample
        rot_idx = random.randint(0, 3)
        
        if self.precompute_means:
            # Use precomputed features
            features_g = self.all_features[rot_idx][idx]
            mean = self.orbit_means[idx]
            residual = features_g - mean
        else:
            # Compute on the fly (slower)
            data, _ = self.base_dataset[idx]
            data = data.unsqueeze(0).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                # Compute mean over orbit
                mean = torch.zeros(1, self.latent_dim, device=self.device)
                for h_idx in range(4):
                    data_h = apply_rotation_by_index(data, h_idx)
                    mean += self.model.get_features(data_h)
                mean /= 4.0
                
                # Compute features for chosen rotation
                data_g = apply_rotation_by_index(data, rot_idx)
                features_g = self.model.get_features(data_g)
                
                residual = (features_g - mean).squeeze(0).cpu()
        
        return residual, rot_idx


class ResidualEvalDataset(Dataset):
    """
    Dataset for evaluation that returns ALL 4 rotations per sample.
    Used for comprehensive evaluation of the probe.
    """
    def __init__(self, base_dataset, model, device, latent_dim):
        self.base_dataset = base_dataset
        self.model = model
        self.device = device
        self.latent_dim = latent_dim
        self._precompute_all()
    
    @torch.no_grad()
    def _precompute_all(self):
        """Precompute all residuals for all rotations."""
        print("Precomputing all residuals for evaluation...")
        self.model.eval()
        
        batch_size = 256
        temp_loader = DataLoader(self.base_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=2)
        
        all_residuals = []
        all_rot_labels = []
        
        for data, _ in tqdm(temp_loader, desc="Computing eval residuals"):
            data = data.to(self.device)
            
            # Compute orbit mean
            batch_mean = torch.zeros(data.size(0), self.latent_dim, device=self.device)
            batch_features = []
            
            for rot_idx in range(4):
                data_rotated = apply_rotation_by_index(data, rot_idx)
                features = self.model.get_features(data_rotated)
                batch_mean += features
                batch_features.append(features)
            
            batch_mean /= 4.0
            
            # Compute residuals for each rotation
            for rot_idx in range(4):
                residual = batch_features[rot_idx] - batch_mean
                all_residuals.append(residual.cpu())
                all_rot_labels.extend([rot_idx] * data.size(0))
        
        self.residuals = torch.cat(all_residuals, dim=0)
        self.rot_labels = torch.tensor(all_rot_labels, dtype=torch.long)
        print(f"Precomputed {len(self.residuals)} residual samples (4 per original), latent_dim={self.latent_dim}")
    
    def __len__(self):
        return len(self.residuals)
    
    def __getitem__(self, idx):
        return self.residuals[idx], self.rot_labels[idx]


# =============================================================================
# Loss Functions for Main Model Training
# =============================================================================
def classification_loss(logits, targets):
    return F.cross_entropy(logits, targets)


def invariance_loss(model, x, c4_representations):
    features_original = model.get_features(x)
    x_rotated, _ = apply_random_rotation(x)
    features_rotated = model.get_features(x_rotated)
    return F.mse_loss(features_rotated, features_original)


def equivariance_loss(model, x, c4_representations):
    features_original = model.get_features(x)
    x_rotated, rotation_indices = apply_random_rotation(x)
    features_rotated = model.get_features(x_rotated)
    features_transformed = torch.zeros_like(features_original)
    for rot_idx in range(4):
        mask = rotation_indices == rot_idx
        if mask.any():
            rho_g = c4_representations[rot_idx]
            features_transformed[mask] = features_original[mask] @ rho_g.T
    return F.mse_loss(features_rotated, features_transformed)


def compute_full_invariance_loss(model, x, c4_representations):
    """Compute invariance loss over all rotations (for evaluation)."""
    features_original = model.get_features(x)
    total_loss = 0.0
    for rotation in ROTATIONS[1:]:
        x_rotated = rotation(x)
        features_rotated = model.get_features(x_rotated)
        total_loss += F.mse_loss(features_rotated, features_original)
    return total_loss / 3


def compute_full_equivariance_loss(model, x, c4_representations):
    """Compute equivariance loss over all rotations (for evaluation)."""
    features_original = model.get_features(x)
    total_loss = 0.0
    for i, rotation in enumerate(ROTATIONS[1:], start=1):
        x_rotated = rotation(x)
        features_rotated = model.get_features(x_rotated)
        rho_g = c4_representations[i]
        features_transformed = features_original @ rho_g.T
        total_loss += F.mse_loss(features_rotated, features_transformed)
    return total_loss / 3


# =============================================================================
# Training Functions for Main Model
# =============================================================================
def train_model_epoch(model, train_loader, optimizer, device, lambda_equiv, loss_type, c4_representations,
                      step_callback=None, global_step_offset=0):
    """
    Train for one epoch. 
    
    Args:
        step_callback: Optional function(model, step) called after each batch during epoch 1
        global_step_offset: Starting step count for this epoch
    
    Returns:
        (cls_loss, equiv_loss, accuracy, final_step)
    """
    model.train()
    equiv_loss_fn = equivariance_loss if loss_type == 'equivariance' else invariance_loss
    total_cls_loss, total_equiv_loss, total_correct, total_samples = 0.0, 0.0, 0, 0
    
    step = global_step_offset
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        data_augmented, _ = apply_random_rotation(data)
        optimizer.zero_grad()
        logits, _ = model(data_augmented)
        cls_loss = classification_loss(logits, targets)
        equiv_loss = equiv_loss_fn(model, data, c4_representations)
        total_loss = cls_loss + lambda_equiv * equiv_loss
        total_loss.backward()
        optimizer.step()
        total_cls_loss += cls_loss.item() * data.size(0)
        total_equiv_loss += equiv_loss.item() * data.size(0)
        _, predicted = logits.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += data.size(0)
        step += 1
        
        # Callback for step-level tracking
        if step_callback is not None:
            step_callback(model, step)
    
    return (total_cls_loss / total_samples, 
            total_equiv_loss / total_samples, 
            100. * total_correct / total_samples,
            step)


@torch.no_grad()
def evaluate_model(model, data_loader, device, loss_type=None, c4_representations=None):
    """Evaluate model accuracy and optionally compute equiv/inv loss."""
    model.eval()
    total_correct, total_samples = 0, 0
    total_equiv_loss = 0.0
    
    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        logits, _ = model(data)
        _, predicted = logits.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += data.size(0)
        
        # Compute equiv/inv loss if requested
        if loss_type is not None and c4_representations is not None:
            if loss_type == 'equivariance':
                equiv_loss = compute_full_equivariance_loss(model, data, c4_representations)
            else:
                equiv_loss = compute_full_invariance_loss(model, data, c4_representations)
            total_equiv_loss += equiv_loss.item() * data.size(0)
    
    acc = 100. * total_correct / total_samples
    
    if loss_type is not None and c4_representations is not None:
        return acc, total_equiv_loss / total_samples
    return acc


# =============================================================================
# Linear Probe Training
# =============================================================================
def train_probe_epoch(probe, train_loader, optimizer, device):
    probe.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    for residuals, rot_labels in train_loader:
        residuals, rot_labels = residuals.to(device), rot_labels.to(device)
        
        optimizer.zero_grad()
        logits = probe(residuals)
        loss = F.cross_entropy(logits, rot_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * residuals.size(0)
        _, predicted = logits.max(1)
        total_correct += predicted.eq(rot_labels).sum().item()
        total_samples += residuals.size(0)
    
    return total_loss / total_samples, 100. * total_correct / total_samples


@torch.no_grad()
def evaluate_probe(probe, eval_loader, device):
    probe.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    # Per-rotation accuracy
    correct_per_rot = {i: 0 for i in range(4)}
    total_per_rot = {i: 0 for i in range(4)}
    
    for residuals, rot_labels in eval_loader:
        residuals, rot_labels = residuals.to(device), rot_labels.to(device)
        
        logits = probe(residuals)
        loss = F.cross_entropy(logits, rot_labels)
        
        total_loss += loss.item() * residuals.size(0)
        _, predicted = logits.max(1)
        total_correct += predicted.eq(rot_labels).sum().item()
        total_samples += residuals.size(0)
        
        for rot_idx in range(4):
            mask = rot_labels == rot_idx
            correct_per_rot[rot_idx] += predicted[mask].eq(rot_labels[mask]).sum().item()
            total_per_rot[rot_idx] += mask.sum().item()
    
    per_rot_acc = {ROTATION_NAMES[i]: 100. * correct_per_rot[i] / total_per_rot[i] 
                   for i in range(4) if total_per_rot[i] > 0}
    
    return total_loss / total_samples, 100. * total_correct / total_samples, per_rot_acc


def train_quick_probe(model, val_dataset, device, latent_dim, num_epochs=10, batch_size=256, lr=1e-2):
    """
    Train a quick linear probe to evaluate how well rotation can be predicted.
    Used for checkpoint evaluations during training.
    """
    model.eval()
    
    # Create residual dataset from validation set
    residual_dataset = ResidualEvalDataset(val_dataset, model, device, latent_dim)
    residual_loader = DataLoader(residual_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Create and train probe
    probe = LinearProbe(latent_dim, num_rotations=4).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    
    for _ in range(num_epochs):
        probe.train()
        for residuals, rot_labels in residual_loader:
            residuals, rot_labels = residuals.to(device), rot_labels.to(device)
            optimizer.zero_grad()
            logits = probe(residuals)
            loss = F.cross_entropy(logits, rot_labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    _, probe_acc, _ = evaluate_probe(probe, residual_loader, device)
    return probe_acc


# =============================================================================
# Main Experiment Function
# =============================================================================
def run_single_experiment(loss_type: str, config: dict, device: torch.device, run_idx: int = 0):
    """Run a single experiment: train model, then train and evaluate linear probe."""
    
    print(f"\n{'='*70}")
    print(f"RUN {run_idx+1} - {loss_type.upper()} MODE")
    print(f"{'='*70}")
    
    # Set seed
    seed = 42 + run_idx * 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize C4 representations
    c4_representations = get_c4_representation_for_latent_dim(config['latent_dim'], device)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_size = int((1 - config['val_split']) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, num_workers=2, pin_memory=True)
    
    # =========================================================================
    # PHASE 1: Train the main classification model
    # =========================================================================
    print(f"\n--- Phase 1: Training {loss_type} model ---")
    
    model = InvariantMLP(
        input_dim=784,
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        num_classes=10,
        dropout_rate=config['dropout_rate'],
        use_layernorm=config.get('use_layernorm', False)
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['learning_rate'], 
                           weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    model_history = {
        'train_cls_loss': [], 'train_equiv_loss': [], 'train_acc': [], 
        'val_acc': [], 'val_equiv_loss': [],
        'probe_checkpoints': [],  # List of checkpoint dicts with probe_acc and global_probe_accs
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    # Helper function to compute and store a probe checkpoint
    def compute_probe_checkpoint(model, step, epoch, is_step_checkpoint=False):
        """Compute probe accuracies and store checkpoint."""
        model.eval()
        val_acc_now, val_equiv_now = evaluate_model(model, val_loader, device, loss_type, c4_representations)
        
        # Standard probe (trained with SGD) - skip for step checkpoints to save time
        if not is_step_checkpoint:
            probe_acc = train_quick_probe(
                model, val_dataset, device, config['latent_dim'],
                num_epochs=config['quick_probe_epochs'],
                batch_size=config['probe_batch_size'],
                lr=config['probe_learning_rate']
            )
        else:
            probe_acc = None  # Skip for step-level checkpoints
        
        # Global probe approaches (closed-form) - fast enough to run at every checkpoint
        probe_layer = config.get('probe_layer', -1)
        global_probe_accs = compute_quick_global_probes(
            model, val_dataset, device, config['latent_dim'],
            c4_representations, max_samples=500, probe_layer=probe_layer
        )
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'val_equiv_loss': val_equiv_now,
            'probe_acc': probe_acc,
            'global_probe_accs': global_probe_accs,
        }
        model_history['probe_checkpoints'].append(checkpoint)
        
        # Print progress
        baseline_acc = global_probe_accs.get('baseline_residuals', 0)
        E_eq_acc = global_probe_accs.get('E_eq_unit', 0)
        res_norm = global_probe_accs.get('avg_residual_norm', 0)
        E_inv = global_probe_accs.get('avg_E_inv_norm', 0)
        E_eq = global_probe_accs.get('avg_E_equiv_norm', 0)
        probe_dim = global_probe_accs.get('probe_layer_dim', config['latent_dim'])
        layer_str = f"[L{probe_layer}:{probe_dim}d]" if probe_layer != -1 else ""
        if is_step_checkpoint:
            print(f"    Step {step}{layer_str}: Loss={val_equiv_now:.4f}, "
                  f"Probe_r={baseline_acc:.1f}%, Probe_Eeq={E_eq_acc:.1f}%, "
                  f"||E_inv||={E_inv:.2f}, ||E_eq||={E_eq:.2f}")
        else:
            print(f"  Epoch {epoch} (step {step}){layer_str}: Val={val_acc_now:.1f}%, Loss={val_equiv_now:.4f}, "
                  f"Probe_r={probe_acc:.1f}%, Probe_Eeq={E_eq_acc:.1f}%, ||E_inv||={E_inv:.2f}, ||E_eq||={E_eq:.2f}")
        
        model.train()
        return val_acc_now, val_equiv_now
    
    # Checkpoint at initialization (step 0)
    print("  Checkpoint at initialization...")
    compute_probe_checkpoint(model, step=0, epoch=0, is_step_checkpoint=True)
    
    # Track steps for epoch 1 callback
    step_interval = config.get('step_probe_interval', 50)
    steps_to_probe = set()
    
    global_step = 0
    for epoch in range(1, config['epochs'] + 1):
        
        # For epoch 1, create step callback for granular tracking
        if epoch == 1:
            # Determine which steps to probe
            steps_per_epoch = len(train_loader)
            steps_to_probe = {i * step_interval for i in range(1, steps_per_epoch // step_interval + 1)}
            
            def step_callback(model, step):
                if step in steps_to_probe:
                    compute_probe_checkpoint(model, step=step, epoch=1, is_step_checkpoint=True)
            
            print(f"  Epoch 1: Probing at steps {sorted(steps_to_probe)} (total {steps_per_epoch} steps)")
            train_cls, train_equiv, train_acc, global_step = train_model_epoch(
                model, train_loader, optimizer, device, 
                config['lambda_equiv'], loss_type, c4_representations,
                step_callback=step_callback, global_step_offset=global_step)
        else:
            # Normal training without step callbacks
            train_cls, train_equiv, train_acc, global_step = train_model_epoch(
                model, train_loader, optimizer, device, 
                config['lambda_equiv'], loss_type, c4_representations,
                step_callback=None, global_step_offset=global_step)
        
        val_acc, val_equiv = evaluate_model(model, val_loader, device, loss_type, c4_representations)
        scheduler.step()
        
        model_history['train_cls_loss'].append(train_cls)
        model_history['train_equiv_loss'].append(train_equiv)
        model_history['train_acc'].append(train_acc)
        model_history['val_acc'].append(val_acc)
        model_history['val_equiv_loss'].append(val_equiv)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Epoch-level probe checkpoint
        if epoch == 1 or epoch % config['probe_every'] == 0:
            compute_probe_checkpoint(model, step=global_step, epoch=epoch, is_step_checkpoint=False)
        elif epoch % 10 == 0:
            print(f"  Epoch {epoch}/{config['epochs']} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(best_model_state)
    model = model.to(device)
    model_test_acc = evaluate_model(model, test_loader, device)
    print(f"  Final Model Test Accuracy (digit classification): {model_test_acc:.2f}%")
    
    model_history['test_acc'] = model_test_acc
    
    # =========================================================================
    # DIAGNOSTIC ANALYSIS (between Phase 1 and Phase 2)
    # =========================================================================
    print(f"\n--- Diagnostic Analysis: Residual Separability ---")
    
    # Run comprehensive diagnostic analysis
    diagnostic_results = analyze_residual_separability(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        device=device,
        latent_dim=config['latent_dim'],
        verbose=True
    )
    
    # Run global probe approaches analysis
    print(f"\n--- Global Probe Analysis: Testing E(x) Normalization Approaches ---")
    global_probe_results = analyze_global_probe_approaches(
        model=model,
        test_dataset=test_dataset,
        device=device,
        latent_dim=config['latent_dim'],
        c4_representations=c4_representations,
        verbose=True
    )
    
    # Run residual decomposition analysis
    print(f"\n--- Residual Decomposition: Structured vs Error Components ---")
    decomposition_results = analyze_residual_decomposition(
        model=model,
        test_dataset=test_dataset,
        device=device,
        latent_dim=config['latent_dim'],
        c4_representations=c4_representations,
        verbose=True
    )
    
    # =========================================================================
    # PHASE 2: Create residual datasets and train linear probe
    # =========================================================================
    print(f"\n--- Phase 2: Training linear probe ---")
    
    # Freeze the model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Create residual datasets
    latent_dim = config['latent_dim']
    
    # For training: one random rotation per sample per epoch
    train_residual_dataset = ResidualDataset(train_dataset, model, device, latent_dim, precompute_means=True)
    
    # For evaluation: all 4 rotations per sample
    test_residual_dataset = ResidualEvalDataset(test_dataset, model, device, latent_dim)
    
    train_residual_loader = DataLoader(train_residual_dataset, 
                                       batch_size=config['probe_batch_size'],
                                       shuffle=True, num_workers=0)  # num_workers=0 for custom dataset
    test_residual_loader = DataLoader(test_residual_dataset,
                                      batch_size=config['probe_batch_size'],
                                      shuffle=False, num_workers=0)
    
    # Create and train probe
    probe = LinearProbe(config['latent_dim'], num_rotations=4).to(device)
    probe_optimizer = optim.Adam(probe.parameters(), lr=config['probe_learning_rate'])
    probe_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        probe_optimizer, T_max=config['probe_epochs'], eta_min=1e-5)
    
    probe_history = {
        'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'test_per_rot_acc': []
    }
    
    for epoch in range(1, config['probe_epochs'] + 1):
        train_loss, probe_train_acc = train_probe_epoch(probe, train_residual_loader, probe_optimizer, device)
        probe_test_loss, probe_test_acc, per_rot_acc = evaluate_probe(probe, test_residual_loader, device)
        probe_scheduler.step()
        
        probe_history['train_loss'].append(train_loss)
        probe_history['train_acc'].append(probe_train_acc)
        probe_history['test_loss'].append(probe_test_loss)
        probe_history['test_acc'].append(probe_test_acc)
        probe_history['test_per_rot_acc'].append(per_rot_acc)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Probe Epoch {epoch}/{config['probe_epochs']} - "
                  f"Train Acc: {probe_train_acc:.2f}% - Test Acc: {probe_test_acc:.2f}%")
    
    # Final evaluation
    final_test_loss, final_test_acc, final_per_rot_acc = evaluate_probe(
        probe, test_residual_loader, device)
    
    print(f"\n  Final Probe Results:")
    print(f"    Overall Test Accuracy (rotation prediction): {final_test_acc:.2f}%")
    print(f"    Per-Rotation Accuracy:")
    for name, acc in final_per_rot_acc.items():
        print(f"      {name}: {acc:.2f}%")
    
    # Debug: verify we're returning different values
    print(f"\n  DEBUG: model_test_acc={model_test_acc:.4f}% (digit), probe_test_acc={final_test_acc:.4f}% (rotation)")
    
    return {
        'loss_type': loss_type,
        'model_history': model_history,
        'probe_history': probe_history,
        'final_model_test_acc': model_test_acc,
        'final_probe_test_acc': final_test_acc,
        'final_per_rot_acc': final_per_rot_acc,
        'diagnostic_results': diagnostic_results,
        'global_probe_results': global_probe_results,
        'decomposition_results': decomposition_results,
    }


def run_multiple_experiments(loss_type: str, num_runs: int, config: dict, device: torch.device):
    """Run multiple experiments and aggregate results."""
    all_results = []
    for run_idx in range(num_runs):
        result = run_single_experiment(loss_type, config, device, run_idx)
        all_results.append(result)
    return all_results


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_comparison(equiv_results: List[Dict], inv_results: List[Dict], output_dir: str = '.'):
    """Plot comparison of probe performance between equivariance and invariance modes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    num_runs = len(equiv_results)
    
    equiv_color = '#2ecc71'
    inv_color = '#e74c3c'
    alpha_fill = 0.2
    
    # Get probe layer info
    equiv_checkpoints = [r['model_history']['probe_checkpoints'] for r in equiv_results]
    probe_layer = equiv_checkpoints[0][0]['global_probe_accs'].get('probe_layer', -1) if equiv_checkpoints[0] else -1
    probe_dim = equiv_checkpoints[0][0]['global_probe_accs'].get('probe_layer_dim', 64) if equiv_checkpoints[0] else 64
    layer_str = f" [Layer {probe_layer}, {probe_dim}d]" if probe_layer != -1 else ""
    
    fig.suptitle(f'Linear Probe Comparison: Equivariance vs Invariance ({num_runs} runs){layer_str}', fontsize=14, y=1.02)
    
    # -------------------------------------------------------------------------
    # Plot 1: Probe Training Accuracy over epochs
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    
    equiv_train_accs = np.array([r['probe_history']['train_acc'] for r in equiv_results])
    inv_train_accs = np.array([r['probe_history']['train_acc'] for r in inv_results])
    epochs = range(1, len(equiv_train_accs[0]) + 1)
    
    equiv_mean = np.mean(equiv_train_accs, axis=0)
    equiv_std = np.std(equiv_train_accs, axis=0)
    ax.plot(epochs, equiv_mean, color=equiv_color, linewidth=2, label='Equivariance')
    ax.fill_between(epochs, equiv_mean - equiv_std, equiv_mean + equiv_std, 
                   color=equiv_color, alpha=alpha_fill)
    
    inv_mean = np.mean(inv_train_accs, axis=0)
    inv_std = np.std(inv_train_accs, axis=0)
    ax.plot(epochs, inv_mean, color=inv_color, linewidth=2, label='Invariance')
    ax.fill_between(epochs, inv_mean - inv_std, inv_mean + inv_std, 
                   color=inv_color, alpha=alpha_fill)
    
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Chance (25%)')
    ax.set_xlabel('Probe Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title('Probe Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Probe Test Accuracy over epochs
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    
    equiv_test_accs = np.array([r['probe_history']['test_acc'] for r in equiv_results])
    inv_test_accs = np.array([r['probe_history']['test_acc'] for r in inv_results])
    
    equiv_mean = np.mean(equiv_test_accs, axis=0)
    equiv_std = np.std(equiv_test_accs, axis=0)
    ax.plot(epochs, equiv_mean, color=equiv_color, linewidth=2, label='Equivariance')
    ax.fill_between(epochs, equiv_mean - equiv_std, equiv_mean + equiv_std, 
                   color=equiv_color, alpha=alpha_fill)
    
    inv_mean = np.mean(inv_test_accs, axis=0)
    inv_std = np.std(inv_test_accs, axis=0)
    ax.plot(epochs, inv_mean, color=inv_color, linewidth=2, label='Invariance')
    ax.fill_between(epochs, inv_mean - inv_std, inv_mean + inv_std, 
                   color=inv_color, alpha=alpha_fill)
    
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Chance (25%)')
    ax.set_xlabel('Probe Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Probe Test Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Final Probe Accuracy Bar Plot
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    
    equiv_final_accs = [r['final_probe_test_acc'] for r in equiv_results]
    inv_final_accs = [r['final_probe_test_acc'] for r in inv_results]
    
    x = np.arange(2)
    means = [np.mean(equiv_final_accs), np.mean(inv_final_accs)]
    stds = [np.std(equiv_final_accs), np.std(inv_final_accs)]
    colors = [equiv_color, inv_color]
    
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=10, alpha=0.8)
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Chance (25%)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Equivariance', 'Invariance'])
    ax.set_ylabel('Final Probe Test Accuracy (%)')
    ax.set_title('Final Probe Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{mean:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Plot 4: Per-Rotation Accuracy
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    
    equiv_rot_accs = {name: [] for name in ROTATION_NAMES}
    inv_rot_accs = {name: [] for name in ROTATION_NAMES}
    
    for r in equiv_results:
        for name in ROTATION_NAMES:
            equiv_rot_accs[name].append(r['final_per_rot_acc'][name])
    
    for r in inv_results:
        for name in ROTATION_NAMES:
            inv_rot_accs[name].append(r['final_per_rot_acc'][name])
    
    x = np.arange(len(ROTATION_NAMES))
    width = 0.35
    
    equiv_means = [np.mean(equiv_rot_accs[name]) for name in ROTATION_NAMES]
    equiv_stds = [np.std(equiv_rot_accs[name]) for name in ROTATION_NAMES]
    inv_means = [np.mean(inv_rot_accs[name]) for name in ROTATION_NAMES]
    inv_stds = [np.std(inv_rot_accs[name]) for name in ROTATION_NAMES]
    
    ax.bar(x - width/2, equiv_means, width, yerr=equiv_stds, label='Equivariance', 
           color=equiv_color, capsize=5, alpha=0.8)
    ax.bar(x + width/2, inv_means, width, yerr=inv_stds, label='Invariance', 
           color=inv_color, capsize=5, alpha=0.8)
    
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels(ROTATION_NAMES)
    ax.set_xlabel('Rotation')
    ax.set_ylabel('Probe Accuracy (%)')
    ax.set_title('Per-Rotation Probe Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'linear_probe_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Print Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY: Linear Probe Results")
    print("="*70)
    
    print(f"\nEQUIVARIANCE Mode:")
    print(f"  Model Test Acc: {np.mean([r['final_model_test_acc'] for r in equiv_results]):.2f}% "
          f"± {np.std([r['final_model_test_acc'] for r in equiv_results]):.2f}%")
    print(f"  Probe Test Acc: {np.mean(equiv_final_accs):.2f}% ± {np.std(equiv_final_accs):.2f}%")
    
    print(f"\nINVARIANCE Mode:")
    print(f"  Model Test Acc: {np.mean([r['final_model_test_acc'] for r in inv_results]):.2f}% "
          f"± {np.std([r['final_model_test_acc'] for r in inv_results]):.2f}%")
    print(f"  Probe Test Acc: {np.mean(inv_final_accs):.2f}% ± {np.std(inv_final_accs):.2f}%")
    
    print(f"\nInterpretation:")
    if np.mean(equiv_final_accs) > np.mean(inv_final_accs) + 5:
        print("  ✓ Equivariant features encode rotation information better (as expected)")
    elif np.mean(inv_final_accs) > np.mean(equiv_final_accs) + 5:
        print("  ✗ Unexpected: Invariant features encode rotation better than equivariant")
    else:
        print("  ~ Both modes encode similar amounts of rotation information")
    
    if np.mean(inv_final_accs) < 35:
        print("  ✓ Invariant features are close to chance (25%) for rotation prediction")
    else:
        print("  ! Invariant features still encode significant rotation information")


def plot_equiv_loss_vs_probe_acc(equiv_results: List[Dict], inv_results: List[Dict], output_dir: str = '.'):
    """
    Plot equiv/inv loss (x-axis) vs probe accuracy (y-axis) over training.
    Shows trajectory as training progresses with confidence bands.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    num_runs = len(equiv_results)
    
    equiv_color = '#2ecc71'
    inv_color = '#e74c3c'
    alpha_fill = 0.15
    
    # Get probe layer info
    all_checkpoints = [r['model_history']['probe_checkpoints'] for r in equiv_results]
    probe_layer = all_checkpoints[0][0]['global_probe_accs'].get('probe_layer', -1) if all_checkpoints[0] else -1
    probe_dim = all_checkpoints[0][0]['global_probe_accs'].get('probe_layer_dim', 64) if all_checkpoints[0] else 64
    layer_str = f" [Layer {probe_layer}, {probe_dim}d]" if probe_layer != -1 else ""
    
    fig.suptitle(f'Equiv/Inv Loss vs Probe Accuracy During Training ({num_runs} runs){layer_str}', fontsize=14)
    
    # -------------------------------------------------------------------------
    # Left plot: Trajectory plot with error bands
    # -------------------------------------------------------------------------
    ax = axes[0]
    
    # Extract checkpoint data for equivariance (only epoch-level checkpoints with probe_acc)
    equiv_checkpoints = [[cp for cp in r['model_history']['probe_checkpoints'] if cp.get('probe_acc') is not None] 
                        for r in equiv_results]
    inv_checkpoints = [[cp for cp in r['model_history']['probe_checkpoints'] if cp.get('probe_acc') is not None] 
                      for r in inv_results]
    
    # Get number of checkpoints (should be same for all runs)
    n_checkpoints = len(equiv_checkpoints[0])
    
    # Aggregate equiv data
    equiv_losses = np.array([[cp['val_equiv_loss'] for cp in run] for run in equiv_checkpoints])
    equiv_accs = np.array([[cp['probe_acc'] for cp in run] for run in equiv_checkpoints])
    
    equiv_loss_mean = np.mean(equiv_losses, axis=0)
    equiv_loss_std = np.std(equiv_losses, axis=0)
    equiv_acc_mean = np.mean(equiv_accs, axis=0)
    equiv_acc_std = np.std(equiv_accs, axis=0)
    
    # Aggregate inv data
    inv_losses = np.array([[cp['val_equiv_loss'] for cp in run] for run in inv_checkpoints])
    inv_accs = np.array([[cp['probe_acc'] for cp in run] for run in inv_checkpoints])
    
    inv_loss_mean = np.mean(inv_losses, axis=0)
    inv_loss_std = np.std(inv_losses, axis=0)
    inv_acc_mean = np.mean(inv_accs, axis=0)
    inv_acc_std = np.std(inv_accs, axis=0)
    
    # Plot equivariance trajectory
    ax.plot(equiv_loss_mean, equiv_acc_mean, 'o-', color=equiv_color, linewidth=2, 
            markersize=8, label='Equivariance')
    ax.fill_between(equiv_loss_mean, equiv_acc_mean - equiv_acc_std, equiv_acc_mean + equiv_acc_std,
                   color=equiv_color, alpha=alpha_fill)
    # Add horizontal error bars for loss
    ax.errorbar(equiv_loss_mean, equiv_acc_mean, xerr=equiv_loss_std, fmt='none', 
                color=equiv_color, alpha=0.5, capsize=3)
    
    # Plot invariance trajectory
    ax.plot(inv_loss_mean, inv_acc_mean, 's-', color=inv_color, linewidth=2, 
            markersize=8, label='Invariance')
    ax.fill_between(inv_loss_mean, inv_acc_mean - inv_acc_std, inv_acc_mean + inv_acc_std,
                   color=inv_color, alpha=alpha_fill)
    ax.errorbar(inv_loss_mean, inv_acc_mean, xerr=inv_loss_std, fmt='none', 
                color=inv_color, alpha=0.5, capsize=3)
    
    # Mark start and end points
    ax.scatter([equiv_loss_mean[0]], [equiv_acc_mean[0]], color=equiv_color, s=150, 
               marker='>', zorder=5, edgecolor='black', linewidth=1.5)
    ax.scatter([equiv_loss_mean[-1]], [equiv_acc_mean[-1]], color=equiv_color, s=150, 
               marker='*', zorder=5, edgecolor='black', linewidth=1.5)
    ax.scatter([inv_loss_mean[0]], [inv_acc_mean[0]], color=inv_color, s=150, 
               marker='>', zorder=5, edgecolor='black', linewidth=1.5)
    ax.scatter([inv_loss_mean[-1]], [inv_acc_mean[-1]], color=inv_color, s=150, 
               marker='*', zorder=5, edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Chance (25%)')
    ax.set_xlabel('Validation Equiv/Inv Loss', fontsize=12)
    ax.set_ylabel('Probe Accuracy (%)', fontsize=12)
    ax.set_title('Training Trajectory (▶=start, ★=end)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Right plot: Final loss vs final probe accuracy scatter
    # -------------------------------------------------------------------------
    ax = axes[1]
    
    # Get final checkpoint values
    equiv_final_losses = equiv_losses[:, -1]
    equiv_final_accs = equiv_accs[:, -1]
    inv_final_losses = inv_losses[:, -1]
    inv_final_accs = inv_accs[:, -1]
    
    ax.scatter(equiv_final_losses, equiv_final_accs, color=equiv_color, s=100, 
               alpha=0.7, label='Equivariance', marker='o', edgecolor='black')
    ax.scatter(inv_final_losses, inv_final_accs, color=inv_color, s=100, 
               alpha=0.7, label='Invariance', marker='s', edgecolor='black')
    
    # Add mean markers
    ax.scatter([np.mean(equiv_final_losses)], [np.mean(equiv_final_accs)], 
               color=equiv_color, s=250, marker='*', edgecolor='black', linewidth=2,
               label=f'Equiv Mean', zorder=5)
    ax.scatter([np.mean(inv_final_losses)], [np.mean(inv_final_accs)], 
               color=inv_color, s=250, marker='*', edgecolor='black', linewidth=2,
               label=f'Inv Mean', zorder=5)
    
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Final Validation Equiv/Inv Loss', fontsize=12)
    ax.set_ylabel('Final Probe Accuracy (%)', fontsize=12)
    ax.set_title('Final Values (individual runs)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'equiv_loss_vs_probe_acc.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print checkpoint summary
    print("\n" + "="*70)
    print("CHECKPOINT SUMMARY: Equiv/Inv Loss vs Probe Accuracy")
    print("="*70)
    print(f"\nEquivariance checkpoints:")
    for i, (loss, acc) in enumerate(zip(equiv_loss_mean, equiv_acc_mean)):
        epoch = equiv_checkpoints[0][i]['epoch']
        print(f"  Epoch {epoch:3d}: Loss={loss:.4f}±{equiv_loss_std[i]:.4f}, Probe Acc={acc:.2f}%±{equiv_acc_std[i]:.2f}%")
    
    print(f"\nInvariance checkpoints:")
    for i, (loss, acc) in enumerate(zip(inv_loss_mean, inv_acc_mean)):
        epoch = inv_checkpoints[0][i]['epoch']
        print(f"  Epoch {epoch:3d}: Loss={loss:.4f}±{inv_loss_std[i]:.4f}, Probe Acc={acc:.2f}%±{inv_acc_std[i]:.2f}%")


def plot_global_probe_trajectories(equiv_results: List[Dict], inv_results: List[Dict], output_dir: str = '.'):
    """
    Plot equiv/inv loss (x-axis) vs probe accuracy (y-axis) for all global probe approaches.
    Each approach gets its own subplot showing how it evolves during training.
    Now includes E_eq probes which are the key insight.
    """
    approach_names = [
        'baseline_residuals', 'E_eq_unit', 'E_eq_direct', 'E_eq_normalized',
        'normalize_by_Ex', 'unit_direction', 'pairwise_cosines_E_eq', 'whitened'
    ]
    approach_labels = [
        'Baseline r(g·x)', 'E_eq Unit (KEY)', 'E_eq Direct', 'E_eq Normalized',
        'Normalized r/||E(x)||', 'Unit Direction', 'Pairwise Cos (E_eq)', 'Whitened'
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    equiv_color = '#2ecc71'
    inv_color = '#e74c3c'
    alpha_fill = 0.15
    
    # Extract checkpoint data
    equiv_checkpoints = [r['model_history']['probe_checkpoints'] for r in equiv_results]
    inv_checkpoints = [r['model_history']['probe_checkpoints'] for r in inv_results]
    
    # Check if global probe data exists
    if 'global_probe_accs' not in equiv_checkpoints[0][0]:
        print("Warning: No global_probe_accs data found in checkpoints. Skipping global probe trajectory plot.")
        plt.close()
        return
    
    # Get probe layer info if available
    probe_layer = equiv_checkpoints[0][0]['global_probe_accs'].get('probe_layer', -1)
    probe_dim = equiv_checkpoints[0][0]['global_probe_accs'].get('probe_layer_dim', 64)
    layer_str = f" [Layer {probe_layer}, {probe_dim}d]" if probe_layer != -1 else " [Final Layer]"
    
    fig.suptitle(f'Global Probe Approaches: Equiv/Inv Loss vs Probe Accuracy{layer_str}', fontsize=14, y=1.02)
    
    n_checkpoints = len(equiv_checkpoints[0])
    
    # Get loss data (same for all approaches)
    equiv_losses = np.array([[cp['val_equiv_loss'] for cp in run] for run in equiv_checkpoints])
    inv_losses = np.array([[cp['val_equiv_loss'] for cp in run] for run in inv_checkpoints])
    
    equiv_loss_mean = np.mean(equiv_losses, axis=0)
    equiv_loss_std = np.std(equiv_losses, axis=0)
    inv_loss_mean = np.mean(inv_losses, axis=0)
    inv_loss_std = np.std(inv_losses, axis=0)
    
    for idx, (approach, label) in enumerate(zip(approach_names, approach_labels)):
        ax = axes[idx]
        
        # Extract accuracy data for this approach
        try:
            equiv_accs = np.array([[cp['global_probe_accs'].get(approach, 25.0) 
                                   for cp in run] for run in equiv_checkpoints])
            inv_accs = np.array([[cp['global_probe_accs'].get(approach, 25.0) 
                                 for cp in run] for run in inv_checkpoints])
        except (KeyError, TypeError):
            # If data doesn't exist, skip this subplot
            ax.text(0.5, 0.5, f'No data for\n{label}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(label, fontsize=11)
            continue
        
        equiv_acc_mean = np.mean(equiv_accs, axis=0)
        equiv_acc_std = np.std(equiv_accs, axis=0)
        inv_acc_mean = np.mean(inv_accs, axis=0)
        inv_acc_std = np.std(inv_accs, axis=0)
        
        # Plot equivariance trajectory
        ax.plot(equiv_loss_mean, equiv_acc_mean, 'o-', color=equiv_color, linewidth=2, 
                markersize=6, label='Equiv')
        ax.fill_between(equiv_loss_mean, equiv_acc_mean - equiv_acc_std, equiv_acc_mean + equiv_acc_std,
                       color=equiv_color, alpha=alpha_fill)
        ax.errorbar(equiv_loss_mean, equiv_acc_mean, xerr=equiv_loss_std, fmt='none', 
                    color=equiv_color, alpha=0.4, capsize=2)
        
        # Plot invariance trajectory
        ax.plot(inv_loss_mean, inv_acc_mean, 's-', color=inv_color, linewidth=2, 
                markersize=6, label='Inv')
        ax.fill_between(inv_loss_mean, inv_acc_mean - inv_acc_std, inv_acc_mean + inv_acc_std,
                       color=inv_color, alpha=alpha_fill)
        ax.errorbar(inv_loss_mean, inv_acc_mean, xerr=inv_loss_std, fmt='none', 
                    color=inv_color, alpha=0.4, capsize=2)
        
        # Mark start and end points
        ax.scatter([equiv_loss_mean[0]], [equiv_acc_mean[0]], color=equiv_color, s=80, 
                   marker='>', zorder=5, edgecolor='black', linewidth=1)
        ax.scatter([equiv_loss_mean[-1]], [equiv_acc_mean[-1]], color=equiv_color, s=80, 
                   marker='*', zorder=5, edgecolor='black', linewidth=1)
        ax.scatter([inv_loss_mean[0]], [inv_acc_mean[0]], color=inv_color, s=80, 
                   marker='>', zorder=5, edgecolor='black', linewidth=1)
        ax.scatter([inv_loss_mean[-1]], [inv_acc_mean[-1]], color=inv_color, s=80, 
                   marker='*', zorder=5, edgecolor='black', linewidth=1)
        
        ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_xlabel('Equiv/Inv Loss', fontsize=10)
        ax.set_ylabel('Probe Acc (%)', fontsize=10)
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add final accuracy annotation
        final_equiv = equiv_acc_mean[-1]
        final_inv = inv_acc_mean[-1]
        ax.annotate(f'E:{final_equiv:.1f}%\nI:{final_inv:.1f}%', 
                   xy=(0.95, 0.05), xycoords='axes fraction',
                   fontsize=8, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'global_probe_trajectories.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("GLOBAL PROBE APPROACHES: Final Accuracy Summary")
    print("="*80)
    print(f"\n{'Approach':<25} {'Equiv Final':<15} {'Inv Final':<15} {'Difference':<12}")
    print("-"*70)
    
    for approach, label in zip(approach_names, approach_labels):
        try:
            equiv_accs = np.array([[cp['global_probe_accs'].get(approach, 25.0) 
                                   for cp in run] for run in equiv_checkpoints])
            inv_accs = np.array([[cp['global_probe_accs'].get(approach, 25.0) 
                                 for cp in run] for run in inv_checkpoints])
            
            equiv_final = np.mean(equiv_accs[:, -1])
            equiv_std = np.std(equiv_accs[:, -1])
            inv_final = np.mean(inv_accs[:, -1])
            inv_std = np.std(inv_accs[:, -1])
            diff = equiv_final - inv_final
            
            print(f"{label:<25} {equiv_final:.1f}%±{equiv_std:.1f}%  {inv_final:.1f}%±{inv_std:.1f}%  {diff:+.1f}%")
        except (KeyError, TypeError):
            print(f"{label:<25} {'N/A':<15} {'N/A':<15} {'N/A':<12}")


def plot_global_probe_by_step(equiv_results: List[Dict], inv_results: List[Dict], output_dir: str = '.'):
    """
    Plot probe accuracy vs training step for all global probe approaches.
    X-axis is now training step (not loss), providing a clearer view of the training dynamics.
    Includes E_eq probes which are the key for rotation prediction.
    """
    approach_names = [
        'baseline_residuals', 'E_eq_unit', 'E_eq_direct', 'E_eq_normalized',
        'normalize_by_Ex', 'unit_direction', 'pairwise_cosines_E_eq', 'whitened'
    ]
    approach_labels = [
        'Baseline r(g·x)', 'E_eq Unit (KEY)', 'E_eq Direct', 'E_eq Normalized',
        'Normalized r/||E(x)||', 'Unit Direction', 'Pairwise Cos (E_eq)', 'Whitened'
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    equiv_color = '#2ecc71'
    inv_color = '#e74c3c'
    alpha_fill = 0.15
    
    # Extract checkpoint data
    equiv_checkpoints = [r['model_history']['probe_checkpoints'] for r in equiv_results]
    inv_checkpoints = [r['model_history']['probe_checkpoints'] for r in inv_results]
    
    # Check if global probe data exists
    if 'global_probe_accs' not in equiv_checkpoints[0][0]:
        print("Warning: No global_probe_accs data found in checkpoints. Skipping step plot.")
        plt.close()
        return
    
    # Get probe layer info
    probe_layer = equiv_checkpoints[0][0]['global_probe_accs'].get('probe_layer', -1)
    probe_dim = equiv_checkpoints[0][0]['global_probe_accs'].get('probe_layer_dim', 64)
    layer_str = f" [Layer {probe_layer}, {probe_dim}d]" if probe_layer != -1 else " [Final Layer]"
    
    fig.suptitle(f'Global Probe Approaches: Accuracy vs Training Step{layer_str}', fontsize=14, y=1.02)
    
    # Check if step data exists
    if 'step' not in equiv_checkpoints[0][0]:
        print("Warning: No step data found in checkpoints. Skipping step plot.")
        plt.close()
        return
    
    n_checkpoints = len(equiv_checkpoints[0])
    
    # Get step data
    equiv_steps = np.array([[cp.get('step', i) for i, cp in enumerate(run)] for run in equiv_checkpoints])
    inv_steps = np.array([[cp.get('step', i) for i, cp in enumerate(run)] for run in inv_checkpoints])
    
    equiv_step_mean = np.mean(equiv_steps, axis=0)
    inv_step_mean = np.mean(inv_steps, axis=0)
    
    # Also get loss data for annotations
    equiv_losses = np.array([[cp['val_equiv_loss'] for cp in run] for run in equiv_checkpoints])
    inv_losses = np.array([[cp['val_equiv_loss'] for cp in run] for run in inv_checkpoints])
    
    for idx, (approach, label) in enumerate(zip(approach_names, approach_labels)):
        ax = axes[idx]
        
        # Extract accuracy data for this approach
        try:
            equiv_accs = np.array([[cp['global_probe_accs'].get(approach, 25.0) 
                                   for cp in run] for run in equiv_checkpoints])
            inv_accs = np.array([[cp['global_probe_accs'].get(approach, 25.0) 
                                 for cp in run] for run in inv_checkpoints])
        except (KeyError, TypeError):
            ax.text(0.5, 0.5, f'No data for\n{label}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(label, fontsize=11)
            continue
        
        equiv_acc_mean = np.mean(equiv_accs, axis=0)
        equiv_acc_std = np.std(equiv_accs, axis=0)
        inv_acc_mean = np.mean(inv_accs, axis=0)
        inv_acc_std = np.std(inv_accs, axis=0)
        
        # Plot equivariance trajectory
        ax.plot(equiv_step_mean, equiv_acc_mean, 'o-', color=equiv_color, linewidth=2, 
                markersize=5, label='Equiv', alpha=0.9)
        ax.fill_between(equiv_step_mean, equiv_acc_mean - equiv_acc_std, equiv_acc_mean + equiv_acc_std,
                       color=equiv_color, alpha=alpha_fill)
        
        # Plot invariance trajectory
        ax.plot(inv_step_mean, inv_acc_mean, 's-', color=inv_color, linewidth=2, 
                markersize=5, label='Inv', alpha=0.9)
        ax.fill_between(inv_step_mean, inv_acc_mean - inv_acc_std, inv_acc_mean + inv_acc_std,
                       color=inv_color, alpha=alpha_fill)
        
        # Mark initialization and end points
        ax.scatter([equiv_step_mean[0]], [equiv_acc_mean[0]], color=equiv_color, s=100, 
                   marker='o', zorder=5, edgecolor='black', linewidth=1.5, label='Init')
        ax.scatter([equiv_step_mean[-1]], [equiv_acc_mean[-1]], color=equiv_color, s=100, 
                   marker='*', zorder=5, edgecolor='black', linewidth=1.5)
        ax.scatter([inv_step_mean[0]], [inv_acc_mean[0]], color=inv_color, s=100, 
                   marker='o', zorder=5, edgecolor='black', linewidth=1.5)
        ax.scatter([inv_step_mean[-1]], [inv_acc_mean[-1]], color=inv_color, s=100, 
                   marker='*', zorder=5, edgecolor='black', linewidth=1.5)
        
        ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7, linewidth=1, label='Chance')
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel('Probe Acc (%)', fontsize=10)
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Add init and final accuracy annotation
        init_equiv = equiv_acc_mean[0]
        init_inv = inv_acc_mean[0]
        final_equiv = equiv_acc_mean[-1]
        final_inv = inv_acc_mean[-1]
        ax.annotate(f'Init: E={init_equiv:.0f}%, I={init_inv:.0f}%\nFinal: E={final_equiv:.0f}%, I={final_inv:.0f}%', 
                   xy=(0.98, 0.02), xycoords='axes fraction',
                   fontsize=7, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'global_probe_by_step.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print detailed step-by-step summary for E_eq_unit (the key approach)
    print("\n" + "="*100)
    print(f"E_eq PROBES: Step-by-Step Accuracy{layer_str} (key for rotation prediction)")
    print("="*100)
    
    for run_idx, checkpoints in enumerate(equiv_checkpoints):
        if len(equiv_results) > 1:
            print(f"\n--- Equivariance Run {run_idx+1} ---")
        for cp in checkpoints:
            step = cp.get('step', '?')
            epoch = cp.get('epoch', '?')
            loss = cp['val_equiv_loss']
            E_eq_acc = cp['global_probe_accs'].get('E_eq_unit', 0)
            baseline = cp['global_probe_accs'].get('baseline_residuals', 0)
            E_inv_norm = cp['global_probe_accs'].get('avg_E_inv_norm', 0)
            E_eq_norm = cp['global_probe_accs'].get('avg_E_equiv_norm', 0)
            print(f"  Step {step:>4} (epoch {epoch}): Loss={loss:.4f}, E_eq={E_eq_acc:.1f}%, Baseline={baseline:.1f}%, ||E_inv||={E_inv_norm:.2f}, ||E_eq||={E_eq_norm:.2f}")
    
    for run_idx, checkpoints in enumerate(inv_checkpoints):
        if len(inv_results) > 1:
            print(f"\n--- Invariance Run {run_idx+1} ---")
        else:
            print(f"\n--- Invariance ---")
        for cp in checkpoints:
            step = cp.get('step', '?')
            epoch = cp.get('epoch', '?')
            loss = cp['val_equiv_loss']
            E_eq_acc = cp['global_probe_accs'].get('E_eq_unit', 0)
            baseline = cp['global_probe_accs'].get('baseline_residuals', 0)
            E_inv_norm = cp['global_probe_accs'].get('avg_E_inv_norm', 0)
            E_eq_norm = cp['global_probe_accs'].get('avg_E_equiv_norm', 0)
            print(f"  Step {step:>4} (epoch {epoch}): Loss={loss:.4f}, E_eq={E_eq_acc:.1f}%, Baseline={baseline:.1f}%, ||E_inv||={E_inv_norm:.2f}, ||E_eq||={E_eq_norm:.2f}")


def plot_diagnostic_comparison(equiv_results: List[Dict], inv_results: List[Dict], output_dir: str = '.'):
    """
    Plot comparison of diagnostic results between equivariance and invariance modes.
    Shows how different input representations perform for linear probes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    equiv_color = '#2ecc71'
    inv_color = '#e74c3c'
    
    # Get probe layer info from checkpoints
    equiv_checkpoints = [r['model_history']['probe_checkpoints'] for r in equiv_results]
    probe_layer = equiv_checkpoints[0][0]['global_probe_accs'].get('probe_layer', -1) if equiv_checkpoints[0] else -1
    probe_dim = equiv_checkpoints[0][0]['global_probe_accs'].get('probe_layer_dim', 64) if equiv_checkpoints[0] else 64
    layer_str = f" [Layer {probe_layer}, {probe_dim}d]" if probe_layer != -1 else " [Final Layer]"
    
    fig.suptitle(f'Diagnostic Analysis: Comparison of Input Representations{layer_str}', fontsize=14, y=1.02)
    
    # -------------------------------------------------------------------------
    # Left plot: Bar chart of different representation accuracies
    # -------------------------------------------------------------------------
    ax = axes[0]
    
    representations = ['residuals', 'raw_features', 'concat_base_transformed', 
                       'residual_plus_base', 'normalized_residuals']
    short_names = ['Residuals', 'Raw E(g·x)', '[E(x),E(g·x)]', '[E(x),r(g·x)]', 'Norm. Res.']
    
    # Aggregate over runs
    equiv_accs = {rep: [] for rep in representations}
    inv_accs = {rep: [] for rep in representations}
    
    for result in equiv_results:
        if 'diagnostic_results' in result:
            for rep in representations:
                equiv_accs[rep].append(result['diagnostic_results'][rep]['accuracy'])
    
    for result in inv_results:
        if 'diagnostic_results' in result:
            for rep in representations:
                inv_accs[rep].append(result['diagnostic_results'][rep]['accuracy'])
    
    x = np.arange(len(representations))
    width = 0.35
    
    equiv_means = [np.mean(equiv_accs[rep]) if equiv_accs[rep] else 0 for rep in representations]
    equiv_stds = [np.std(equiv_accs[rep]) if equiv_accs[rep] else 0 for rep in representations]
    inv_means = [np.mean(inv_accs[rep]) if inv_accs[rep] else 0 for rep in representations]
    inv_stds = [np.std(inv_accs[rep]) if inv_accs[rep] else 0 for rep in representations]
    
    bars1 = ax.bar(x - width/2, equiv_means, width, yerr=equiv_stds, label='Equivariance',
                   color=equiv_color, capsize=5, alpha=0.8)
    bars2 = ax.bar(x + width/2, inv_means, width, yerr=inv_stds, label='Invariance', 
                   color=inv_color, capsize=5, alpha=0.8)
    
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Chance')
    ax.set_ylabel('Probe Accuracy (%)', fontsize=12)
    ax.set_title('Linear Probe Accuracy by Input Representation', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Middle plot: Variance decomposition
    # -------------------------------------------------------------------------
    ax = axes[1]
    
    # Aggregate variance analysis
    equiv_var_ratio = []
    inv_var_ratio = []
    
    for result in equiv_results:
        if 'diagnostic_results' in result and 'variance_analysis' in result['diagnostic_results']:
            equiv_var_ratio.append(result['diagnostic_results']['variance_analysis']['variance_ratio_rotation'] * 100)
    
    for result in inv_results:
        if 'diagnostic_results' in result and 'variance_analysis' in result['diagnostic_results']:
            inv_var_ratio.append(result['diagnostic_results']['variance_analysis']['variance_ratio_rotation'] * 100)
    
    x = [0, 1]
    means = [np.mean(equiv_var_ratio) if equiv_var_ratio else 0, 
             np.mean(inv_var_ratio) if inv_var_ratio else 0]
    stds = [np.std(equiv_var_ratio) if equiv_var_ratio else 0,
            np.std(inv_var_ratio) if inv_var_ratio else 0]
    
    bars = ax.bar(x, means, yerr=stds, color=[equiv_color, inv_color], capsize=10, alpha=0.8)
    ax.set_ylabel('Variance Explained by Rotation (%)', fontsize=12)
    ax.set_title('How Much Residual Variance is Due to Rotation?', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Equivariance', 'Invariance'])
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Right plot: Fisher's discriminant ratio
    # -------------------------------------------------------------------------
    ax = axes[2]
    
    equiv_fisher = []
    inv_fisher = []
    
    for result in equiv_results:
        if 'diagnostic_results' in result and 'fisher_ratio' in result['diagnostic_results']:
            equiv_fisher.append(result['diagnostic_results']['fisher_ratio'])
    
    for result in inv_results:
        if 'diagnostic_results' in result and 'fisher_ratio' in result['diagnostic_results']:
            inv_fisher.append(result['diagnostic_results']['fisher_ratio'])
    
    means = [np.mean(equiv_fisher) if equiv_fisher else 0, 
             np.mean(inv_fisher) if inv_fisher else 0]
    stds = [np.std(equiv_fisher) if equiv_fisher else 0,
            np.std(inv_fisher) if inv_fisher else 0]
    
    bars = ax.bar(x, means, yerr=stds, color=[equiv_color, inv_color], capsize=10, alpha=0.8)
    ax.set_ylabel("Fisher's Discriminant Ratio", fontsize=12)
    ax.set_title('Linear Separability of Rotations', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Equivariance', 'Invariance'])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'diagnostic_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPARISON SUMMARY")
    print("="*70)
    
    print("\nLinear Probe Accuracy by Input Representation:")
    print(f"{'Representation':<25} {'Equivariance':>15} {'Invariance':>15}")
    print("-"*60)
    for rep, short_name in zip(representations, short_names):
        eq_mean = np.mean(equiv_accs[rep]) if equiv_accs[rep] else 0
        eq_std = np.std(equiv_accs[rep]) if equiv_accs[rep] else 0
        iv_mean = np.mean(inv_accs[rep]) if inv_accs[rep] else 0
        iv_std = np.std(inv_accs[rep]) if inv_accs[rep] else 0
        print(f"{short_name:<25} {eq_mean:>10.2f}%±{eq_std:>4.2f} {iv_mean:>10.2f}%±{iv_std:>4.2f}")
    
    print(f"\nVariance Explained by Rotation:")
    if equiv_var_ratio:
        print(f"  Equivariance: {np.mean(equiv_var_ratio):.2f}%±{np.std(equiv_var_ratio):.2f}%")
    if inv_var_ratio:
        print(f"  Invariance:   {np.mean(inv_var_ratio):.2f}%±{np.std(inv_var_ratio):.2f}%")
    
    print(f"\nFisher's Discriminant Ratio (higher = better linear separability):")
    if equiv_fisher:
        print(f"  Equivariance: {np.mean(equiv_fisher):.4f}±{np.std(equiv_fisher):.4f}")
    if inv_fisher:
        print(f"  Invariance:   {np.mean(inv_fisher):.4f}±{np.std(inv_fisher):.4f}")
    
    # Direction consistency metrics
    equiv_dir_consistency = []
    inv_dir_consistency = []
    equiv_cross_sim = []
    inv_cross_sim = []
    
    for result in equiv_results:
        if 'diagnostic_results' in result and 'direction_consistency' in result['diagnostic_results']:
            equiv_dir_consistency.append(result['diagnostic_results']['direction_consistency']['average_cos_similarity'])
        if 'diagnostic_results' in result and 'cross_class_similarity' in result['diagnostic_results']:
            equiv_cross_sim.append(result['diagnostic_results']['cross_class_similarity']['average_off_diagonal'])
    
    for result in inv_results:
        if 'diagnostic_results' in result and 'direction_consistency' in result['diagnostic_results']:
            inv_dir_consistency.append(result['diagnostic_results']['direction_consistency']['average_cos_similarity'])
        if 'diagnostic_results' in result and 'cross_class_similarity' in result['diagnostic_results']:
            inv_cross_sim.append(result['diagnostic_results']['cross_class_similarity']['average_off_diagonal'])
    
    print(f"\nDirection Consistency (higher = more structured residuals):")
    if equiv_dir_consistency:
        print(f"  Equivariance: {np.mean(equiv_dir_consistency):.4f}±{np.std(equiv_dir_consistency):.4f}")
    if inv_dir_consistency:
        print(f"  Invariance:   {np.mean(inv_dir_consistency):.4f}±{np.std(inv_dir_consistency):.4f}")
    
    print(f"\nCross-class Direction Similarity (lower = better separation):")
    if equiv_cross_sim:
        print(f"  Equivariance: {np.mean(equiv_cross_sim):.4f}±{np.std(equiv_cross_sim):.4f}")
    if inv_cross_sim:
        print(f"  Invariance:   {np.mean(inv_cross_sim):.4f}±{np.std(inv_cross_sim):.4f}")
    
    # Per-sample separability (KEY METRIC)
    equiv_oracle = []
    inv_oracle = []
    
    for result in equiv_results:
        if 'diagnostic_results' in result and 'per_sample_analysis' in result['diagnostic_results']:
            equiv_oracle.append(result['diagnostic_results']['per_sample_analysis']['oracle_accuracy'])
    
    for result in inv_results:
        if 'diagnostic_results' in result and 'per_sample_analysis' in result['diagnostic_results']:
            inv_oracle.append(result['diagnostic_results']['per_sample_analysis']['oracle_accuracy'])
    
    print(f"\n*** KEY METRIC: Per-Sample Oracle Accuracy ***")
    print(f"  (Can a separate linear classifier per sample distinguish rotations?)")
    if equiv_oracle:
        print(f"  Equivariance: {np.mean(equiv_oracle):.1f}%±{np.std(equiv_oracle):.1f}%")
    if inv_oracle:
        print(f"  Invariance:   {np.mean(inv_oracle):.1f}%±{np.std(inv_oracle):.1f}%")
    
    # Interpretation
    print("\n" + "-"*70)
    print("INTERPRETATION:")
    
    if equiv_oracle and inv_oracle:
        eq_oracle_mean = np.mean(equiv_oracle)
        eq_probe_mean = np.mean(equiv_accs['residuals']) if equiv_accs['residuals'] else 0
        print(f"\n  Per-Sample Separability Analysis:")
        print(f"    Equivariance oracle: {eq_oracle_mean:.1f}% vs shared probe: {eq_probe_mean:.1f}%")
        print(f"    Gap: {eq_oracle_mean - eq_probe_mean:.1f}%")
        if eq_oracle_mean > 95 and eq_probe_mean < 80:
            print(f"\n  ✓ CONFIRMED: For equivariant features:")
            print(f"    - Rotations ARE perfectly separable within each sample")
            print(f"    - The ~{100 - eq_probe_mean:.0f}% error is due to cross-sample interference")
            print(f"    - A single linear probe W cannot handle varying E(x)")
            print(f"    - This is the fundamental limitation of r(g·x) = M(g)·E(x)")
    
    if equiv_dir_consistency and inv_dir_consistency:
        if np.mean(equiv_dir_consistency) > np.mean(inv_dir_consistency):
            print("  ✓ Equivariance has MORE CONSISTENT residual directions")
            print("    → Supports theory: r(g·x) = M(g)·E(x) with deterministic M(g)")
        else:
            print("  ✗ Invariance has more consistent directions (unexpected)")
    
    if equiv_cross_sim and inv_cross_sim:
        if np.mean(equiv_cross_sim) < np.mean(inv_cross_sim):
            print("  ✓ Equivariance has MORE DISTINCT rotation directions")
        else:
            print("  ~ Invariance has more distinct directions")
    
    # -------------------------------------------------------------------------
    # Global Probe Approaches Comparison
    # -------------------------------------------------------------------------
    global_approaches = ['baseline_residuals', 'normalize_by_Ex', 'unit_direction', 
                         'pairwise_cosines', 'projection_scores', 'whitened', 
                         'theoretical_matching', 'ratio_features']
    approach_names = ['Baseline', 'Norm by ||E(x)||', 'Unit direction', 
                      'Pairwise cosines', 'Proj onto M(g)', 'Whitened',
                      'Theoretical M(g)^T', 'Ratio r(g)/r(0)']
    
    equiv_global = {a: [] for a in global_approaches}
    inv_global = {a: [] for a in global_approaches}
    
    for result in equiv_results:
        if 'global_probe_results' in result:
            for a in global_approaches:
                if a in result['global_probe_results']:
                    equiv_global[a].append(result['global_probe_results'][a])
    
    for result in inv_results:
        if 'global_probe_results' in result:
            for a in global_approaches:
                if a in result['global_probe_results']:
                    inv_global[a].append(result['global_probe_results'][a])
    
    if any(equiv_global[a] for a in global_approaches):
        print(f"\n" + "="*70)
        print("GLOBAL PROBE APPROACHES: Eliminating E(x) Dependence")
        print("="*70)
        print(f"\n{'Approach':<25} {'Equivariance':>15} {'Invariance':>15}")
        print("-"*60)
        
        for a, name in zip(global_approaches, approach_names):
            eq_mean = np.mean(equiv_global[a]) if equiv_global[a] else 0
            eq_std = np.std(equiv_global[a]) if equiv_global[a] else 0
            iv_mean = np.mean(inv_global[a]) if inv_global[a] else 0
            iv_std = np.std(inv_global[a]) if inv_global[a] else 0
            print(f"{name:<25} {eq_mean:>10.2f}%±{eq_std:>4.2f} {iv_mean:>10.2f}%±{iv_std:>4.2f}")
        
        # Find best approach
        equiv_best = max(global_approaches, key=lambda a: np.mean(equiv_global[a]) if equiv_global[a] else 0)
        inv_best = max(global_approaches, key=lambda a: np.mean(inv_global[a]) if inv_global[a] else 0)
        
        print(f"\n  Best for Equivariance: {equiv_best} ({np.mean(equiv_global[equiv_best]):.2f}%)")
        print(f"  Best for Invariance: {inv_best} ({np.mean(inv_global[inv_best]):.2f}%)")


def plot_subspace_decomposition(equiv_results: List[Dict], inv_results: List[Dict], output_dir: str = '.'):
    """
    Plot the invariant vs equivariant subspace decomposition during training.
    Shows how ||E_inv|| and ||E_eq|| evolve, and how this relates to probe accuracy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    equiv_color = '#2ecc71'
    inv_color = '#e74c3c'
    alpha_fill = 0.15
    
    # Extract checkpoint data
    equiv_checkpoints = [r['model_history']['probe_checkpoints'] for r in equiv_results]
    inv_checkpoints = [r['model_history']['probe_checkpoints'] for r in inv_results]
    
    if 'global_probe_accs' not in equiv_checkpoints[0][0]:
        print("Warning: No global_probe_accs data. Skipping subspace decomposition plot.")
        plt.close()
        return
    
    # Get probe layer info
    probe_layer = equiv_checkpoints[0][0]['global_probe_accs'].get('probe_layer', -1)
    probe_dim = equiv_checkpoints[0][0]['global_probe_accs'].get('probe_layer_dim', 64)
    layer_str = f"Layer {probe_layer} ({probe_dim}d)" if probe_layer != -1 else f"Final Layer ({probe_dim}d)"
    
    fig.suptitle(f'Subspace Decomposition During Training - {layer_str}', fontsize=14, y=1.02)
    
    # Get step/epoch data
    equiv_steps = np.array([[cp.get('step', i) for i, cp in enumerate(run)] for run in equiv_checkpoints])
    inv_steps = np.array([[cp.get('step', i) for i, cp in enumerate(run)] for run in inv_checkpoints])
    equiv_step_mean = np.mean(equiv_steps, axis=0)
    inv_step_mean = np.mean(inv_steps, axis=0)
    
    # Extract E_inv and E_eq norms
    equiv_E_inv = np.array([[cp['global_probe_accs'].get('avg_E_inv_norm', 0) for cp in run] for run in equiv_checkpoints])
    equiv_E_eq = np.array([[cp['global_probe_accs'].get('avg_E_equiv_norm', 0) for cp in run] for run in equiv_checkpoints])
    inv_E_inv = np.array([[cp['global_probe_accs'].get('avg_E_inv_norm', 0) for cp in run] for run in inv_checkpoints])
    inv_E_eq = np.array([[cp['global_probe_accs'].get('avg_E_equiv_norm', 0) for cp in run] for run in inv_checkpoints])
    
    # Extract probe accuracies
    equiv_E_eq_acc = np.array([[cp['global_probe_accs'].get('E_eq_unit', 25) for cp in run] for run in equiv_checkpoints])
    equiv_baseline_acc = np.array([[cp['global_probe_accs'].get('baseline_residuals', 25) for cp in run] for run in equiv_checkpoints])
    inv_E_eq_acc = np.array([[cp['global_probe_accs'].get('E_eq_unit', 25) for cp in run] for run in inv_checkpoints])
    inv_baseline_acc = np.array([[cp['global_probe_accs'].get('baseline_residuals', 25) for cp in run] for run in inv_checkpoints])
    
    # -------------------------------------------------------------------------
    # Top-left: ||E_inv|| and ||E_eq|| vs step for Equivariance
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(equiv_step_mean, np.mean(equiv_E_inv, axis=0), 'o-', color='blue', 
            linewidth=2, markersize=5, label='||E_inv|| (invariant)')
    ax.fill_between(equiv_step_mean, 
                    np.mean(equiv_E_inv, axis=0) - np.std(equiv_E_inv, axis=0),
                    np.mean(equiv_E_inv, axis=0) + np.std(equiv_E_inv, axis=0),
                    color='blue', alpha=alpha_fill)
    ax.plot(equiv_step_mean, np.mean(equiv_E_eq, axis=0), 's-', color='orange', 
            linewidth=2, markersize=5, label='||E_eq|| (equivariant)')
    ax.fill_between(equiv_step_mean, 
                    np.mean(equiv_E_eq, axis=0) - np.std(equiv_E_eq, axis=0),
                    np.mean(equiv_E_eq, axis=0) + np.std(equiv_E_eq, axis=0),
                    color='orange', alpha=alpha_fill)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Subspace Norm', fontsize=11)
    ax.set_title('EQUIVARIANCE: Subspace Norms', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Top-right: ||E_inv|| and ||E_eq|| vs step for Invariance
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(inv_step_mean, np.mean(inv_E_inv, axis=0), 'o-', color='blue', 
            linewidth=2, markersize=5, label='||E_inv|| (invariant)')
    ax.fill_between(inv_step_mean, 
                    np.mean(inv_E_inv, axis=0) - np.std(inv_E_inv, axis=0),
                    np.mean(inv_E_inv, axis=0) + np.std(inv_E_inv, axis=0),
                    color='blue', alpha=alpha_fill)
    ax.plot(inv_step_mean, np.mean(inv_E_eq, axis=0), 's-', color='orange', 
            linewidth=2, markersize=5, label='||E_eq|| (equivariant)')
    ax.fill_between(inv_step_mean, 
                    np.mean(inv_E_eq, axis=0) - np.std(inv_E_eq, axis=0),
                    np.mean(inv_E_eq, axis=0) + np.std(inv_E_eq, axis=0),
                    color='orange', alpha=alpha_fill)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Subspace Norm', fontsize=11)
    ax.set_title('INVARIANCE: Subspace Norms', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Bottom-left: E_eq probe accuracy vs ||E_eq||
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ax.scatter(np.mean(equiv_E_eq, axis=0), np.mean(equiv_E_eq_acc, axis=0), 
               c=equiv_step_mean, cmap='Greens', s=80, edgecolor='black', linewidth=1, label='Equiv')
    ax.scatter(np.mean(inv_E_eq, axis=0), np.mean(inv_E_eq_acc, axis=0), 
               c=inv_step_mean, cmap='Reds', s=80, edgecolor='black', linewidth=1, marker='s', label='Inv')
    # Connect points with lines
    ax.plot(np.mean(equiv_E_eq, axis=0), np.mean(equiv_E_eq_acc, axis=0), '-', color=equiv_color, alpha=0.5)
    ax.plot(np.mean(inv_E_eq, axis=0), np.mean(inv_E_eq_acc, axis=0), '-', color=inv_color, alpha=0.5)
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Chance')
    ax.set_xlabel('||E_eq|| (equivariant subspace norm)', fontsize=11)
    ax.set_ylabel('E_eq Probe Accuracy (%)', fontsize=11)
    ax.set_title('Probe Accuracy vs Equivariant Content', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Bottom-right: Comparison of E_eq vs Baseline probe
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(equiv_step_mean, np.mean(equiv_E_eq_acc, axis=0), 'o-', color=equiv_color, 
            linewidth=2, markersize=5, label='Equiv: E_eq probe')
    ax.plot(equiv_step_mean, np.mean(equiv_baseline_acc, axis=0), 's--', color=equiv_color, 
            linewidth=2, markersize=5, alpha=0.6, label='Equiv: Baseline')
    ax.plot(inv_step_mean, np.mean(inv_E_eq_acc, axis=0), 'o-', color=inv_color, 
            linewidth=2, markersize=5, label='Inv: E_eq probe')
    ax.plot(inv_step_mean, np.mean(inv_baseline_acc, axis=0), 's--', color=inv_color, 
            linewidth=2, markersize=5, alpha=0.6, label='Inv: Baseline')
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Probe Accuracy (%)', fontsize=11)
    ax.set_title('E_eq Probe vs Baseline Probe', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add annotation about key finding
    final_equiv_E_eq = np.mean(equiv_E_eq_acc[:, -1])
    final_equiv_baseline = np.mean(equiv_baseline_acc[:, -1])
    final_inv_E_eq = np.mean(inv_E_eq_acc[:, -1])
    ax.annotate(f'Final E_eq Acc:\nEquiv={final_equiv_E_eq:.1f}%\nInv={final_inv_E_eq:.1f}%', 
               xy=(0.98, 0.05), xycoords='axes fraction',
               fontsize=9, ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'subspace_decomposition.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print(f"SUBSPACE DECOMPOSITION SUMMARY - {layer_str}")
    print("="*80)
    print(f"\n{'Metric':<30} {'Equiv Init':<12} {'Equiv Final':<12} {'Inv Init':<12} {'Inv Final':<12}")
    print("-"*80)
    print(f"{'||E_inv|| (invariant)':<30} {np.mean(equiv_E_inv[:, 0]):<12.2f} {np.mean(equiv_E_inv[:, -1]):<12.2f} "
          f"{np.mean(inv_E_inv[:, 0]):<12.2f} {np.mean(inv_E_inv[:, -1]):<12.2f}")
    print(f"{'||E_eq|| (equivariant)':<30} {np.mean(equiv_E_eq[:, 0]):<12.2f} {np.mean(equiv_E_eq[:, -1]):<12.2f} "
          f"{np.mean(inv_E_eq[:, 0]):<12.2f} {np.mean(inv_E_eq[:, -1]):<12.2f}")
    print(f"{'E_eq Probe Acc (%)':<30} {np.mean(equiv_E_eq_acc[:, 0]):<12.1f} {np.mean(equiv_E_eq_acc[:, -1]):<12.1f} "
          f"{np.mean(inv_E_eq_acc[:, 0]):<12.1f} {np.mean(inv_E_eq_acc[:, -1]):<12.1f}")
    print(f"{'Baseline Probe Acc (%)':<30} {np.mean(equiv_baseline_acc[:, 0]):<12.1f} {np.mean(equiv_baseline_acc[:, -1]):<12.1f} "
          f"{np.mean(inv_baseline_acc[:, 0]):<12.1f} {np.mean(inv_baseline_acc[:, -1]):<12.1f}")


def plot_invariance_sigma_analysis(inv_results: List[Dict], output_dir: str = '.'):
    """
    Generate two plots analyzing invariance model behavior:
    
    PLOT 1: Baseline probe accuracy vs training step (invariance only)
            with text annotations showing avg invariance loss and avg σ_min
    
    PLOT 2: Avg invariance loss and avg σ_min vs training step
            (both with fill_between for multiple runs)
    
    σ_min is the smallest singular value of the orbit matrix [E(x), E(90°·x), E(180°·x), E(270°·x)]
    """
    if inv_results is None:
        print("Warning: No invariance results provided. Skipping sigma analysis plots.")
        return
    
    # Extract checkpoint data
    inv_checkpoints = [r['model_history']['probe_checkpoints'] for r in inv_results]
    
    if 'global_probe_accs' not in inv_checkpoints[0][0]:
        print("Warning: No global_probe_accs data. Skipping sigma analysis plots.")
        return
    
    # Get probe layer info
    probe_layer = inv_checkpoints[0][0]['global_probe_accs'].get('probe_layer', -1)
    probe_dim = inv_checkpoints[0][0]['global_probe_accs'].get('probe_layer_dim', 64)
    layer_str = f"Layer {probe_layer} ({probe_dim}d)" if probe_layer != -1 else f"Final Layer ({probe_dim}d)"
    
    n_runs = len(inv_results)
    n_checkpoints = len(inv_checkpoints[0])
    
    # Extract data arrays
    inv_steps = np.array([[cp.get('step', i) for i, cp in enumerate(run)] for run in inv_checkpoints])
    inv_losses = np.array([[cp['val_equiv_loss'] for cp in run] for run in inv_checkpoints])
    inv_baseline_acc = np.array([[cp['global_probe_accs'].get('baseline_residuals', 25) for cp in run] for run in inv_checkpoints])
    inv_sigma_min = np.array([[cp['global_probe_accs'].get('avg_sigma_min', 0) for cp in run] for run in inv_checkpoints])
    
    # Compute means and stds
    step_mean = np.mean(inv_steps, axis=0)
    loss_mean = np.mean(inv_losses, axis=0)
    loss_std = np.std(inv_losses, axis=0)
    acc_mean = np.mean(inv_baseline_acc, axis=0)
    acc_std = np.std(inv_baseline_acc, axis=0)
    sigma_mean = np.mean(inv_sigma_min, axis=0)
    sigma_std = np.std(inv_sigma_min, axis=0)
    
    inv_color = '#e74c3c'
    alpha_fill = 0.2
    
    # =========================================================================
    # PLOT 1: Baseline probe accuracy with annotations
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot baseline accuracy with fill_between
    ax.plot(step_mean, acc_mean, 'o-', color=inv_color, linewidth=2, markersize=8, label='Baseline Probe Accuracy')
    ax.fill_between(step_mean, acc_mean - acc_std, acc_mean + acc_std, color=inv_color, alpha=alpha_fill)
    
    # Add chance line
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Chance (25%)')
    
    # Add text annotations at each checkpoint
    for i in range(n_checkpoints):
        # Offset annotations to avoid overlap
        y_offset = 3 if i % 2 == 0 else -5
        annotation_text = f"Loss={loss_mean[i]:.3f}±{loss_std[i]:.3f}\nσ_min={sigma_mean[i]:.2f}±{sigma_std[i]:.2f}"
        ax.annotate(annotation_text, 
                   xy=(step_mean[i], acc_mean[i]),
                   xytext=(0, y_offset), textcoords='offset points',
                   fontsize=7, ha='center', va='bottom' if y_offset > 0 else 'top',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Baseline Probe Accuracy (%)', fontsize=12)
    ax.set_title(f'Invariance Model: Probe Accuracy vs Training Step\n{layer_str} | {n_runs} runs', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits with some padding
    ax.set_ylim([max(20, acc_mean.min() - 10), min(100, acc_mean.max() + 15)])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'invariance_probe_with_sigma.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # =========================================================================
    # PLOT 2: Invariance loss and σ_min vs step (dual y-axis)
    # =========================================================================
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Left y-axis: Invariance loss
    color_loss = '#3498db'  # blue
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Invariance Loss (MSE)', fontsize=12, color=color_loss)
    line1 = ax1.plot(step_mean, loss_mean, 'o-', color=color_loss, linewidth=2, markersize=8, label='Inv Loss')
    ax1.fill_between(step_mean, loss_mean - loss_std, loss_mean + loss_std, color=color_loss, alpha=alpha_fill)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.set_ylim([0, max(loss_mean.max() * 1.2, 0.1)])
    
    # Right y-axis: σ_min
    ax2 = ax1.twinx()
    color_sigma = '#9b59b6'  # purple
    ax2.set_ylabel('Average σ_min (smallest singular value)', fontsize=12, color=color_sigma)
    line2 = ax2.plot(step_mean, sigma_mean, 's-', color=color_sigma, linewidth=2, markersize=8, label='σ_min')
    ax2.fill_between(step_mean, sigma_mean - sigma_std, sigma_mean + sigma_std, color=color_sigma, alpha=alpha_fill)
    ax2.tick_params(axis='y', labelcolor=color_sigma)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    ax1.set_title(f'Invariance Model: Loss and σ_min During Training\n{layer_str} | {n_runs} runs', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'invariance_loss_sigma_min.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*90)
    print(f"INVARIANCE σ_min ANALYSIS - {layer_str} | {n_runs} runs")
    print("="*90)
    print(f"\n{'Step':<10} {'Inv Loss':<20} {'Baseline Acc':<20} {'σ_min':<20}")
    print("-"*70)
    for i in range(n_checkpoints):
        print(f"{int(step_mean[i]):<10} {loss_mean[i]:.4f}±{loss_std[i]:.4f}      "
              f"{acc_mean[i]:.1f}%±{acc_std[i]:.1f}%        "
              f"{sigma_mean[i]:.3f}±{sigma_std[i]:.3f}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='MNIST Linear Probe Experiment')
    parser.add_argument('--mode', type=str, default='both', choices=['both', 'equivariance', 'invariance'],
                       help='Which mode(s) to run')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs per mode')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for main model')
    parser.add_argument('--probe_epochs', type=int, default=30, help='Number of epochs for final probe')
    parser.add_argument('--probe_every', type=int, default=5, help='Evaluate probe every N epochs during training')
    parser.add_argument('--step_probe_interval', type=int, default=50, help='Evaluate probe every N steps during epoch 1')
    parser.add_argument('--quick_probe_epochs', type=int, default=30, help='Number of epochs for checkpoint probes')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension (must be divisible by 4)')
    parser.add_argument('--lambda_equiv', type=float, default=0.1, help='Weight for equiv/inv loss')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for plots')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--use_layernorm', action='store_true', help='Use LayerNorm instead of BatchNorm (keeps feature magnitudes bounded)')
    parser.add_argument('--probe_layer', type=int, default=-1, 
                       help='Which layer to probe: -1=final (64), 0=first (512), 1=second (256), 2=third (128), 3=fourth (64)')
    args = parser.parse_args()
    
    # Validate
    assert args.latent_dim % 4 == 0, "latent_dim must be divisible by 4"
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config['epochs'] = args.epochs
    config['probe_epochs'] = args.probe_epochs
    config['probe_every'] = args.probe_every
    config['step_probe_interval'] = args.step_probe_interval
    config['quick_probe_epochs'] = args.quick_probe_epochs
    config['batch_size'] = args.batch_size
    config['latent_dim'] = args.latent_dim
    config['lambda_equiv'] = args.lambda_equiv
    config['use_layernorm'] = args.use_layernorm
    config['probe_layer'] = args.probe_layer
    
    print("\n" + "="*70)
    print("MNIST Linear Probe Experiment")
    print("="*70)
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"\nMode: {args.mode}")
    print(f"Num runs: {args.num_runs}")
    
    # Explain probe layer choice
    hidden_dims = [512, 256, 128, 64]
    if args.probe_layer == -1:
        layer_desc = f"final (latent_dim={args.latent_dim})"
    elif 0 <= args.probe_layer < len(hidden_dims):
        layer_desc = f"layer {args.probe_layer} (dim={hidden_dims[args.probe_layer]})"
    else:
        layer_desc = f"layer {args.probe_layer} (clamped)"
    print(f"Probing {layer_desc}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    equiv_results = None
    inv_results = None
    
    if args.mode in ['both', 'equivariance']:
        equiv_results = run_multiple_experiments('equivariance', args.num_runs, config, device)
        with open(os.path.join(args.output_dir, 'equiv_probe_results.pkl'), 'wb') as f:
            pickle.dump(equiv_results, f)
    
    if args.mode in ['both', 'invariance']:
        inv_results = run_multiple_experiments('invariance', args.num_runs, config, device)
        with open(os.path.join(args.output_dir, 'inv_probe_results.pkl'), 'wb') as f:
            pickle.dump(inv_results, f)
    
    # Generate comparison plots if both modes were run
    if args.mode == 'both':
        plot_comparison(equiv_results, inv_results, args.output_dir)
        plot_equiv_loss_vs_probe_acc(equiv_results, inv_results, args.output_dir)
        plot_global_probe_trajectories(equiv_results, inv_results, args.output_dir)
        plot_global_probe_by_step(equiv_results, inv_results, args.output_dir)
        plot_subspace_decomposition(equiv_results, inv_results, args.output_dir)
        plot_diagnostic_comparison(equiv_results, inv_results, args.output_dir)
    
    # Generate invariance-specific sigma analysis plots
    if inv_results is not None:
        plot_invariance_sigma_analysis(inv_results, args.output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
