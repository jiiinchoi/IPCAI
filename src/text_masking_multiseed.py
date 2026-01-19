"""
Multi-seed Text Masking Experiment
3개 seed로 평균±표준편차 계산 (N=8)
통계적 안정성 검증
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import BraTSDataset
from torch.utils.data import DataLoader
from src.models import ImageModel, TextModel
from utils.metrics import (
    set_seed,
    compute_metrics,
    save_json,
    torch_load_weights,
)


class MaskedTextDataset(BraTSDataset):
    """텍스트 토큰을 [MASK]로 가리는 Dataset"""
    def __init__(self, *args, mask_ratio=0.0, mask_token_id=103, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio
        self.mask_token_id = mask_token_id
        self.seed = seed
        self.rng = random.Random(seed)  # Instance-specific RNG
    
    def __getitem__(self, idx):
        if self.mode == 'image':
            return super().__getitem__(idx)
        elif self.mode == 'text':
            tokens, label = super().__getitem__(idx)
        else:  # multimodal
            images, tokens, label = super().__getitem__(idx)
        
        # Masking 적용
        if self.mask_ratio > 0:
            input_ids = tokens['input_ids'].clone()
            attention_mask = tokens['attention_mask']
            
            # Maskable positions
            maskable_positions = []
            for i, (token_id, attn) in enumerate(zip(input_ids, attention_mask)):
                if attn == 1 and token_id not in [0, 101, 102]:
                    maskable_positions.append(i)
            
            # Random masking (deterministic per seed+idx)
            n_mask = int(len(maskable_positions) * self.mask_ratio)
            if n_mask > 0:
                # Use seed + idx for determinism
                local_rng = random.Random(self.seed + idx)
                mask_positions = local_rng.sample(maskable_positions, n_mask)
                for pos in mask_positions:
                    input_ids[pos] = self.mask_token_id
            
            tokens['input_ids'] = input_ids
        
        if self.mode == 'text':
            return tokens, label
        else:
            return images, tokens, label


def get_masked_dataloader(csv_path, mask_ratio, seed, batch_size=8, **kwargs):
    """Masked text dataloader"""
    dataset = MaskedTextDataset(
        csv_path=csv_path,
        mode='multimodal',
        mask_ratio=mask_ratio,
        seed=seed,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


@torch.no_grad()
def extract_logits(img_model, txt_model, loader, device):
    """LR Fusion용 logits 추출"""
    img_model.eval()
    txt_model.eval()
    
    ys, img_logits_list, txt_logits_list = [], [], []
    
    for images, tokens, labels in loader:
        images = images.to(device)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        
        img_logits = img_model(images)
        txt_logits = txt_model(input_ids, attention_mask)
        
        img_logit_pos = (img_logits[:, 1] - img_logits[:, 0]).cpu().numpy()
        txt_logit_pos = (txt_logits[:, 1] - txt_logits[:, 0]).cpu().numpy()
        
        ys.append(labels.numpy())
        img_logits_list.append(img_logit_pos)
        txt_logits_list.append(txt_logit_pos)
    
    y = np.concatenate(ys).astype(int)
    X = np.stack([
        np.concatenate(img_logits_list),
        np.concatenate(txt_logits_list)
    ], axis=1)
    
    return y, X


def run_single_seed_experiment(seed, args, device):
    """단일 seed 실험"""
    print(f"\n{'='*70}")
    print(f"  Seed {seed}")
    print(f"{'='*70}")
    
    set_seed(seed)
    
    # Load models
    img_ckpt = ROOT / f"runs/image_{args.modality}/models/best_image.pt"
    txt_ckpt = ROOT / "runs/text/models/best_text.pt"
    
    img_model = ImageModel(num_classes=2, pretrained=True).to(device)
    txt_model = TextModel(
        num_classes=2,
        model_name=args.tokenizer,
        freeze_bert=True
    ).to(device)
    
    img_model.load_state_dict(torch_load_weights(img_ckpt, device))
    txt_model.load_state_dict(torch_load_weights(txt_ckpt, device))
    
    # Train LR on original text
    train_loader = get_masked_dataloader(
        csv_path=str(ROOT / args.train_csv),
        mask_ratio=0.0,
        seed=seed,
        batch_size=args.batch_size,
        modality=args.modality,
        n_slices=args.n_slices,
        image_size=(args.img_size, args.img_size),
        tokenizer_name=args.tokenizer,
        max_text_length=args.max_text_len,
    )
    
    y_train, X_train = extract_logits(img_model, txt_model, train_loader, device)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)
    lr.fit(X_train_s, y_train)
    
    # Evaluate with different masking ratios
    results = {}
    
    for mask_ratio in args.mask_ratios:
        test_loader = get_masked_dataloader(
            csv_path=str(ROOT / args.test_csv),
            mask_ratio=mask_ratio,
            seed=seed,
            batch_size=args.batch_size,
            modality=args.modality,
            n_slices=args.n_slices,
            image_size=(args.img_size, args.img_size),
            tokenizer_name=args.tokenizer,
            max_text_length=args.max_text_len,
        )
        
        y_test, X_test = extract_logits(img_model, txt_model, test_loader, device)
        X_test_s = scaler.transform(X_test)
        p_test = lr.predict_proba(X_test_s)[:, 1]
        
        metrics = compute_metrics(y_test, p_test)
        
        results[mask_ratio] = {
            'roc_auc': metrics['roc_auc'],
            'pr_auc': metrics['pr_auc'],
        }
        
        print(f"  Mask {int(mask_ratio*100):2d}%: ROC={metrics['roc_auc']:.4f}, PR={metrics['pr_auc']:.4f}")
    
    return results


def run_multi_seed_experiment(args):
    """Multi-seed 실험"""
    device = get_device()
    
    print("\n" + "="*70)
    print("  Multi-Seed Text Masking Experiment")
    print("="*70)
    print(f"Seeds: {args.seeds}")
    print(f"Masking ratios: {args.mask_ratios}")
    print(f"N_slices: {args.n_slices}")
    print()
    
    # Run for each seed
    all_results = {}
    
    for seed in args.seeds:
        seed_results = run_single_seed_experiment(seed, args, device)
        
        for mask_ratio, metrics in seed_results.items():
            if mask_ratio not in all_results:
                all_results[mask_ratio] = {'roc_auc': [], 'pr_auc': []}
            all_results[mask_ratio]['roc_auc'].append(metrics['roc_auc'])
            all_results[mask_ratio]['pr_auc'].append(metrics['pr_auc'])
    
    # Compute statistics
    print("\n" + "="*70)
    print("SUMMARY: Multi-Seed Results (Mean ± Std)")
    print("="*70)
    print(f"{'Mask %':<12} {'ROC-AUC':<25} {'PR-AUC':<25}")
    print("-"*70)
    
    stats = {}
    baseline_roc_mean = None
    
    for mask_ratio in sorted(all_results.keys()):
        roc_aucs = all_results[mask_ratio]['roc_auc']
        pr_aucs = all_results[mask_ratio]['pr_auc']
        
        roc_mean = np.mean(roc_aucs)
        roc_std = np.std(roc_aucs, ddof=1)  # Sample std
        pr_mean = np.mean(pr_aucs)
        pr_std = np.std(pr_aucs, ddof=1)
        
        if mask_ratio == 0.0:
            baseline_roc_mean = roc_mean
        
        delta_roc = roc_mean - baseline_roc_mean if baseline_roc_mean else 0.0
        
        stats[mask_ratio] = {
            'roc_mean': roc_mean,
            'roc_std': roc_std,
            'pr_mean': pr_mean,
            'pr_std': pr_std,
            'delta_roc': delta_roc,
        }
        
        pct = int(mask_ratio * 100)
        print(f"{pct}%{'':<9} {roc_mean:.4f} ± {roc_std:.4f}      {pr_mean:.4f} ± {pr_std:.4f}")
    
    print("="*70)
    
    # Delta analysis
    print("\n" + "="*70)
    print("DELTA from 0% (Mean ± Std)")
    print("="*70)
    print(f"{'Mask %':<12} {'Δ ROC-AUC':<20} {'Significant?':<15}")
    print("-"*70)
    
    for mask_ratio in sorted(all_results.keys()):
        if mask_ratio == 0.0:
            continue
        
        delta = stats[mask_ratio]['delta_roc']
        
        # Compute delta std (propagation of error)
        baseline_std = stats[0.0]['roc_std']
        current_std = stats[mask_ratio]['roc_std']
        delta_std = np.sqrt(baseline_std**2 + current_std**2)
        
        # Simple significance test: |delta| > 2*std
        is_significant = abs(delta) > 2 * delta_std
        
        pct = int(mask_ratio * 100)
        sig_marker = "✓" if is_significant else "✗"
        
        print(f"{pct}%{'':<9} {delta:+.4f} ± {delta_std:.4f}    {sig_marker} ({abs(delta)/delta_std:.2f}σ)")
    
    print("="*70)
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    delta_60 = stats[0.6]['delta_roc']
    delta_60_std = np.sqrt(stats[0.0]['roc_std']**2 + stats[0.6]['roc_std']**2)
    
    print(f"60% Masking Effect:")
    print(f"  Mean delta: {delta_60:+.4f}")
    print(f"  Std error:  {delta_60_std:.4f}")
    print(f"  Magnitude:  {abs(delta_60)/delta_60_std:.2f}σ")
    print()
    
    if delta_60 > 0 and abs(delta_60) > 2 * delta_60_std:
        print("✅ STATISTICALLY SIGNIFICANT IMPROVEMENT")
        print("   → 60% masking consistently improves performance across seeds")
        print("   → Text provides REGULARIZATION (removing noise helps)")
    elif abs(delta_60) < delta_60_std:
        print("✅ STABLE PERFORMANCE (no significant change)")
        print("   → Text content doesn't matter (supports regularization)")
    elif delta_60 < 0 and abs(delta_60) > 2 * delta_60_std:
        print("⚠️  STATISTICALLY SIGNIFICANT DEGRADATION")
        print("   → Text contains task-relevant information")
    else:
        print("⚠️  INCONCLUSIVE (high variance)")
        print("   → Need more seeds or larger test set")
    
    print("="*70)
    
    # Save results
    output_dir = ROOT / "results/robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "text_masking_multiseed.json"
    
    final_results = {
        'experiment': 'text_masking_multiseed',
        'n_slices': args.n_slices,
        'seeds': args.seeds,
        'statistics': {
            f"mask_{int(ratio*100)}pct": {
                'roc_mean': float(stats[ratio]['roc_mean']),
                'roc_std': float(stats[ratio]['roc_std']),
                'pr_mean': float(stats[ratio]['pr_mean']),
                'pr_std': float(stats[ratio]['pr_std']),
                'delta_roc': float(stats[ratio]['delta_roc']),
            }
            for ratio in sorted(stats.keys())
        },
        'raw_results': {
            f"mask_{int(ratio*100)}pct": {
                'roc_auc': [float(x) for x in all_results[ratio]['roc_auc']],
                'pr_auc': [float(x) for x in all_results[ratio]['pr_auc']],
            }
            for ratio in sorted(all_results.keys())
        }
    }
    
    save_json(output_path, final_results)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("\n✅ Experiment completed!")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Multi-Seed Text Masking Experiment')
    
    # Experiment settings
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                       help='Random seeds')
    parser.add_argument('--mask_ratios', type=float, nargs='+', default=[0.0, 0.2, 0.4, 0.6],
                       help='Text masking ratios')
    parser.add_argument('--n_slices', type=int, default=8,
                       help='N_slices (moderate degradation)')
    parser.add_argument('--modality', type=str, default='t1ce',
                       choices=['flair', 't1', 't1ce', 't2'])
    
    # Data paths
    parser.add_argument('--train_csv', type=str, default='data/splits/train.csv')
    parser.add_argument('--test_csv', type=str, default='data/splits/test.csv')
    
    # Model settings
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--tokenizer', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--max_text_len', type=int, default=512)
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_multi_seed_experiment(args)


if __name__ == "__main__":
    main()