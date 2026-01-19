"""
Text Masking Experiment
텍스트 토큰 masking으로 "정보 vs 안정화" 역할 구분
N=8 (moderate degradation) 조건에서 실험
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
    """
    텍스트 토큰을 [MASK]로 가리는 Dataset
    """
    def __init__(self, *args, mask_ratio=0.0, mask_token_id=103, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio
        self.mask_token_id = mask_token_id  # BERT [MASK] token ID
    
    def __getitem__(self, idx):
        # 부모 클래스에서 데이터 가져오기
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
            
            # [CLS], [SEP], [PAD] 제외하고 masking
            # [CLS] = 101, [SEP] = 102, [PAD] = 0
            maskable_positions = []
            for i, (token_id, attn) in enumerate(zip(input_ids, attention_mask)):
                if attn == 1 and token_id not in [0, 101, 102]:
                    maskable_positions.append(i)
            
            # 랜덤하게 mask_ratio만큼 선택
            n_mask = int(len(maskable_positions) * self.mask_ratio)
            if n_mask > 0:
                mask_positions = random.sample(maskable_positions, n_mask)
                for pos in mask_positions:
                    input_ids[pos] = self.mask_token_id
            
            tokens['input_ids'] = input_ids
        
        # Return
        if self.mode == 'text':
            return tokens, label
        else:  # multimodal
            return images, tokens, label


def get_masked_dataloader(
    csv_path: str,
    mask_ratio: float,
    batch_size: int = 8,
    **kwargs
):
    """Masked text dataloader"""
    dataset = MaskedTextDataset(
        csv_path=csv_path,
        mode='multimodal',
        mask_ratio=mask_ratio,
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
    print(f"[Device] {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
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


def run_text_masking_experiment(args):
    """
    Main experiment
    """
    set_seed(args.seed)
    device = get_device()
    
    print("\n" + "="*70)
    print("  Text Masking Experiment")
    print("="*70)
    print(f"Masking ratios: {args.mask_ratios}")
    print(f"N_slices: {args.n_slices} (moderate degradation)")
    print(f"Modality: {args.modality}")
    print()
    
    # Load models
    img_ckpt = ROOT / f"runs/image_{args.modality}/models/best_image.pt"
    txt_ckpt = ROOT / "runs/text/models/best_text.pt"
    
    if not img_ckpt.exists():
        raise FileNotFoundError(f"Image checkpoint not found: {img_ckpt}")
    if not txt_ckpt.exists():
        raise FileNotFoundError(f"Text checkpoint not found: {txt_ckpt}")
    
    print(f"Loading checkpoints...")
    print(f"  Image: {img_ckpt}")
    print(f"  Text:  {txt_ckpt}")
    
    img_model = ImageModel(num_classes=2, pretrained=True).to(device)
    txt_model = TextModel(
        num_classes=2,
        model_name=args.tokenizer,
        freeze_bert=True
    ).to(device)
    
    img_model.load_state_dict(torch_load_weights(img_ckpt, device))
    txt_model.load_state_dict(torch_load_weights(txt_ckpt, device))
    
    print("✅ Models loaded!\n")
    
    # Train LR on original text (mask_ratio=0)
    print("="*70)
    print(f"Training LR Fusion (N={args.n_slices}, original text)...")
    print("="*70)
    
    train_loader = get_masked_dataloader(
        csv_path=str(ROOT / args.train_csv),
        mask_ratio=0.0,
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
    
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=args.seed)
    lr.fit(X_train_s, y_train)
    
    coef = lr.coef_[0]
    print(f"  LR Coef: img={coef[0]:.4f}, txt={coef[1]:.4f}")
    print(f"  Ratio: {coef[0]/(coef[1]+1e-8):.2f}:1")
    print()
    
    # Evaluate with different masking ratios
    results = {}
    
    for mask_ratio in args.mask_ratios:
        print("="*70)
        print(f"Evaluating mask_ratio={mask_ratio:.0%}")
        print("="*70)
        
        # Test loader with masked text
        test_loader = get_masked_dataloader(
            csv_path=str(ROOT / args.test_csv),
            mask_ratio=mask_ratio,
            batch_size=args.batch_size,
            modality=args.modality,
            n_slices=args.n_slices,
            image_size=(args.img_size, args.img_size),
            tokenizer_name=args.tokenizer,
            max_text_length=args.max_text_len,
        )
        
        # Evaluate
        y_test, X_test = extract_logits(img_model, txt_model, test_loader, device)
        X_test_s = scaler.transform(X_test)
        p_test = lr.predict_proba(X_test_s)[:, 1]
        
        metrics = compute_metrics(y_test, p_test)
        
        print(f"  Multi-LR: ROC={metrics['roc_auc']:.4f}, PR={metrics['pr_auc']:.4f}")
        
        # Compare with original (0%)
        if mask_ratio == 0.0:
            baseline_roc = metrics['roc_auc']
            baseline_pr = metrics['pr_auc']
            delta_roc = 0.0
            delta_pr = 0.0
        else:
            delta_roc = metrics['roc_auc'] - baseline_roc
            delta_pr = metrics['pr_auc'] - baseline_pr
        
        print(f"  Δ ROC-AUC: {delta_roc:+.4f} ({delta_roc/baseline_roc*100:+.2f}%)")
        print(f"  Δ PR-AUC:  {delta_pr:+.4f}")
        print()
        
        results[f"mask_{int(mask_ratio*100)}pct"] = {
            'mask_ratio': float(mask_ratio),
            'metrics': metrics,
            'delta_roc_auc': float(delta_roc),
            'delta_pr_auc': float(delta_pr),
            'delta_roc_pct': float(delta_roc/baseline_roc*100) if baseline_roc > 0 else 0.0,
        }
    
    # Save results
    output_dir = ROOT / "results/robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "text_masking.json"
    
    final_results = {
        'experiment': 'text_masking',
        'n_slices': args.n_slices,
        'modality': args.modality,
        'lr_coefficients': {
            'img': float(coef[0]),
            'txt': float(coef[1]),
            'ratio': float(coef[0] / (coef[1] + 1e-8))
        },
        'results': results
    }
    
    save_json(output_path, final_results)
    
    print("="*70)
    print(f"✅ Results saved to: {output_path}")
    print("="*70)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Text Masking Effect")
    print("="*70)
    print(f"{'Mask %':<15} {'ROC-AUC':<12} {'Δ from 0%':<12} {'% Change':<12}")
    print("-"*70)
    
    for key in ['mask_0pct', 'mask_20pct', 'mask_40pct', 'mask_60pct']:
        if key in results:
            r = results[key]
            pct = int(r['mask_ratio'] * 100)
            roc = r['metrics']['roc_auc']
            delta = r['delta_roc_auc']
            pct_change = r['delta_roc_pct']
            
            print(f"{pct}%{'':<12} {roc:<12.4f} {delta:<+12.4f} {pct_change:<+12.2f}%")
    
    print("="*70)
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Check if performance is stable under masking
    if 'mask_60pct' in results:
        drop_abs = abs(results['mask_60pct']['delta_roc_auc'])
        drop_pct = abs(results['mask_60pct']['delta_roc_pct'])
        
        print(f"Performance drop at 60% masking:")
        print(f"  Absolute: {drop_abs:.4f}")
        print(f"  Relative: {drop_pct:.2f}%")
        print()
        
        if drop_pct < 2.0:
            print("✅ Performance STABLE under heavy masking (Δ<2%)")
            print()
            print("CONCLUSION:")
            print("  → Text provides MODEL STABILIZATION, not task-specific information")
            print("  → Supports 'regularization/smoothing' hypothesis")
            print("  → Even when 60% of tokens are masked, multimodal performance")
            print("    remains nearly unchanged, indicating text content matters less")
            print("    than its presence for decision stability")
            print()
            print("Clinical Implication:")
            print("  Incomplete/noisy radiology reports may still provide value")
            print("  through statistical regularization rather than semantic content")
        elif drop_pct < 5.0:
            print("⚠️  Performance MODERATELY affected (2%<Δ<5%)")
            print()
            print("CONCLUSION:")
            print("  → Text contains SOME task-relevant information")
            print("  → But effect is modest, suggesting limited ET-specific content")
            print("  → Combination of weak information + regularization")
        else:
            print("⚠️  Performance SIGNIFICANTLY drops (Δ>5%)")
            print()
            print("CONCLUSION:")
            print("  → Text contains MEANINGFUL task-relevant information")
            print("  → Need to revise 'low-information' hypothesis")
            print("  → Semantic content is important for fusion benefits")
    
    print("="*70)
    
    print("\n✅ Experiment completed!")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Text Masking Experiment')
    
    # Experiment settings
    parser.add_argument('--mask_ratios', type=float, nargs='+', default=[0.0, 0.2, 0.4, 0.6],
                       help='Text masking ratios')
    parser.add_argument('--n_slices', type=int, default=8,
                       help='N_slices (use moderate degradation)')
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
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_text_masking_experiment(args)


if __name__ == "__main__":
    main()