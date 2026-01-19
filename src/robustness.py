"""
Robustness Experiment: Image Degradation
N_slices를 16 → 8 → 4로 줄이면서 성능 변화 측정
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import get_dataloader
from src.models import ImageModel, TextModel
from utils.metrics import (
    set_seed,
    compute_metrics,
    save_json,
    torch_load_weights,
)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    return device


@torch.no_grad()
def eval_image_probs(model, loader, device):
    """Image-only evaluation"""
    model.eval()
    ys, ps = [], []
    
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        
        ys.append(labels.numpy())
        ps.append(probs.cpu().numpy())
    
    return np.concatenate(ys), np.concatenate(ps)


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


def run_robustness_experiment(args):
    """
    Main robustness experiment
    """
    set_seed(args.seed)
    device = get_device()
    
    print("\n" + "="*70)
    print("  Robustness Experiment: Image Degradation")
    print("="*70)
    print(f"N_slices: {args.n_slices_list}")
    print(f"Modality: {args.modality}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Load best models
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
    
    print("="*70)
    print("Note: LR will be trained separately for each N_slices")
    print("="*70)
    print()
    
    # Evaluate on each N_slices
    results = {}
    
    for n in args.n_slices_list:
        print("="*70)
        print(f"Evaluating N={n} slices")
        print("="*70)
        
        # Train LR on this N (N별로 재학습!)
        print(f"  Training LR on N={n}...")
        
        train_loader_mm_n = get_dataloader(
            csv_path=str(ROOT / args.train_csv),
            mode='multimodal',
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            modality=args.modality,
            n_slices=n,  # ← 이게 핵심!
            image_size=(args.img_size, args.img_size),
            tokenizer_name=args.tokenizer,
            max_text_length=args.max_text_len,
        )
        
        y_train_n, X_train_n = extract_logits(img_model, txt_model, train_loader_mm_n, device)
        
        scaler_n = StandardScaler()
        X_train_s_n = scaler_n.fit_transform(X_train_n)
        
        lr_n = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=args.seed)
        lr_n.fit(X_train_s_n, y_train_n)
        
        coef_n = lr_n.coef_[0]
        print(f"    LR Coef: img={coef_n[0]:.4f}, txt={coef_n[1]:.4f}")
        
        # Test loaders
        test_loader_img = get_dataloader(
            csv_path=str(ROOT / args.test_csv),
            mode='image',
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            modality=args.modality,
            n_slices=n,
            image_size=(args.img_size, args.img_size),
        )
        
        test_loader_mm = get_dataloader(
            csv_path=str(ROOT / args.test_csv),
            mode='multimodal',
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            modality=args.modality,
            n_slices=n,
            image_size=(args.img_size, args.img_size),
            tokenizer_name=args.tokenizer,
            max_text_length=args.max_text_len,
        )
        
        # Image-only
        y_test, p_img = eval_image_probs(img_model, test_loader_img, device)
        img_metrics = compute_metrics(y_test, p_img)
        
        print(f"  Image-only: ROC={img_metrics['roc_auc']:.4f}, PR={img_metrics['pr_auc']:.4f}")
        
        # Multimodal-LR (N별로 학습된 LR 사용!)
        y_test, X_test = extract_logits(img_model, txt_model, test_loader_mm, device)
        X_test_s = scaler_n.transform(X_test)  # ← N별 scaler
        p_lr = lr_n.predict_proba(X_test_s)[:, 1]  # ← N별 LR
        
        lr_metrics = compute_metrics(y_test, p_lr)
        
        print(f"  Multi-LR:   ROC={lr_metrics['roc_auc']:.4f}, PR={lr_metrics['pr_auc']:.4f}")
        
        # Delta
        delta_roc = lr_metrics['roc_auc'] - img_metrics['roc_auc']
        delta_pr = lr_metrics['pr_auc'] - img_metrics['pr_auc']
        
        print(f"  Δ ROC-AUC: {delta_roc:+.4f}")
        print(f"  Δ PR-AUC:  {delta_pr:+.4f}")
        print()
        
        results[f"N{n}"] = {
            'n_slices': n,
            'image_only': img_metrics,
            'multimodal_lr': lr_metrics,
            'delta_roc_auc': float(delta_roc),
            'delta_pr_auc': float(delta_pr),
            'lr_coefficients': {
                'img': float(coef_n[0]),
                'txt': float(coef_n[1]),
                'ratio': float(coef_n[0] / (coef_n[1] + 1e-8))
            }
        }
    
    # Save results
    output_dir = ROOT / "results/robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "image_degradation.json"
    
    final_results = {
        'experiment': 'image_degradation',
        'modality': args.modality,
        'note': 'LR trained separately for each N_slices',
        'results': results
    }
    
    save_json(output_path, final_results)
    
    print("="*70)
    print(f"✅ Results saved to: {output_path}")
    print("="*70)
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Test Set Performance")
    print("="*70)
    print(f"{'N_slices':<12} {'Image-only':<12} {'Multi-LR':<12} {'Δ ROC-AUC':<12}")
    print("-"*70)
    
    for key in ['N16', 'N8', 'N4']:
        if key in results:
            r = results[key]
            img_auc = r['image_only']['roc_auc']
            lr_auc = r['multimodal_lr']['roc_auc']
            delta = r['delta_roc_auc']
            
            print(f"{r['n_slices']:<12} {img_auc:<12.4f} {lr_auc:<12.4f} {delta:<+12.4f}")
    
    print("="*70)
    print("\n✅ Experiment completed!")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Robustness Experiment: Image Degradation')
    
    # Experiment settings
    parser.add_argument('--n_slices_list', type=int, nargs='+', default=[16, 8, 4],
                       help='List of n_slices to evaluate')
    parser.add_argument('--modality', type=str, default='t1ce',
                       choices=['flair', 't1', 't1ce', 't2'])
    
    # Data paths
    parser.add_argument('--train_csv', type=str, default='data/splits/train.csv')
    parser.add_argument('--test_csv', type=str, default='data/splits/test.csv')
    
    # Model settings
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--tokenizer', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--max_text_len', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_robustness_experiment(args)


if __name__ == "__main__":
    main()