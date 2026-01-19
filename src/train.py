"""
Main Training Script
- Image-only
- Text-only
- Multimodal Weighted
- Multimodal LR
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import get_dataloader
from src.models import ImageModel, TextModel, MultimodalModel
from utils.metrics import (
    set_seed,
    compute_metrics,
    pick_threshold_by_balanced_acc,
    confusion_at_threshold,
    save_json,
    torch_load_weights,
)


def get_device():
    """GPU/CPU 자동 선택"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    return device


# ====================================
# Evaluation Functions
# ====================================

@torch.no_grad()
def eval_image_probs(model, loader, device):
    """Image-only 평가 - (y_true, y_prob) 반환"""
    model.eval()
    ys, ps = [], []
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)  # (B, 2)
        probs = torch.softmax(logits, dim=1)[:, 1]  # positive class prob
        
        ys.append(labels.cpu().numpy())
        ps.append(probs.cpu().numpy())
    
    y_true = np.concatenate(ys).astype(int)
    y_prob = np.concatenate(ps).astype(float)
    
    return y_true, y_prob


@torch.no_grad()
def eval_text_probs(model, loader, device):
    """Text-only 평가 - (y_true, y_prob) 반환"""
    model.eval()
    ys, ps = [], []
    
    for tokens, labels in loader:
        labels = labels.to(device)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        
        logits = model(input_ids, attention_mask)  # (B, 2)
        probs = torch.softmax(logits, dim=1)[:, 1]
        
        ys.append(labels.cpu().numpy())
        ps.append(probs.cpu().numpy())
    
    y_true = np.concatenate(ys).astype(int)
    y_prob = np.concatenate(ps).astype(float)
    
    return y_true, y_prob


@torch.no_grad()
def eval_multimodal_weighted(model, loader, device, alpha):
    """Multimodal weighted fusion 평가"""
    model.eval()
    ys, ps = [], []
    
    for images, tokens, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        
        # Forward
        _, img_logits, txt_logits = model(images, input_ids, attention_mask)
        
        # Weighted fusion at probability level
        img_probs = torch.softmax(img_logits, dim=1)[:, 1]
        txt_probs = torch.softmax(txt_logits, dim=1)[:, 1]
        fused_probs = alpha * img_probs + (1 - alpha) * txt_probs
        
        ys.append(labels.cpu().numpy())
        ps.append(fused_probs.cpu().numpy())
    
    y_true = np.concatenate(ys).astype(int)
    y_prob = np.concatenate(ps).astype(float)
    
    return y_true, y_prob


# ====================================
# Training Functions
# ====================================

def train_image_only(args, device):
    """Image-only 학습"""
    print("\n" + "="*50)
    print("  Training Image-only Model")
    print("="*50)
    
    # Dataloaders
    train_loader = get_dataloader(
        csv_path=args.train_csv,
        mode='image',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        modality=args.modality,
        n_slices=args.n_slices,
        image_size=(args.img_size, args.img_size),
    )
    
    val_loader = get_dataloader(
        csv_path=args.val_csv,
        mode='image',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        modality=args.modality,
        n_slices=args.n_slices,
        image_size=(args.img_size, args.img_size),
    )
    
    test_loader = get_dataloader(
        csv_path=args.test_csv,
        mode='image',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        modality=args.modality,
        n_slices=args.n_slices,
        image_size=(args.img_size, args.img_size),
    )
    
    # Model
    model = ImageModel(num_classes=2, pretrained=True).to(device)
    
    # Class weights (inverse frequency)
    y_train = []
    for _, labels in train_loader:
        y_train.append(labels.numpy())
    y_train = np.concatenate(y_train)
    
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    class_weights = torch.tensor([1.0/neg_count, 1.0/pos_count], dtype=torch.float32).to(device)
    
    print(f"\n[Class Balance] Neg={neg_count}, Pos={pos_count}")
    print(f"[Class Weights] {class_weights.cpu().numpy()}")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # Training loop
    best_val_pr = -1.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        y_val, p_val = eval_image_probs(model, val_loader, device)
        val_metrics = compute_metrics(y_val, p_val)
        
        avg_loss = np.mean(train_losses)
        print(f"[Epoch {epoch:02d}] Loss={avg_loss:.4f} | Val ROC-AUC={val_metrics['roc_auc']:.4f} PR-AUC={val_metrics['pr_auc']:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': float(avg_loss),
            'val_roc_auc': val_metrics['roc_auc'],
            'val_pr_auc': val_metrics['pr_auc'],
        })
        
        # Save best model (based on PR-AUC)
        if not np.isnan(val_metrics['pr_auc']) and val_metrics['pr_auc'] > best_val_pr:
            best_val_pr = val_metrics['pr_auc']
            save_dir = Path(f"runs/image_{args.modality}/models")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "best_image.pt")
            print(f"  → Best model saved! (PR-AUC={best_val_pr:.4f})")
    
    # Load best model for final evaluation
    model.load_state_dict(torch_load_weights(save_dir / "best_image.pt", device))
    
    # Final evaluation
    y_val, p_val = eval_image_probs(model, val_loader, device)
    y_test, p_test = eval_image_probs(model, test_loader, device)
    
    val_metrics = compute_metrics(y_val, p_val)
    test_metrics = compute_metrics(y_test, p_test)
    
    # Threshold from val
    threshold = pick_threshold_by_balanced_acc(y_val, p_val)
    val_cm = confusion_at_threshold(y_val, p_val, threshold)
    test_cm = confusion_at_threshold(y_test, p_test, threshold)
    
    # Save results
    results = {
        'mode': 'image',
        'modality': args.modality,
        'n_slices': args.n_slices,
        'val': {**val_metrics, 'threshold': threshold, 'confusion_matrix': val_cm},
        'test': {**test_metrics, 'threshold': threshold, 'confusion_matrix': test_cm},
        'history': history,
    }
    
    save_path = Path(f"runs/image_{args.modality}/logs/metrics.json")
    save_json(save_path, results)
    
    print(f"\n[Final Results]")
    print(f"  Val:  ROC-AUC={val_metrics['roc_auc']:.4f}, PR-AUC={val_metrics['pr_auc']:.4f}")
    print(f"  Test: ROC-AUC={test_metrics['roc_auc']:.4f}, PR-AUC={test_metrics['pr_auc']:.4f}")
    print(f"  Saved: {save_path}")


def train_text_only(args, device):
    """Text-only 학습"""
    print("\n" + "="*50)
    print("  Training Text-only Model")
    print("="*50)
    
    # Dataloaders
    train_loader = get_dataloader(
        csv_path=args.train_csv,
        mode='text',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        tokenizer_name=args.tokenizer,
        max_text_length=args.max_text_len,
    )
    
    val_loader = get_dataloader(
        csv_path=args.val_csv,
        mode='text',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        tokenizer_name=args.tokenizer,
        max_text_length=args.max_text_len,
    )
    
    test_loader = get_dataloader(
        csv_path=args.test_csv,
        mode='text',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        tokenizer_name=args.tokenizer,
        max_text_length=args.max_text_len,
    )
    
    # Model
    model = TextModel(
        num_classes=2,
        model_name=args.tokenizer,
        freeze_bert=args.freeze_bert
    ).to(device)
    
    # Class weights
    y_train = []
    for _, labels in train_loader:
        y_train.append(labels.numpy())
    y_train = np.concatenate(y_train)
    
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    class_weights = torch.tensor([1.0/neg_count, 1.0/pos_count], dtype=torch.float32).to(device)
    
    print(f"\n[Class Balance] Neg={neg_count}, Pos={pos_count}")
    print(f"[Class Weights] {class_weights.cpu().numpy()}")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    
    # Training loop
    best_val_pr = -1.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        
        for tokens, labels in train_loader:
            labels = labels.to(device)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        y_val, p_val = eval_text_probs(model, val_loader, device)
        val_metrics = compute_metrics(y_val, p_val)
        
        avg_loss = np.mean(train_losses)
        print(f"[Epoch {epoch:02d}] Loss={avg_loss:.4f} | Val ROC-AUC={val_metrics['roc_auc']:.4f} PR-AUC={val_metrics['pr_auc']:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': float(avg_loss),
            'val_roc_auc': val_metrics['roc_auc'],
            'val_pr_auc': val_metrics['pr_auc'],
        })
        
        # Save best model
        if not np.isnan(val_metrics['pr_auc']) and val_metrics['pr_auc'] > best_val_pr:
            best_val_pr = val_metrics['pr_auc']
            save_dir = Path("runs/text/models")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "best_text.pt")
            print(f"  → Best model saved! (PR-AUC={best_val_pr:.4f})")
    
    # Load best model
    model.load_state_dict(torch_load_weights(Path("runs/text/models/best_text.pt"), device))
    
    # Final evaluation
    y_val, p_val = eval_text_probs(model, val_loader, device)
    y_test, p_test = eval_text_probs(model, test_loader, device)
    
    val_metrics = compute_metrics(y_val, p_val)
    test_metrics = compute_metrics(y_test, p_test)
    
    threshold = pick_threshold_by_balanced_acc(y_val, p_val)
    val_cm = confusion_at_threshold(y_val, p_val, threshold)
    test_cm = confusion_at_threshold(y_test, p_test, threshold)
    
    # Save results
    results = {
        'mode': 'text',
        'tokenizer': args.tokenizer,
        'val': {**val_metrics, 'threshold': threshold, 'confusion_matrix': val_cm},
        'test': {**test_metrics, 'threshold': threshold, 'confusion_matrix': test_cm},
        'history': history,
    }
    
    save_path = Path("runs/text/logs/metrics.json")
    save_json(save_path, results)
    
    print(f"\n[Final Results]")
    print(f"  Val:  ROC-AUC={val_metrics['roc_auc']:.4f}, PR-AUC={val_metrics['pr_auc']:.4f}")
    print(f"  Test: ROC-AUC={test_metrics['roc_auc']:.4f}, PR-AUC={test_metrics['pr_auc']:.4f}")
    print(f"  Saved: {save_path}")


def eval_multimodal_weighted_fusion(args, device):
    """Multimodal Weighted Fusion 평가"""
    print("\n" + "="*50)
    print("  Evaluating Multimodal Weighted Fusion")
    print("="*50)
    
    # Load best models
    img_ckpt = Path(f"runs/image_{args.modality}/models/best_image.pt")
    txt_ckpt = Path("runs/text/models/best_text.pt")
    
    if not img_ckpt.exists():
        raise FileNotFoundError(f"Image checkpoint not found: {img_ckpt}")
    if not txt_ckpt.exists():
        raise FileNotFoundError(f"Text checkpoint not found: {txt_ckpt}")
    
    print(f"Loading checkpoints...")
    print(f"  Image: {img_ckpt}")
    print(f"  Text:  {txt_ckpt}")
    
    img_model = ImageModel(num_classes=2, pretrained=True)
    txt_model = TextModel(num_classes=2, model_name=args.tokenizer, freeze_bert=True)
    
    img_model.load_state_dict(torch_load_weights(img_ckpt, device))
    txt_model.load_state_dict(torch_load_weights(txt_ckpt, device))
    
    model = MultimodalModel(img_model, txt_model, fusion_mode='weighted', alpha=0.5).to(device)
    
    # Dataloaders
    val_loader = get_dataloader(
        csv_path=args.val_csv,
        mode='multimodal',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        modality=args.modality,
        n_slices=args.n_slices,
        image_size=(args.img_size, args.img_size),
        tokenizer_name=args.tokenizer,
        max_text_length=args.max_text_len,
    )
    
    test_loader = get_dataloader(
        csv_path=args.test_csv,
        mode='multimodal',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        modality=args.modality,
        n_slices=args.n_slices,
        image_size=(args.img_size, args.img_size),
        tokenizer_name=args.tokenizer,
        max_text_length=args.max_text_len,
    )
    
    # Tune alpha on validation set
    print("\nTuning alpha...")
    best_alpha = 0.5
    best_roc = -1.0
    
    for alpha in np.linspace(0, 1, 21):
        y_val, p_val = eval_multimodal_weighted(model, val_loader, device, alpha)
        roc = compute_metrics(y_val, p_val)['roc_auc']
        
        if not np.isnan(roc) and roc > best_roc:
            best_roc = roc
            best_alpha = float(alpha)
    
    print(f"  Best alpha: {best_alpha:.2f} (Val ROC-AUC={best_roc:.4f})")
    
    # Final evaluation with best alpha
    y_val, p_val = eval_multimodal_weighted(model, val_loader, device, best_alpha)
    y_test, p_test = eval_multimodal_weighted(model, test_loader, device, best_alpha)
    
    val_metrics = compute_metrics(y_val, p_val)
    test_metrics = compute_metrics(y_test, p_test)
    
    threshold = pick_threshold_by_balanced_acc(y_val, p_val)
    val_cm = confusion_at_threshold(y_val, p_val, threshold)
    test_cm = confusion_at_threshold(y_test, p_test, threshold)
    
    # Save results
    results = {
        'mode': 'multimodal_weighted',
        'modality': args.modality,
        'best_alpha': best_alpha,
        'val': {**val_metrics, 'threshold': threshold, 'confusion_matrix': val_cm},
        'test': {**test_metrics, 'threshold': threshold, 'confusion_matrix': test_cm},
    }
    
    save_path = Path(f"runs/multimodal_{args.modality}/logs/metrics_weighted.json")
    save_json(save_path, results)
    
    print(f"\n[Final Results]")
    print(f"  Alpha: {best_alpha:.2f}")
    print(f"  Val:  ROC-AUC={val_metrics['roc_auc']:.4f}, PR-AUC={val_metrics['pr_auc']:.4f}")
    print(f"  Test: ROC-AUC={test_metrics['roc_auc']:.4f}, PR-AUC={test_metrics['pr_auc']:.4f}")
    print(f"  Saved: {save_path}")


def eval_multimodal_lr_fusion(args, device):
    """Multimodal LR Fusion 평가"""
    print("\n" + "="*50)
    print("  Evaluating Multimodal LR Fusion")
    print("="*50)
    
    # Load best models
    img_ckpt = Path(f"runs/image_{args.modality}/models/best_image.pt")
    txt_ckpt = Path("runs/text/models/best_text.pt")
    
    if not img_ckpt.exists():
        raise FileNotFoundError(f"Image checkpoint not found: {img_ckpt}")
    if not txt_ckpt.exists():
        raise FileNotFoundError(f"Text checkpoint not found: {txt_ckpt}")
    
    print(f"Loading checkpoints...")
    print(f"  Image: {img_ckpt}")
    print(f"  Text:  {txt_ckpt}")
    
    img_model = ImageModel(num_classes=2, pretrained=True).to(device)
    txt_model = TextModel(num_classes=2, model_name=args.tokenizer, freeze_bert=True).to(device)
    
    img_model.load_state_dict(torch_load_weights(img_ckpt, device))
    txt_model.load_state_dict(torch_load_weights(txt_ckpt, device))
    
    # Dataloaders
    train_loader = get_dataloader(
        csv_path=args.train_csv,
        mode='multimodal',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        modality=args.modality,
        n_slices=args.n_slices,
        image_size=(args.img_size, args.img_size),
        tokenizer_name=args.tokenizer,
        max_text_length=args.max_text_len,
    )
    
    val_loader = get_dataloader(
        csv_path=args.val_csv,
        mode='multimodal',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        modality=args.modality,
        n_slices=args.n_slices,
        image_size=(args.img_size, args.img_size),
        tokenizer_name=args.tokenizer,
        max_text_length=args.max_text_len,
    )
    
    test_loader = get_dataloader(
        csv_path=args.test_csv,
        mode='multimodal',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        modality=args.modality,
        n_slices=args.n_slices,
        image_size=(args.img_size, args.img_size),
        tokenizer_name=args.tokenizer,
        max_text_length=args.max_text_len,
    )
    
    # Extract logits
    print("\nExtracting logits...")
    
    @torch.no_grad()
    def extract_logits(loader):
        img_model.eval()
        txt_model.eval()
        
        ys, img_logits_list, txt_logits_list = [], [], []
        
        for images, tokens, labels in loader:
            images = images.to(device)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            
            img_logits = img_model(images)  # (B, 2)
            txt_logits = txt_model(input_ids, attention_mask)  # (B, 2)
            
            # Use log-odds: logit[:,1] - logit[:,0]
            img_logit_pos = (img_logits[:, 1] - img_logits[:, 0]).cpu().numpy()
            txt_logit_pos = (txt_logits[:, 1] - txt_logits[:, 0]).cpu().numpy()
            
            ys.append(labels.numpy())
            img_logits_list.append(img_logit_pos)
            txt_logits_list.append(txt_logit_pos)
        
        y = np.concatenate(ys).astype(int)
        X = np.stack([
            np.concatenate(img_logits_list),
            np.concatenate(txt_logits_list)
        ], axis=1)  # (N, 2)
        
        return y, X
    
    y_train, X_train = extract_logits(train_loader)
    y_val, X_val = extract_logits(val_loader)
    y_test, X_test = extract_logits(test_loader)
    
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Train LR with standardization
    print("\nTraining Logistic Regression...")
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=args.seed)
    lr.fit(X_train_s, y_train)
    
    # Coefficients
    coef = lr.coef_[0]
    print(f"  LR Coefficients: img={coef[0]:.4f}, txt={coef[1]:.4f}")
    print(f"  Intercept: {lr.intercept_[0]:.4f}")
    
    # Predictions
    p_val = lr.predict_proba(X_val_s)[:, 1]
    p_test = lr.predict_proba(X_test_s)[:, 1]
    
    val_metrics = compute_metrics(y_val, p_val)
    test_metrics = compute_metrics(y_test, p_test)
    
    threshold = pick_threshold_by_balanced_acc(y_val, p_val)
    val_cm = confusion_at_threshold(y_val, p_val, threshold)
    test_cm = confusion_at_threshold(y_test, p_test, threshold)
    
    # Save results
    results = {
        'mode': 'multimodal_lr',
        'modality': args.modality,
        'lr_coefficients': {
            'img': float(coef[0]),
            'txt': float(coef[1]),
            'intercept': float(lr.intercept_[0])
        },
        'val': {**val_metrics, 'threshold': threshold, 'confusion_matrix': val_cm},
        'test': {**test_metrics, 'threshold': threshold, 'confusion_matrix': test_cm},
    }
    
    save_path = Path(f"runs/multimodal_{args.modality}/logs/metrics_lr.json")
    save_json(save_path, results)
    
    print(f"\n[Final Results]")
    print(f"  LR Coef (img, txt): ({coef[0]:.4f}, {coef[1]:.4f})")
    print(f"  Val:  ROC-AUC={val_metrics['roc_auc']:.4f}, PR-AUC={val_metrics['pr_auc']:.4f}")
    print(f"  Test: ROC-AUC={test_metrics['roc_auc']:.4f}, PR-AUC={test_metrics['pr_auc']:.4f}")
    print(f"  Saved: {save_path}")


# ====================================
# Main
# ====================================

def main():
    parser = argparse.ArgumentParser()
    
    # Mode
    parser.add_argument('--mode', type=str, required=True,
                       choices=['image', 'text', 'weighted', 'lr'],
                       help='Training/Evaluation mode')
    
    # Data paths
    parser.add_argument('--train_csv', type=str, default='data/splits/train.csv')
    parser.add_argument('--val_csv', type=str, default='data/splits/val.csv')
    parser.add_argument('--test_csv', type=str, default='data/splits/test.csv')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Image settings
    parser.add_argument('--modality', type=str, default='t1ce',
                       choices=['flair', 't1', 't1ce', 't2'])
    parser.add_argument('--n_slices', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=128)
    
    # Text settings
    parser.add_argument('--tokenizer', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--max_text_len', type=int, default=512)
    parser.add_argument('--freeze_bert', action='store_true',
                       help='Freeze BERT weights')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Run
    if args.mode == 'image':
        train_image_only(args, device)
    elif args.mode == 'text':
        train_text_only(args, device)
    elif args.mode == 'weighted':
        eval_multimodal_weighted_fusion(args, device)
    elif args.mode == 'lr':
        eval_multimodal_lr_fusion(args, device)
    
    print("\n" + "="*50)
    print("  DONE!")
    print("="*50)


if __name__ == "__main__":
    main()