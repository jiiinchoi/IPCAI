"""
Utility functions
- Seed 설정
- Metrics 계산 (ROC-AUC, PR-AUC)
- Threshold 선택
- JSON 저장
"""
import random
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix,
    f1_score,
    classification_report
)


def set_seed(seed: int = 42):
    """재현성을 위한 seed 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # 속도 느려질 수 있음


def safe_auroc(y_true, y_prob) -> float:
    """ROC-AUC 계산 (클래스 1개만 있으면 NaN 반환)"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    if len(np.unique(y_true)) < 2:
        return float('nan')
    
    try:
        return float(roc_auc_score(y_true, y_prob))
    except:
        return float('nan')


def safe_auprc(y_true, y_prob) -> float:
    """PR-AUC 계산"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    if len(np.unique(y_true)) < 2:
        return float('nan')
    
    try:
        return float(average_precision_score(y_true, y_prob))
    except:
        return float('nan')


def compute_metrics(y_true, y_prob) -> dict:
    """
    전체 metrics 한번에 계산
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities (positive class)
    
    Returns:
        dict: {'roc_auc': float, 'pr_auc': float}
    """
    return {
        'roc_auc': safe_auroc(y_true, y_prob),
        'pr_auc': safe_auprc(y_true, y_prob),
    }


def pick_threshold_by_balanced_acc(y_true, y_prob) -> float:
    """
    Balanced accuracy를 최대화하는 threshold 선택
    Validation set에서 선택한 threshold를 test에 적용
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    best_thr = 0.5
    best_score = -1.0
    
    for thr in np.linspace(0.1, 0.9, 17):
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn + 1e-8)  # Sensitivity
            tnr = tn / (tn + fp + 1e-8)  # Specificity
            bal_acc = 0.5 * (tpr + tnr)
            
            if bal_acc > best_score:
                best_score = bal_acc
                best_thr = float(thr)
    
    return best_thr


def confusion_at_threshold(y_true, y_prob, threshold: float):
    """주어진 threshold에서 confusion matrix 계산"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    return cm.tolist()


def save_json(filepath: Path, data: dict):
    """JSON 파일로 저장"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> dict:
    """JSON 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def torch_load_weights(path: Path, device):
    """
    PyTorch 모델 가중치 로드
    weights_only=True 지원 여부에 따라 자동 처리
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # PyTorch 버전이 weights_only를 지원하지 않는 경우
        return torch.load(path, map_location=device)


if __name__ == "__main__":
    # 테스트
    print("=== Utils Test ===")
    
    # Seed 테스트
    set_seed(42)
    print(f"Random: {random.random():.4f}")
    print(f"Numpy: {np.random.rand():.4f}")
    print(f"Torch: {torch.rand(1).item():.4f}")
    
    # Metrics 테스트
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
    
    print(f"\nROC-AUC: {safe_auroc(y_true, y_prob):.4f}")
    print(f"PR-AUC: {safe_auprc(y_true, y_prob):.4f}")
    
    thr = pick_threshold_by_balanced_acc(y_true, y_prob)
    print(f"Best threshold: {thr:.3f}")
    
    cm = confusion_at_threshold(y_true, y_prob, thr)
    print(f"Confusion Matrix:\n{np.array(cm)}")
    
    print("\n✅ Utils test passed!")