"""
Case Analysis: 텍스트가 예측을 바꾼 케이스 (Flip/Stabilize/Agree)
N=6 조건에서 분석
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import get_dataloader
from src.models import ImageModel, TextModel
from utils.metrics import torch_load_weights, set_seed


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def extract_logits(img_model, txt_model, loader, device):
    """Extract (img_logit_pos, txt_logit_pos) where logit_pos = logit1-logit0"""
    img_model.eval()
    txt_model.eval()

    ys, img_list, txt_list = [], [], []

    for images, tokens, labels in loader:
        images = images.to(device)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        img_logits = img_model(images)                   # (B,2)
        txt_logits = txt_model(input_ids, attention_mask)  # (B,2)

        img_logit_pos = (img_logits[:, 1] - img_logits[:, 0]).cpu().numpy()
        txt_logit_pos = (txt_logits[:, 1] - txt_logits[:, 0]).cpu().numpy()

        ys.append(labels.numpy())
        img_list.append(img_logit_pos)
        txt_list.append(txt_logit_pos)

    y = np.concatenate(ys).astype(int)
    X = np.stack([np.concatenate(img_list), np.concatenate(txt_list)], axis=1)  # (N,2)
    return y, X


def main():
    set_seed(42)
    device = get_device()
    n_slices = 6

    print("\n" + "="*70)
    print("  Case Analysis: Text Influence on Predictions")
    print("="*70)
    print(f"Condition: N={n_slices} slices\n")

    # Load models
    img_ckpt = ROOT / "runs/image_t1ce/models/best_image.pt"
    txt_ckpt = ROOT / "runs/text/models/best_text.pt"

    img_model = ImageModel(num_classes=2, pretrained=True).to(device)
    txt_model = TextModel(num_classes=2, model_name='emilyalsentzer/Bio_ClinicalBERT', freeze_bert=True).to(device)

    img_model.load_state_dict(torch_load_weights(img_ckpt, device))
    txt_model.load_state_dict(torch_load_weights(txt_ckpt, device))

    # ---- Train LR on TRAIN (N=6) ----
    train_loader = get_dataloader(
        csv_path=str(ROOT / 'data/splits/train.csv'),
        mode='multimodal',
        modality='t1ce',
        n_slices=n_slices,
        batch_size=8,
        shuffle=False,          # ✅ 재현/인덱스 일관
        num_workers=0
    )

    y_train, X_train = extract_logits(img_model, txt_model, train_loader, device)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)

    beta_img, beta_txt = lr.coef_[0]
    bias = lr.intercept_[0]

    print(f"LR trained: β_img={beta_img:.4f}, β_txt={beta_txt:.4f}, b={bias:.4f}\n")

    # ---- Test (N=6) ----
    test_loader = get_dataloader(
        csv_path=str(ROOT / 'data/splits/test.csv'),
        mode='multimodal',
        modality='t1ce',
        n_slices=n_slices,
        batch_size=8,
        shuffle=False,          # ✅ 인덱스/케이스 고정
        num_workers=0
    )

    y_test, X_test = extract_logits(img_model, txt_model, test_loader, device)
    X_test_s = scaler.transform(X_test)

    # ✅ Image-only prob: sigmoid(raw img_logit_pos)  (softmax(2-class)와 동치)
    p_img_only = sigmoid(X_test[:, 0])

    # ✅ Multi-LR prob: sigmoid(beta·z + intercept)
    score = (X_test_s[:, 0] * beta_img) + (X_test_s[:, 1] * beta_txt) + bias
    p_multi = sigmoid(score)

    # (검증) sklearn predict_proba와 거의 같아야 정상
    p_multi_sklearn = lr.predict_proba(X_test_s)[:, 1]
    max_diff = np.max(np.abs(p_multi - p_multi_sklearn))
    print(f"[Check] max |p_multi - sklearn| = {max_diff:.6f}\n")

    # 기여도(표준화 feature 기준)
    contrib_img = X_test_s[:, 0] * beta_img
    contrib_txt = X_test_s[:, 1] * beta_txt

    pred_img = (p_img_only >= 0.5).astype(int)
    pred_multi = (p_multi >= 0.5).astype(int)

    # -----------------------
    # CASE 1: Flip (Image wrong -> Multi correct) + Text contrib positive
    # -----------------------
    print("="*70)
    print("CASE 1: Text FLIPPED prediction (Image wrong → Multi correct)")
    print("="*70)

    flip = np.where((pred_img != y_test) & (pred_multi == y_test) & (contrib_txt > 0))[0]
    if len(flip) > 0:
        idx = flip[np.argmax(contrib_txt[flip])]  # 텍스트가 가장 많이 도운 케이스
        print(f"\nCase #{idx}:")
        print(f"  Label: {y_test[idx]}")
        print(f"  img_logit_pos(raw): {X_test[idx,0]:.4f}  -> p_img={p_img_only[idx]:.4f} (pred={pred_img[idx]})")
        print(f"  txt_logit_pos(raw): {X_test[idx,1]:.4f}")
        print(f"  p_multi={p_multi[idx]:.4f} (pred={pred_multi[idx]})")
        print(f"\n  contrib_img=β_img*z_img: {contrib_img[idx]:+.4f}")
        print(f"  contrib_txt=β_txt*z_txt: {contrib_txt[idx]:+.4f}")
        print(f"  score: {score[idx]:+.4f} (includes bias)")
        print("\n  → Text contribution helped move the decision boundary toward the correct class.")
    else:
        print("  No such case found (with positive text contribution).")

    # -----------------------
    # CASE 2: Stabilize (Image uncertain) + Text contrib positive
    # -----------------------
    print("\n" + "="*70)
    print("CASE 2: Text STABILIZED an uncertain prediction")
    print("="*70)

    uncertain = np.where((np.abs(p_img_only - 0.5) < 0.1) & (contrib_txt > 0))[0]
    if len(uncertain) > 0:
        idx = uncertain[np.argmax(contrib_txt[uncertain])]
        print(f"\nCase #{idx}:")
        print(f"  Label: {y_test[idx]}")
        print(f"  p_img={p_img_only[idx]:.4f} (uncertain), pred_img={pred_img[idx]}")
        print(f"  p_multi={p_multi[idx]:.4f}, pred_multi={pred_multi[idx]}")
        print(f"\n  contrib_img: {contrib_img[idx]:+.4f}")
        print(f"  contrib_txt: {contrib_txt[idx]:+.4f}")
        print("\n  → Text increases confidence / stabilizes the decision.")
    else:
        print("  No such case found.")

    # -----------------------
    # CASE 3: Agree (typical correct) + small text contrib
    # -----------------------
    print("\n" + "="*70)
    print("CASE 3: Both modalities agree (typical)")
    print("="*70)

    agree = np.where((pred_img == y_test) & (pred_multi == y_test))[0]
    if len(agree) > 0:
        # 텍스트 기여가 작은 케이스를 보여주면 'image-dominant' 설명에 좋음
        idx = agree[np.argmin(np.abs(contrib_txt[agree]))]
        print(f"\nCase #{idx}:")
        print(f"  Label: {y_test[idx]}")
        print(f"  p_img={p_img_only[idx]:.4f} (pred={pred_img[idx]})")
        print(f"  p_multi={p_multi[idx]:.4f} (pred={pred_multi[idx]})")
        print(f"  contrib_txt: {contrib_txt[idx]:+.4f}")
        print("\n  → Typical case: decision is mostly image-driven.")
    else:
        print("  No agree case found.")

    # -----------------------
    # SUMMARY
    # -----------------------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    img_acc = (pred_img == y_test).mean()
    multi_acc = (pred_multi == y_test).mean()

    text_helped = np.sum((pred_multi == y_test) & (pred_img != y_test))
    text_hurt = np.sum((pred_multi != y_test) & (pred_img == y_test))

    print(f"\nImage-only accuracy: {img_acc:.3f}")
    print(f"Multi-LR accuracy:   {multi_acc:.3f}")
    print(f"Improvement:         {multi_acc - img_acc:+.3f}")
    print(f"\nText helped: {text_helped} cases")
    print(f"Text hurt:   {text_hurt} cases")
    print(f"Net benefit: {text_helped - text_hurt} cases")

    print("\n✅ Case analysis completed!")


if __name__ == "__main__":
    main()
