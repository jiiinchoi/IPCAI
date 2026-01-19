"""
Token Attribution Analysis
텍스트 모델에서 high-weight 토큰 추출
Low-information 증거 확보
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models import TextModel
from utils.metrics import set_seed, torch_load_weights


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def analyze_token_importance(
    model,
    tokenizer,
    texts,
    device,
    n_samples=50
):
    """
    간단한 Gradient-based token attribution
    """
    model.eval()
    
    # 토큰별 gradient 누적
    token_gradients = Counter()
    token_counts = Counter()
    
    for i, text in enumerate(texts[:n_samples]):
        if i % 10 == 0:
            print(f"  Processing {i}/{n_samples}...")
        
        # Tokenize
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        
        # Enable gradient for input_ids directly
        input_ids.requires_grad = False  # Keep False for embeddings lookup
        
        # Get embeddings and detach/clone to make it a leaf variable
        embeddings = model.bert.embeddings.word_embeddings(input_ids)
        embeddings = embeddings.detach().clone()
        embeddings.requires_grad = True
        
        # Forward
        outputs = model.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = model.classifier(pooled)
        
        # Positive class score
        score = logits[0, 1]
        
        # Backward
        model.zero_grad()
        score.backward()
        
        # Get gradient magnitude per token
        grad_mag = embeddings.grad.abs().sum(dim=-1).squeeze()  # (seq_len,)
        
        # Extract valid tokens
        valid_tokens = attention_mask.squeeze().cpu().numpy()
        input_ids_cpu = input_ids.squeeze().cpu().numpy()
        grad_mag_cpu = grad_mag.detach().cpu().numpy()
        
        for j, (token_id, valid, grad) in enumerate(zip(input_ids_cpu, valid_tokens, grad_mag_cpu)):
            if valid == 1 and token_id not in [0, 101, 102]:  # PAD, CLS, SEP 제외
                token = tokenizer.decode([token_id])
                token_gradients[token] += float(grad)
                token_counts[token] += 1
    
    # Average gradients
    token_importance = {}
    for token, total_grad in token_gradients.items():
        count = token_counts[token]
        token_importance[token] = total_grad / count
    
    # Sort by importance
    sorted_tokens = sorted(token_importance.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_tokens


def categorize_tokens(tokens):
    """토큰을 카테고리로 분류"""
    
    # 템플릿/일반 문구
    template_words = {
        'patient', 'shows', 'demonstrated', 'consistent', 'findings',
        'examination', 'study', 'images', 'sequences', 'protocol',
        'with', 'without', 'noted', 'seen', 'present', 'absent',
        'no', 'evidence', 'suggestive', 'compatible', 'indicative'
    }
    
    # 종양 관련
    tumor_words = {
        'tumor', 'mass', 'lesion', 'enhancing', 'enhancement',
        'necrosis', 'necrotic', 'cyst', 'cystic', 'solid',
        'heterogeneous', 'homogeneous', 'infiltrative', 'circumscribed',
        'glioma', 'glioblastoma', 'astrocytoma', 'oligodendroglioma',
        'metastasis', 'metastatic'
    }
    
    # ET 관련 (우리 task)
    et_words = {
        'enhancing', 'enhancement', 'contrast', 'gadolinium',
        'active', 'viable', 'tumor'
    }
    
    categories = {
        'template': [],
        'tumor_general': [],
        'et_specific': [],
        'other': []
    }
    
    for token, score in tokens:
        token_lower = token.lower().strip()
        
        if token_lower in et_words:
            categories['et_specific'].append((token, score))
        elif token_lower in tumor_words:
            categories['tumor_general'].append((token, score))
        elif token_lower in template_words:
            categories['template'].append((token, score))
        else:
            categories['other'].append((token, score))
    
    return categories


def main():
    set_seed(42)
    device = get_device()
    
    print("\n" + "="*70)
    print("  Token Attribution Analysis")
    print("="*70)
    
    # Load model
    txt_ckpt = ROOT / "runs/text/models/best_text.pt"
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    model = TextModel(
        num_classes=2,
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        freeze_bert=False  # Need gradients
    ).to(device)
    
    model.load_state_dict(torch_load_weights(txt_ckpt, device))
    
    print(f"✅ Model loaded\n")
    
    # Load texts
    df = pd.read_csv(ROOT / "data/splits/test.csv")
    df = df[df['text'].notna()]
    
    texts = []
    for text_path in df['text']:
        full_path = ROOT / text_path
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
    
    print(f"Loaded {len(texts)} texts from test set\n")
    
    # Analyze
    print("Computing token importance...")
    token_scores = analyze_token_importance(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        n_samples=min(50, len(texts))
    )
    
    print("\n" + "="*70)
    print("TOP 20 HIGH-IMPORTANCE TOKENS")
    print("="*70)
    print(f"{'Rank':<6} {'Token':<30} {'Avg Gradient':<15}")
    print("-"*70)
    
    for i, (token, score) in enumerate(token_scores[:20], 1):
        print(f"{i:<6} {token:<30} {score:<15.6f}")
    
    # Categorize
    categories = categorize_tokens(token_scores[:50])
    
    print("\n" + "="*70)
    print("TOKEN CATEGORIZATION (Top 50)")
    print("="*70)
    
    for cat_name, cat_tokens in categories.items():
        print(f"\n{cat_name.upper()} ({len(cat_tokens)} tokens):")
        for token, score in cat_tokens[:5]:
            print(f"  - {token:<20} {score:.6f}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    n_template = len(categories['template'])
    n_et = len(categories['et_specific'])
    n_tumor = len(categories['tumor_general'])
    
    total_top50 = n_template + n_et + n_tumor + len(categories['other'])
    
    print(f"\nTop 50 토큰 구성:")
    print(f"  Template/Generic: {n_template} ({n_template/50*100:.1f}%)")
    print(f"  ET-specific:      {n_et} ({n_et/50*100:.1f}%)")
    print(f"  Tumor-general:    {n_tumor} ({n_tumor/50*100:.1f}%)")
    print(f"  Other:            {len(categories['other'])} ({len(categories['other'])/50*100:.1f}%)")
    
    if n_template > n_et + n_tumor:
        print("\n✅ LOW-INFORMATION CONFIRMED")
        print("   → High-weight 토큰이 대부분 템플릿/일반 문구")
        print("   → ET-specific 정보는 미미")
        print("   → 'Low-information' 주장 지지!")
    else:
        print("\n⚠️  SOME TASK-RELEVANT INFORMATION")
        print("   → 일부 종양/ET 관련 토큰 존재")
        print("   → 'Weak but not zero information'")
    
    # Save
    output_path = ROOT / "results/interpretability/token_attribution.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("TOP 50 HIGH-IMPORTANCE TOKENS\n")
        f.write("="*70 + "\n\n")
        for i, (token, score) in enumerate(token_scores[:50], 1):
            f.write(f"{i}. {token:<30} {score:.6f}\n")
        
        f.write("\n\nCATEGORIZATION\n")
        f.write("="*70 + "\n\n")
        for cat_name, cat_tokens in categories.items():
            f.write(f"{cat_name.upper()}:\n")
            for token, score in cat_tokens:
                f.write(f"  - {token:<20} {score:.6f}\n")
            f.write("\n")
    
    print(f"\n✅ Results saved to: {output_path}")
    print("\n✅ Analysis completed!")


if __name__ == "__main__":
    main()