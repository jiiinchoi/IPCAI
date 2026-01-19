"""
BraTS + TextBraTS DataLoader
- Image: 중앙 N장 슬라이스 고정 샘플링 (GT 사용 안함!)
- Text: ClinicalBERT tokenizer
- Mode: 'image' / 'text' / 'multimodal'
"""
import os
import cv2
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class BraTSDataset(Dataset):
    """
    BraTS + TextBraTS Dataset
    
    Args:
        csv_path: train.csv / val.csv / test.csv
        mode: 'image', 'text', 'multimodal'
        modality: 'flair', 't1', 't1ce', 't2'
        n_slices: 샘플링할 슬라이스 개수
        image_size: (H, W) 리사이즈
        tokenizer_name: HuggingFace tokenizer
        max_text_length: 텍스트 최대 길이
    """
    
    def __init__(
        self,
        csv_path: str,
        mode: str = 'multimodal',
        modality: str = 't1ce',
        n_slices: int = 16,
        image_size: tuple = (128, 128),
        tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_text_length: int = 512,
    ):
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.modality = modality
        self.n_slices = n_slices
        self.image_size = image_size
        self.max_text_length = max_text_length
        
        # Tokenizer 로드 (text 모드에서만)
        if mode in ['text', 'multimodal']:
            print(f"Loading tokenizer: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None
        
        # 텍스트 없는 케이스 필터링 (text/multimodal 모드)
        if mode in ['text', 'multimodal']:
            before = len(self.df)
            self.df = self.df[self.df['text'].notna()].copy()
            self.df = self.df[self.df['text'].str.len() > 0].reset_index(drop=True)
            after = len(self.df)
            
            if before != after:
                print(f"⚠️ 텍스트 없는 케이스 {before - after}개 제외")
        
        print(f"Dataset: {len(self.df)} cases | mode={mode} | modality={modality} | n_slices={n_slices}")
    
    def __len__(self):
        return len(self.df)
    
    def _load_and_sample_slices(self, nii_path: str) -> np.ndarray:
        """
        NIfTI 로드 후 중앙 기준 고정 슬라이스 샘플링
        
        Returns:
            (n_slices, H, W) numpy array
        """
        # Load NIfTI
        img = nib.load(nii_path).get_fdata()  # (H, W, D)
        H, W, D = img.shape
        
        # 중앙 기준 슬라이스 선택
        center = D // 2
        start = max(0, center - self.n_slices // 2)
        end = min(D, start + self.n_slices)
        
        # 슬라이스 추출
        slices = img[:, :, start:end]  # (H, W, n)
        
        # 패딩 (n_slices보다 적을 경우)
        if slices.shape[2] < self.n_slices:
            pad_size = self.n_slices - slices.shape[2]
            slices = np.pad(slices, ((0, 0), (0, 0), (0, pad_size)), mode='constant')
        
        slices = slices.transpose(2, 0, 1)  # (n, H, W)
        
        # Resize
        if self.image_size != (H, W):
            resized = []
            for i in range(slices.shape[0]):
                resized.append(cv2.resize(slices[i], self.image_size))
            slices = np.stack(resized, axis=0)
        
        # Z-score 정규화
        mean = slices.mean()
        std = slices.std()
        slices = (slices - mean) / (std + 1e-8)
        
        return slices.astype(np.float32)
    
    def _load_text(self, text_path: str) -> str:
        """텍스트 파일 로드"""
        if not text_path or pd.isna(text_path):
            return ""
        
        text_path = Path(text_path)
        if not text_path.exists():
            return ""
        
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label'])
        
        # IMAGE
        if self.mode in ['image', 'multimodal']:
            nii_path = row[self.modality]
            slices = self._load_and_sample_slices(nii_path)
            slices_tensor = torch.from_numpy(slices)  # (n_slices, H, W)
        
        # TEXT
        if self.mode in ['text', 'multimodal']:
            text = self._load_text(row['text'])
            tokens = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            # Squeeze batch dimension
            tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        
        # Return based on mode
        if self.mode == 'image':
            return slices_tensor, label
        elif self.mode == 'text':
            return tokens, label
        else:  # multimodal
            return slices_tensor, tokens, label


def get_dataloader(
    csv_path: str,
    mode: str = 'multimodal',
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """
    DataLoader 생성 헬퍼 함수
    
    Args:
        csv_path: CSV 파일 경로
        mode: 'image', 'text', 'multimodal'
        batch_size: 배치 크기
        shuffle: 셔플 여부
        num_workers: 워커 수
        **dataset_kwargs: BraTSDataset에 전달할 추가 인자
    
    Returns:
        DataLoader
    """
    dataset = BraTSDataset(csv_path, mode=mode, **dataset_kwargs)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader


# 테스트 코드
if __name__ == "__main__":
    print("==== DataLoader 테스트 ====\n")
    
    # 더미 CSV 생성 (실제로는 data/splits/train.csv 사용)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("case_id,flair,t1,t1ce,t2,seg,text,label\n")
        f.write("test_001,/fake/flair.nii,/fake/t1.nii,/fake/t1ce.nii,/fake/t2.nii,/fake/seg.nii,/fake/text.txt,1\n")
        dummy_csv = f.name
    
    print(f"Dummy CSV: {dummy_csv}")
    print("(실제 사용 시에는 data/splits/train.csv 사용)\n")
    
    # 실제 사용 예시
    print("사용 예시:")
    print("""
    train_loader = get_dataloader(
        csv_path='data/splits/train.csv',
        mode='image',
        batch_size=8,
        num_workers=4,
        modality='t1ce',
        n_slices=16,
        image_size=(128, 128),
    )
    
    for batch in train_loader:
        if mode == 'image':
            images, labels = batch
        elif mode == 'text':
            tokens, labels = batch
        else:  # multimodal
            images, tokens, labels = batch
    """)
    
    print("\n✅ DataLoader 코드 작성 완료!")