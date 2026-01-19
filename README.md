# MRI + Text Multimodal Fusion for ET Presence Classification (BraTS + TextBraTS)

간단한 ET(Enhancing Tumor) 존재 여부(0/1) 분류 프로젝트입니다.
BraTS(원래 segmentation) 마스크에서 ET(라벨=4) voxel 유무로 케이스 라벨을 생성하고, MRI(t1ce) + 텍스트(TextBraTS)를 late fusion으로 결합해 성능/강건성/텍스트 기여도를 분석했습니다.

## 핵심 아이디어

* **Image-only / Text-only / Multimodal(Weighted, LR)** 비교
* **Limited imaging evidence**: 슬라이스 수 감소(예: 24/12/6) 조건에서 성능 변화(robustness) 확인
* **Text ablation**: 랜덤 토큰 마스킹(0~60%)으로 텍스트가 정보인지/안정화인지 분석
* **Explainability**: LR 계수(β_img, β_txt) 및 케이스 분석(flip/stabilize)로 해석

## 폴더 구조(요약)

* `src/`

  * `data_loader.py` : BraTS/TextBraTS Dataset + DataLoader
  * `models.py` : ResNet18(Image), ClinicalBERT(Text), Late Fusion
  * `train.py` : image/text 학습 + weighted/LR fusion 평가
  * `robustness.py` : n_slices 변화 실험(24/12/6 등)
  * `text_masking.py`, `text_masking_multiseed.py` : 텍스트 마스킹 ablation
  * `case_analysis.py`, `token_attribution.py` : 설명가능성(선택)
* `data/`

  * `manifest.csv` (생성)
  * `splits/` (train/val/test)
* `runs/` : 모델 체크포인트/로그
* `results/` : 실험 결과(json/txt)

## 데이터 준비

1. manifest 생성 (seg에서 ET voxel로 label 생성)
2. stratified split 생성

## 실행 예시

### 1) Image-only 학습

```bash
python src/train.py --mode image --modality t1ce --n_slices 24
```

### 2) Text-only 학습

```bash
python src/train.py --mode text --freeze_bert
```

### 3) Weighted fusion 평가

```bash
python src/train.py --mode weighted --modality t1ce --n_slices 24
```

### 4) LR late fusion 평가

```bash
python src/train.py --mode lr --modality t1ce --n_slices 24
```

### 5) Robustness (예: 24/12/6)

```bash
python src/robustness.py --n_slices_list 24 12 6
```

### 6) Text masking (multi-seed)

```bash
python src/text_masking_multiseed.py --seeds 42 43 44 45 46 --mask_ratios 0.0 0.6 --n_slices 12
```

## Notes

* 결과는 `runs/`(학습 로그/모델) 및 `results/`(실험 결과)에 저장됩니다.
* 본 프로젝트는 “멀티모달이 항상 성능을 올린다”보다, **영상 증거가 제한될 때 텍스트가 어떤 식으로 기여/방해하는지** 분석하는 데 초점을 둡니다.
