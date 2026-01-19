# make_manifest.py
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError as e:
    raise ImportError("nibabel이 필요합니다. pip install nibabel") from e


MODALITY_KEYS = ["flair", "t1", "t1ce", "t2"]
SEG_KEY = "seg"

# BraTS segmentation: 0=bg, 1=NET/NCR, 2=ED, 4=ET
ET_LABEL_VALUE = 4


def _infer_case_id_from_filename(p: Path) -> str:
    """
    파일명에서 케이스 ID 추출.
    예: BraTS20_Training_001_flair.nii -> BraTS20_Training_001
    """
    name = p.name
    if name.endswith(".nii.gz"):
        base = name[:-7]
    elif name.endswith(".nii"):
        base = name[:-4]
    else:
        base = p.stem

    for k in MODALITY_KEYS + [SEG_KEY]:
        tail = f"_{k}"
        if base.endswith(tail):
            return base[: -len(tail)]
    return base


def _scan_image_cases(image_root: Path) -> dict:
    """
    image_root 아래에서 *_flair.nii 같은 파일들을 찾아 케이스 단위로 모음.
    반환: {case_id: {"flair": Path, "t1": Path, "t1ce": Path, "t2": Path, "seg": Path}}
    """
    cases = {}
    nii_files = list(image_root.rglob("*.nii.gz")) + list(image_root.rglob("*.nii"))
    for p in nii_files:
        name = p.name.lower()

        found_key = None
        for k in MODALITY_KEYS + [SEG_KEY]:
            # 파일 끝이 _flair.nii(.gz) / _seg.nii(.gz) 형태인지 확인
            if name.endswith(f"_{k}.nii.gz") or name.endswith(f"_{k}.nii"):
                found_key = k
                break

        if found_key is None:
            continue

        case_id = _infer_case_id_from_filename(p)
        cases.setdefault(case_id, {})
        cases[case_id][found_key] = p

    return cases


def _pick_text_path(text_root: Path, case_id: str) -> str:
    """
    TextBraTSData/<case_id>/ 안에서 {case_id}_*_text.txt 우선 선택.
    없으면 txt 아무거나 선택.
    없으면 빈 문자열.
    """
    case_dir = text_root / case_id
    if not case_dir.exists() or not case_dir.is_dir():
        return ""

    cands = sorted(case_dir.glob(f"{case_id}_*_text.txt"))
    if cands:
        return str(cands[0])

    cands = sorted(case_dir.rglob("*.txt"))
    if cands:
        return str(cands[0])

    return ""


def _load_seg_unique_and_et_count(seg_path: Path) -> tuple[list, int]:
    seg = nib.load(str(seg_path)).get_fdata()
    seg = np.asarray(seg).astype(np.int16)
    uniq = np.unique(seg).tolist()
    et_voxels = int(np.sum(seg == ET_LABEL_VALUE))
    return uniq, et_voxels


def write_manifest(
    out_csv: Path,
    cases: dict,
    text_root: Path,
    seed: int = 42,
    sample_print: int = 5,
) -> None:
    rows = []
    missing = 0

    case_ids = sorted(cases.keys())
    for cid in case_ids:
        item = cases[cid]
        ok = all(k in item for k in MODALITY_KEYS + [SEG_KEY])
        if not ok:
            missing += 1
            continue

        uniq, et_voxels = _load_seg_unique_and_et_count(item[SEG_KEY])
        label = 1 if et_voxels > 0 else 0

        text_path = _pick_text_path(text_root, cid)

        rows.append(
            {
                "case_id": cid,
                "flair": str(item["flair"]),
                "t1": str(item["t1"]),
                "t1ce": str(item["t1ce"]),
                "t2": str(item["t2"]),
                "seg": str(item["seg"]),
                "text": text_path,   # 없으면 ""
                "et_voxels": et_voxels,
                "seg_unique": ",".join(map(lambda x: str(int(x)), uniq)),
                "label": label,
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "flair",
                "t1",
                "t1ce",
                "t2",
                "seg",
                "text",
                "et_voxels",
                "seg_unique",
                "label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    labels = [r["label"] for r in rows]
    pos = int(sum(labels))
    neg = int(len(labels) - pos)

    print("==== Manifest 생성 완료 ====")
    print(f"저장 경로: {out_csv}")
    print(f"완전한 케이스 수(4모달+seg): {len(rows)}")
    print(f"누락/불완전 케이스 스킵: {missing}")
    print(f"ET present(label=1): {pos}")
    print(f"ET absent (label=0): {neg}")
    if len(rows) > 0:
        print(f"양성 비율: {pos/len(rows):.4f}")

    random.seed(seed)
    print("\n==== sanity-check 샘플(seg unique / et_voxels / text 존재) ====")
    for r in random.sample(rows, k=min(sample_print, len(rows))):
        exists = Path(r["text"]).exists() if r["text"] else False
        print(
            f"- {r['case_id']}: label={r['label']} et_voxels={r['et_voxels']} "
            f"seg_unique=[{r['seg_unique']}] text_exists={exists}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, default="data/ImageBraTSData")
    parser.add_argument("--text_root", type=str, default="data/TextBraTSData")
    parser.add_argument("--out", type=str, default="data/manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_print", type=int, default=5)
    args = parser.parse_args()

    image_root = Path(args.image_root)
    text_root = Path(args.text_root)
    out_csv = Path(args.out)

    if not image_root.exists():
        raise FileNotFoundError(f"image_root 경로가 없습니다: {image_root}")

    print(f"[scan] image_root={image_root}")
    cases = _scan_image_cases(image_root)
    print(f"[scan] 발견한 case 후보 수: {len(cases)}")

    print(f"[scan] text_root={text_root}")

    write_manifest(out_csv, cases, text_root, seed=args.seed, sample_print=args.sample_print)


if __name__ == "__main__":
    main()
