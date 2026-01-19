"""
manifest.csv를 train/val/test로 stratified split
ET absent가 적으므로 비율 유지가 중요함
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(
    manifest_csv: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Stratified split으로 ET 비율 유지
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율 합이 1이 아님"
    
    df = pd.read_csv(manifest_csv)
    print(f"총 케이스: {len(df)}개")
    print(f"  - ET present (label=1): {(df['label']==1).sum()}개")
    print(f"  - ET absent  (label=0): {(df['label']==0).sum()}개")
    
    # Stratified split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['label'],
        random_state=seed
    )
    
    # val vs test
    val_test_split = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_test_split),
        stratify=temp_df['label'],
        random_state=seed
    )
    
    # 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n==== Split 완료 ====")
    print(f"저장 위치: {output_dir}")
    
    for name, df_split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pos = (df_split['label'] == 1).sum()
        neg = (df_split['label'] == 0).sum()
        print(f"\n{name}: {len(df_split)}개")
        print(f"  - ET present: {pos}개 ({pos/len(df_split)*100:.1f}%)")
        print(f"  - ET absent:  {neg}개 ({neg/len(df_split)*100:.1f}%)")
        print(f"  - 저장: {output_dir / (name.lower() + '.csv')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv가 없습니다: {manifest_path}")
    
    make_splits(
        manifest_csv=manifest_path,
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()