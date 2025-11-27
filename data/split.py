import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ast

# === Global Settings ===
FILE_DIR = "./"
SAVE_NAME = "ID_Index_mappings.pkl"  # ID-Index 매핑테이블
FILE_NAME = "final_preprocessed_df.parquet" # 최종 전처리 파일 (train/test 분리 이전)

# === Train / Valid / Test Split Ratios (전역 파라미터) ===
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO  = 0.1

# === Feature Column 정의 (현재 final_preprocessed_df 기준) ===
DENSE_COLS = [
    "is_weekend",
    "price_scaled",
    "session_rank_scaled",
    "hour_scaled",
    "day_of_week_scaled",
]

SPARSE_BASE_COLS = [
    "product_id_idx",
    "category_id_idx",
    "category_code_idx",
    "brand_idx",
    "user_id_idx",
    "user_session_idx",
]

SEQ_COLS = [
    "seq_prod_1_idx",
    "seq_prod_2_idx",
    "seq_prod_3_idx",
    "seq_prod_4_idx",
    "seq_prod_5_idx",
]

ORIGINAL_SEQ_COL = "seq_prod"  # 디버깅용으로 남겨뒀어요

"""$\text{Train}/\text{Test}$ 분리"""

def split_exact_tvt_from_preprocessed(
    parquet_path: str,
    random_state=42,
):
    """
    전역변수: FILE_DIR, PREPROCESSED_FILENAME, TRAIN_RATIO, VALID_RATIO, TEST_RATIO를 사용해
    parquet 파일을 자동으로 읽고 정확한 비율 split을 수행하는 함수.
    """

    # 비율 합 확인
    assert abs((TRAIN_RATIO + VALID_RATIO + TEST_RATIO) - 1.0) < 1e-6, \
        "TRAIN_RATIO + VALID_RATIO + TEST_RATIO must equal 1."

    # Load parquet
    print(f"[INFO] Loading preprocessed parquet from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    n = len(df)

    # === Test: 최신 TEST_RATIO 만큼 뒤쪽 ===
    test_start = int(n * (1 - TEST_RATIO))
    df_test = df.iloc[test_start:].reset_index(drop=True)
    df_early = df.iloc[:test_start].reset_index(drop=True)

    # === 전체 기준 정확한 train/valid 개수 ===
    exact_train_n = int(n * TRAIN_RATIO)
    exact_valid_n = int(n * VALID_RATIO)

    # === df_early에서 train/valid 랜덤 샘플링 ===
    df_train = df_early.sample(n=exact_train_n, random_state=random_state)
    df_valid = df_early.drop(df_train.index).sample(n=exact_valid_n, random_state=random_state)

    # reset index
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    print(f"[INFO] TOTAL: {n:,}")
    print(f"[INFO] Train: {len(df_train):,} ({TRAIN_RATIO*100:.1f}%)")
    print(f"[INFO] Valid: {len(df_valid):,} ({VALID_RATIO*100:.1f}%)")
    print(f"[INFO] Test : {len(df_test):,} ({TEST_RATIO*100:.1f}%)")

    return df_train, df_valid, df_test

def extract_dims(train_df: pd.DataFrame):
    # (1) Base Sparse Dims: 기본 6개 범주형 컬럼의 최대 인덱스+1
    base_sparse_dims = [int(train_df[col].max() + 1) for col in SPARSE_BASE_COLS]

    # (2) Product Vocab: 시퀀스 컬럼과 상품ID 컬럼 통틀어 가장 큰 값+1
    max_prod_idx = train_df[SEQ_COLS].max().max()
    if "product_id_idx" in train_df.columns:
        max_prod_idx = max(max_prod_idx, train_df["product_id_idx"].max())
    product_vocab_size = int(max_prod_idx + 1)

    # (3) V2용 차원: Base(6개) + Seq(5개)
    v2_sparse_dims = base_sparse_dims + [product_vocab_size] * 5

    return base_sparse_dims, v2_sparse_dims