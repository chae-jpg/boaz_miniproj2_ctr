import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader

# === DataLoader / Target 관련 전역 파라미터 ===
TARGET_COL = "target"
BATCH_SIZE = 1024
NUM_WORKERS = 0
PIN_MEMORY = True
SHUFFLE_TRAIN = True

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

"""V1: \
`seq_prod` 정보 X
"""

# =============================================================================
# [V1] Dataset & Loader (Baseline DeepFM: Seq 정보 X)
# =============================================================================
class DatasetV1(Dataset):
    def __init__(self, df: pd.DataFrame, sparse_dims: list):
        super().__init__()
        self.dense = df[DENSE_COLS].astype("float32").values # Continuous values
        self.sparse = df[SPARSE_BASE_COLS].astype("int64").values # Categorical values
        self.label = df[TARGET_COL].astype("float32").values

        # 메타데이터 저장
        self.field_dims = [1] * len(DENSE_COLS) + sparse_dims  # Sparse 각 컬럼의 vocab size

        print(f"[V1 Created] N={len(self.label)}, Sparse Fields={len(self.field_dims)}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        # Xi : Dense(Continuous) - 0, Sparse(Categorical) - each values
        # Xv : Dense(Continuous) - each values, Sparse(Categorical) - 1

        Xi_dense = np.zeros_like(self.dense[idx])
        Xi_sparse = self.sparse[idx]

        Xv_dense = self.dense[idx]
        Xv_sparse = np.ones_like(self.sparse[idx])

        # Shape 맞추기: (N,) -> (N, 1)로 unsqueeze
        xv = torch.tensor(np.concatenate((Xv_dense, Xv_sparse)), dtype=torch.float32).unsqueeze(-1)
        xi = torch.tensor(np.concatenate((Xi_dense, Xi_sparse)), dtype=torch.long).unsqueeze(-1)
        label = torch.tensor(self.label[idx], dtype=torch.float32)

        return {"xi": xi, "xv": xv, "label": label}

def create_v1_dataloaders(train_df, valid_df, test_df, sparse_dims, batch_size=BATCH_SIZE):
    train_ds = DatasetV1(train_df, sparse_dims)
    valid_ds = DatasetV1(valid_df, sparse_dims)
    test_ds  = DatasetV1(test_df, sparse_dims)

    return (
        DataLoader(train_ds, batch_size, shuffle=SHUFFLE_TRAIN, num_workers=NUM_WORKERS),
        DataLoader(valid_ds, batch_size, shuffle=False, num_workers=NUM_WORKERS),
        DataLoader(test_ds, batch_size, shuffle=False, num_workers=NUM_WORKERS)
    )

"""V2 : \
`seq_prod` 5개 컬럼
"""

# =============================================================================
# [V2] Dataset & Loader (Seq as Sparse: Seq 5개를 펼쳐서 sparse feature로 취급)
# =============================================================================
class DatasetV2(Dataset):
    def __init__(self, df: pd.DataFrame, all_sparse_dims: list):
        super().__init__()
        self.dense = df[DENSE_COLS].astype("float32").values

        # Sparse + Seq(5개) 합치기
        all_sparse_cols = SPARSE_BASE_COLS + SEQ_COLS
        self.sparse = df[all_sparse_cols].fillna(0).astype("int64").values
        self.label = df[TARGET_COL].astype("float32").values

        self.field_dims = [1] * len(DENSE_COLS) + all_sparse_dims # Base(6개) + Seq(5개) = 총 11개 필드 크기 정보

        print(f"[V2 Created] N={len(self.label)}, Total Sparse Fields={len(self.field_dims)}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Xi : Dense(Continuous) - 0, Sparse(Categorical) - each values
        # Xv : Dense(Continuous) - each values, Sparse(Categorical) - 1

        Xi_dense = np.zeros_like(self.dense[idx])
        Xi_sparse = self.sparse[idx]

        Xv_dense = self.dense[idx]
        Xv_sparse = np.ones_like(self.sparse[idx])

        # Shape 맞추기: (N,) -> (N, 1)로 unsqueeze
        xv = torch.tensor(np.concatenate((Xv_dense, Xv_sparse)), dtype=torch.float32).unsqueeze(-1)
        xi = torch.tensor(np.concatenate((Xi_dense, Xi_sparse)), dtype=torch.long).unsqueeze(-1)
        label = torch.tensor(self.label[idx], dtype=torch.float32)

        return {"xi": xi, "xv": xv, "label": label}

def create_v2_dataloaders(train_df, valid_df, test_df, all_sparse_dims, batch_size=BATCH_SIZE):
    train_ds = DatasetV2(train_df, all_sparse_dims)
    valid_ds = DatasetV2(valid_df, all_sparse_dims)
    test_ds  = DatasetV2(test_df, all_sparse_dims)

    return (
        DataLoader(train_ds, batch_size, shuffle=SHUFFLE_TRAIN, num_workers=NUM_WORKERS),
        DataLoader(valid_ds, batch_size, shuffle=False, num_workers=NUM_WORKERS),
        DataLoader(test_ds, batch_size, shuffle=False, num_workers=NUM_WORKERS)
    )

"""V3 : \
`seq_prod` GRU 입력
"""

# =============================================================================
# [V3] Dataset & Loader (DeepFM + GRU: Seq 분리 입력)
# =============================================================================
class DatasetV3(Dataset):
    def __init__(self, df: pd.DataFrame, sparse_dims: list):
        super().__init__()
        self.dense = df[DENSE_COLS].astype("float32").values
        self.sparse = df[SPARSE_BASE_COLS].astype("int64").values
        self.seq = df[SEQ_COLS].fillna(0).astype("int64").values # GRU용
        self.label = df[TARGET_COL].astype("float32").values

        self.field_dims = [1] * len(DENSE_COLS) + sparse_dims # Base Sparse Feature만 해당
        self.seq_vocab_size = sparse_dims[0] # product_id_idx의 vocab size (Sequence Embedding용)

        print(f"[V3 Created] N={len(self.label)}, Sparse Fields={len(self.field_dims)}, Seq Len={self.seq.shape[1]}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Xi : Dense(Continuous) - 0, Sparse(Categorical) - each values
        # Xv : Dense(Continuous) - each values, Sparse(Categorical) - 1

        Xi_dense = np.zeros_like(self.dense[idx])
        Xi_sparse = self.sparse[idx]

        Xv_dense = self.dense[idx]
        Xv_sparse = np.ones_like(self.sparse[idx])

        # Shape 맞추기: (N,) -> (N, 1)로 unsqueeze
        xv = torch.tensor(np.concatenate((Xv_dense, Xv_sparse)), dtype=torch.float32).unsqueeze(-1)
        xi = torch.tensor(np.concatenate((Xi_dense, Xi_sparse)), dtype=torch.long).unsqueeze(-1)

        # GRU 시퀀스는 (Length,) 혹은 (Length, 1) 중 모델에 맞춰 사용
        # 여기서는 (Length,) 로 둡니다.
        seq = torch.tensor(self.seq[idx], dtype=torch.long)

        label = torch.tensor(self.label[idx], dtype=torch.float32)

        return {"xi": xi, "xv": xv, "seq": seq, "label": label}

def create_v3_dataloaders(train_df, valid_df, test_df, sparse_dims, batch_size=BATCH_SIZE):
    train_ds = DatasetV3(train_df, sparse_dims)
    valid_ds = DatasetV3(valid_df, sparse_dims)
    test_ds  = DatasetV3(test_df, sparse_dims)

    return (
        DataLoader(train_ds, batch_size, shuffle=SHUFFLE_TRAIN, num_workers=NUM_WORKERS),
        DataLoader(valid_ds, batch_size, shuffle=False, num_workers=NUM_WORKERS),
        DataLoader(test_ds, batch_size, shuffle=False, num_workers=NUM_WORKERS)
    )