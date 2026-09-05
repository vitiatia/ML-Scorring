import pandas as pd
import gc
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.EDA.data_preprocessing import data_preprocessing, extract_features
from src.schema import category_cols, binary_cols

def get_data() -> pd.DataFrame:
    chunks = []
    
    data_dir = PROJECT_ROOT / "datasets" / "train_data"
    target_path = PROJECT_ROOT / "datasets" / "train_target.csv"

    for n in tqdm(range(0, 11)):  # 0..10 для train, 11-й батч остаётся для test
        file_path = data_dir / f"train_data_{n}.pq"
        df_n = pd.read_parquet(file_path)
        
        df_n = data_preprocessing(df_n, category_cols=category_cols, binary_cols=binary_cols)
        
        df_n = extract_features(df_n, category_cols=category_cols, binary_cols=binary_cols)
        
        chunks.append(df_n)

        del df_n
        gc.collect()

    dft = pd.read_csv(target_path)
    df = pd.concat(chunks, axis=0, ignore_index=True).merge(dft, how='left', on='id')

    return df