import pandas as pd
import sys
import os
import gc
sys.path.append(os.path.abspath('..')) 
from src.EDA.data_preprocessing import data_preprocessing, extract_features
from src.schema import category_cols, binary_cols

def get_data():

    chunks = []

    for n in range(0, 12):
        df_n = pd.read_parquet(f'datasets/train_data/train_data_{n}.pq')
        # Преобразовываем данные для CatBoost
        df_n = extract_features(df=data_preprocessing(df_n, category_cols=category_cols, binary_cols=binary_cols), category_cols=category_cols, binary_cols=binary_cols)
        chunks.append(df_n)

        # Очистка памяти из оперативки
        del df_n
        gc.collect()

    dft = pd.read_csv('datasets/train_target.csv')
    df = pd.concat(chunks, axis=0, ignore_index=True).merge(right=dft, how='left', on='id')

    return df