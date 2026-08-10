import pandas as pd

def nans_checker():
    """
    Функция для проверки наличия пропущенных значений в наборах данных train_data_0.pq - train_data_11.pq
    """

    for n in range(0, 12):
        df = pd.read_parquet(f'../datasets/train_data/train_data_{n}.pq').copy()
        if df.isna().sum().sum() == 0:
            print(f"В данных train_data_{n}.pq отсутствуют пропущенные значения.")

def min_max_checker():
    """
    Функция для проверки минимальных и максимальных значений в наборе данных train_data_0.pq - train_data_11.pq без учета колонки id
    """
    min, max = 0, 0
    for n in range(0, 12):
        df = pd.read_parquet(f'../datasets/train_data/train_data_{n}.pq').copy()
        df_no_id = df.drop(columns=['id'])
        if df_no_id.min().min() < min:
            min = df_no_id.min().min()
        if df_no_id.max().max() > max:
            max = df_no_id.max().max()
        print(f'Минимум в датасете train_data_{n} (без учета id): {df_no_id.min().min()}, максимум (без учета id): {df_no_id.max().max()}')
    print(f'Минимум в датасетax (без учета id): {min}, максимум (без учета id): {max}')