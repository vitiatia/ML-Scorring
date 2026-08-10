import pandas as pd

def data_preprocessing(df: pd.DataFrame, category_cols: list = None, binary_cols: list = None) -> pd.DataFrame:

    """
    Функция для предобработки данных:
    - Перевод данных в категориальные колонки
    - Перевод данных в бинарные колонки
    - Перевод данных из int64 в int8 для уменьшения занимаемой памяти

    :param df: DataFrame с исходными данными
    :param category_cols: список категориальных колонок
    :param binary_cols: список бинарных колонок
    :return: DataFrame с предобработанными данными
    """


    for col in df.columns:
        if col in category_cols:
            # Переводим данные в категориальные колонки
            df[col] = df[col].astype('category')
        elif col in binary_cols:
            # Переводим данные из int64 в bool чтобы уменьшить занимаемую ими память
            df[col] = df[col].astype('bool')
        else:
            # Переводим данные из int64 в int8 чтобы уменьшить занимаемую ими память
            if df[col].min() >= -128 and df[col].max() <= 127 and col != 'id':
                df[col] = df[col].astype('int8')
    # в реальности все значения, кроме id лежат в диапазоне от -128 до 127, поэтому можно смело переводить все int64 в int8

    return df


import pandas as pd

def extract_features(df: pd.DataFrame, category_cols: list = None, binary_cols: list = None) -> pd.DataFrame:
    """
    Функция для извлечения и агрегации признаков:
    - Группировка кредитной истории по id (одна строка на клиента)
    - Агрегация категориальных колонок (число уникальных значений и последнее)
    - Агрегация бинарных колонок (сумма и максимум)
    - Агрегация числовых колонок (минимум, среднее и максимум)
    - Преобразование двухуровневых названий колонок в плоский вид

    :param df: DataFrame с предобработанными данными кредитной истории
    :param category_cols: список категориальных колонок
    :param binary_cols: список бинарных колонок
    :return: DataFrame с агрегированными признаками для каждого клиента
    кредитную историю клиента по id в одну строку с агрегированными фичами.
    """
    # Защита от None и ускорение поиска через set O(1)
    cat_set = set(category_cols) if category_cols is not None else set()
    bin_set = set(binary_cols) if binary_cols is not None else set()

    agg_dict = {}
    
    for col in df.columns:
        if col == 'id':
            continue
        if col in cat_set:
            agg_dict[col] = ['nunique', 'last']
        elif col in bin_set:
            agg_dict[col] = ['sum', 'max']
        elif col == 'rn':
            agg_dict[col] = ['max']
        else:
            agg_dict[col] = ['min', 'mean', 'max']

    # Группируем по id и применяем агрегации
    res = df.groupby('id').agg(agg_dict)
    res.columns = [f"{col}_{func}" for col, func in res.columns]
    
    return res.reset_index()