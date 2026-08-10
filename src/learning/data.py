import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('..')) 
from src.EDA.data_preprocessing import data_preprocessing

category_cols = ['enc_loans_credit_type', 'enc_loans_credit_status', 'enc_loans_account_holder_type', 'enc_loans_impediment_type', 'enc_loans_subject_type'] + [f'enc_paym_{i}' for i in range(25)]
binary_cols = ['pclose_flag', 'fclose_flag', 'is_zero_loans_credit_limit', 'is_zero_loans_next_pay_summ', 'is_zero_loans_outstanding', 'is_zero_loans_total_overdue', 'is_zero_loans_max_overdue_sum', 'is_zero_loans_credit_cost_rate', 'is_zero_loans_5', 'is_zero_loans_5_30', 'is_zero_loans_30_60', 'is_zero_loans_60_90', 'is_zero_loans_90', 'is_zero_util', 'is_zero_overdue_count', 'is_zero_max_overdue_sum']

def get_data():

    for n in range(0, 12):
        df = pd.read_parquet(f'../datasets/train_data/train_data_{n}.pq')
        df = data_preprocessing(df, category_cols=category_cols, binary_cols=binary_cols)

    return None




