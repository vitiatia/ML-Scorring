'''
Список категориальных и бинарных колонок 
'''

category_cols = [
    'enc_loans_credit_type',
    'enc_loans_credit_status', 
    'enc_loans_account_holder_type', 
    'enc_loans_impediment_type', 
    'enc_loans_subject_type'
    ] + [f'enc_paym_{i}' for i in range(25)]

binary_cols = [
    'pclose_flag', 
    'fclose_flag', 
    'is_zero_loans_credit_limit', 
    'is_zero_loans_next_pay_summ', 
    'is_zero_loans_outstanding', 
    'is_zero_loans_total_overdue', 
    'is_zero_loans_max_overdue_sum', 
    'is_zero_loans_credit_cost_rate', 
    'is_zero_loans_5', 
    'is_zero_loans_5_30', 
    'is_zero_loans_30_60', 
    'is_zero_loans_60_90', 
    'is_zero_loans_90', 
    'is_zero_util', 
    'is_zero_overdue_count', 
    'is_zero_max_overdue_sum'
    ]