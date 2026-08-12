import os
import sys
sys.path.append(os.path.abspath('..')) 

from src.learning.CatBoost import train_model
from src.learning.data import get_data

def main():

    df = get_data()
    model, val_auc = train_model(df=df, target_col='flag')

    os.makedirs('models', exist_ok=True)
    model.save_model('models/catboost_model.cbm')
    print(f'ROC-AUC value: {val_auc}')

if __name__ == '__main__':
    main()
