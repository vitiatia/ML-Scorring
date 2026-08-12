import os
import sys
sys.path.append(os.path.abspath('..')) 

from learning.CatBoost import train_model
from learning.data import get_data

df = get_data()
model, val_auc = train_model(df=df, target_col='flag')
