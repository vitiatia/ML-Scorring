import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_model(df: pd.DataFrame, target_col: str):
    X, y = df.drop(columns=[target_col, 'id']), df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=0.8, 
        random_state=42, 
        stratify=y, 
        shuffle=True
        )

    train_pool = Pool(X_train, y_train, cat_features=None)
    test_pool = Pool(X_test, y_test, cat_features=None)

    model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    early_stopping_rounds=100,
    verbose=100
    )

    model.fit(train_pool, eval_set=test_pool)
    val_preds = model.predict_proba(X_test)[:, 1]
    print(f"Validation ROC-AUC: {roc_auc_score(y_test, val_preds):.4f}")
