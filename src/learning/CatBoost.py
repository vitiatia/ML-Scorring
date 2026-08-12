import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_model(df: pd.DataFrame, target_col: str):

    X, y = df.drop(columns=[target_col, 'id']), df[target_col]
    # Все колонки в датасете гарантированно приведены к category и bool
    cat_features = list(X.select_dtypes(include=['category', 'bool']).columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=0.8, 
        random_state=42, 
        stratify=y, 
        shuffle=True
        )

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    early_stopping_rounds=100,
    verbose=100,
    task_type='GPU'
    )

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)
    val_preds = model.predict_proba(test_pool)[:, 1]
    val_auc = roc_auc_score(y_test, val_preds)

    return model, val_auc