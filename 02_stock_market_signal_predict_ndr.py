# =============================================================================
# STOCK MARKET SIGNAL: PREDICT NEXT DAY RETURNS
# Kaggle Competition Solution
# =============================================================================
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/)
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ── IMPORTS ──────────────────────────────────────────────────────────────────
import polars as pl
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')


# ── 1. VERİYİ OKUMA ──────────────────────────────────────────────────────────
df_train = pd.read_csv('/kaggle/input/competitions/stock-market-signal-predict-next-day-returns/train.csv')
df_test  = pd.read_csv('/kaggle/input/competitions/stock-market-signal-predict-next-day-returns/test.csv')


# ── 2. VERİNİN İLK BAKIŞI ────────────────────────────────────────────────────
print("--- Train: İlk 5 Satır ---")
print(df_train.head())

print("\n--- Test: İlk 5 Satır ---")
print(df_test.head())

print("\n--- Sütun Bilgileri ve Eksik Değerler ---")
print(df_train.info())

print("\n--- İstatistiksel Özet ---")
print(df_train.describe())


# ── 3. ÖZELLİK MÜHENDİSLİĞİ ─────────────────────────────────────────────────
def optimize_features(df):
    """
    Finansal sinyaller üzerinde Polars ile hızlı özellik mühendisliği:
      - Volatilite × Momentum etkileşimi
      - RSI × Bollinger Band pozisyonu (trend gücü sinyali)
      - Hacim ivmesi (kısa/uzun vadeli hacim oranları arası fark)
      - Kısa vadeli ortalama getiri
    """
    df_pl = pl.from_pandas(df)

    df_pl = df_pl.with_columns([
        # Volatilite ve Momentum Etkileşimi
        (pl.col('volatility_20d') * pl.col('momentum_20d')).alias('vol_mom_interaction'),

        # Göreceli Güç × Bollinger Konumu  →  trendin hem gücü hem konumu
        (pl.col('rsi_14') * pl.col('bb_position')).alias('rsi_bb_signal'),

        # Hacim İvmesi  →  kısa vadeli hacim artışı uzun vadeye göre
        (pl.col('volume_sma_ratio_10') / (pl.col('volume_sma_ratio_20') + 1e-6)).alias('volume_acceleration'),

        # Kısa Vadeli Ortalama Getiri  →  1 günlük + 5 günlük ortalaması
        ((pl.col('return_1d') + pl.col('return_5d')) / 2).alias('short_term_avg_return'),
    ])

    return df_pl.to_pandas()


# Hedef ve özellik ayrımı
drop_cols = ['id', 'stock_id', 'target']
features  = [c for c in df_train.columns if c not in drop_cols]

X      = optimize_features(df_train)
y      = df_train['target']
X_test = optimize_features(df_test)


# ── 4. MODELLEME: 5-FOLD XGBOOST ─────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=False)

params = {
    'objective'           : 'binary:logistic',
    'eval_metric'         : 'auc',
    'max_depth'           : 8,       # Gürültülü finansal veride aşırı derinliğe gitme
    'learning_rate'       : 0.01,
    'subsample'           : 0.8,
    'colsample_bytree'    : 0.8,
    'min_child_weight'    : 5,       # Overfitting önleme
    'reg_lambda'          : 10,      # L2 Regülarizasyonu
    'early_stopping_rounds': 100,
    'random_state'        : 42,
    'tree_method'         : 'hist',  # GPU/CPU için hızlı histogram yöntemi
    'n_jobs'              : -1,
}

print(f"Eğitim başlıyor... Toplam Örnek: {len(X)}")

oof_preds  = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_va = X.iloc[train_idx][features], X.iloc[val_idx][features]
    y_tr, y_va = y.iloc[train_idx],           y.iloc[val_idx]

    model = xgb.XGBClassifier(**params, n_estimators=1000)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=100,
    )

    fold_preds          = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx]  = fold_preds
    test_preds         += model.predict_proba(X_test[features])[:, 1] / kf.get_n_splits()

    print(f"Fold {fold + 1} AUC: {roc_auc_score(y_va, fold_preds):.5f}")


# ── 5. SONUÇLAR VE SUBMISSION ─────────────────────────────────────────────────
total_auc = roc_auc_score(y, oof_preds)
print(f"\nGenel OOF AUC: {total_auc:.5f}")

submission = pd.DataFrame({'id': df_test['id'], 'target': test_preds})
submission.to_csv('submission.csv', index=False)
print("submission.csv kaydedildi ✓")
