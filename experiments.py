import os
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor, CatBoostClassifier

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
AP_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)

def featurize(smiles_list):
    rows = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            rows.append(np.zeros(2054, dtype=np.float32))
            continue
        fp_m = MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
        fp_a = AP_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
        desc = np.array([
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.qed(mol)
        ], dtype=np.float32)
        rows.append(np.concatenate([fp_m, fp_a, np.nan_to_num(desc, 0.0)]))
    return np.vstack(rows)

def run_experiments():
    print("=========================================================================")
    print(" ОТЧЕТ ПО ЭКСПЕРИМЕНТАМ И КРОСС-ВАЛИДАЦИИ МОДЕЛЕЙ (5-FOLD CV)")
    print("=========================================================================")

    train_path = os.path.join("BASE", "train_v2.csv")
    if not os.path.exists(train_path):
        train_path = os.path.join("BASE", "train.csv")
    train = pd.read_csv(train_path)

    y_raw = train['SI VACV'].values.astype(np.float64)
    ic_num = pd.to_numeric(train['IC50_uM_VACV'], errors='coerce').values
    is_inactive = (y_raw == 10000.0) | np.isnan(ic_num)
    is_active = (~is_inactive).astype(int)
    y_clean = np.where(is_inactive, 0.0, y_raw)
    y_cap = np.clip(y_clean, 0.0, 6500.0)

    X = featurize(train['SMILES'])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_mask = y_clean < 15000.0

    oof_raw_cb = np.zeros(len(train))
    oof_clean_cb = np.zeros(len(train))
    oof_rf = np.zeros(len(train))
    oof_et = np.zeros(len(train))
    oof_act_prob = np.zeros(len(train))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, is_active)):
        X_tr, X_va = X[tr_idx], X[va_idx]

        # Эксперимент 1: Базовый CatBoost на сыром таргете (с заглушками 10000)
        m_raw = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, verbose=False, random_seed=42+fold)
        m_raw.fit(X_tr, y_raw[tr_idx])
        oof_raw_cb[va_idx] = m_raw.predict(X_va)

        # Эксперимент 2: CatBoost на очищенном таргете (10000 -> 0.0)
        m_cln = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, verbose=False, random_seed=42+fold)
        m_cln.fit(X_tr, y_cap[tr_idx])
        oof_clean_cb[va_idx] = np.clip(m_cln.predict(X_va), 0, 6500)

        # Эксперимент 3: RandomForest + ExtraTrees
        rf = RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42+fold)
        et = ExtraTreesRegressor(n_estimators=300, max_depth=14, n_jobs=-1, random_state=42+fold)
        rf.fit(X_tr, y_cap[tr_idx])
        et.fit(X_tr, y_cap[tr_idx])
        oof_rf[va_idx] = rf.predict(X_va)
        oof_et[va_idx] = et.predict(X_va)

        # Эксперимент 4: Классификатор активности P(active) для Hurdle-модели
        clf = CatBoostClassifier(iterations=600, learning_rate=0.03, depth=5, verbose=False, random_seed=42+fold)
        clf.fit(X_tr, is_active[tr_idx])
        oof_act_prob[va_idx] = clf.predict_proba(X_va)[:, 1]

    oof_hurdle = oof_act_prob * (0.50 * oof_clean_cb + 0.25 * oof_rf + 0.25 * oof_et)

    print(f"1. Baseline CatBoost (неочищенный таргет с заглушками 10000) | OOF MSE: {mean_squared_error(y_clean[eval_mask], oof_raw_cb[eval_mask]):12,.2f}")
    print(f"2. RandomForest Regressor (Morgan + AtomPair + RDKit 2D)     | OOF MSE: {mean_squared_error(y_clean[eval_mask], oof_rf[eval_mask]):12,.2f}")
    print(f"3. ExtraTrees Regressor (Morgan + AtomPair + RDKit 2D)       | OOF MSE: {mean_squared_error(y_clean[eval_mask], oof_et[eval_mask]):12,.2f}")
    print(f"4. CatBoost Regressor (очищенный таргет 10000 -> 0.0)        | OOF MSE: {mean_squared_error(y_clean[eval_mask], oof_clean_cb[eval_mask]):12,.2f}")
    print(f"5. Two-Stage Hurdle Ensemble P(active) * E[SI | active]      | OOF MSE: {mean_squared_error(y_clean[eval_mask], oof_hurdle[eval_mask]):12,.2f}")
    print(f"6. Качество классификатора вирусной активности ROC-AUC       | OOF AUC: {roc_auc_score(is_active, oof_act_prob):12.4f}")
    print("=========================================================================")

if __name__ == '__main__':
    run_experiments()
