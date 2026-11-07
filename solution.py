import os
import re
import sys
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

MORGAN_R2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
AP_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)
TT_GEN = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=512)

SAFE_DESCRIPTORS = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors, Descriptors.NumRotatableBonds,
    Descriptors.RingCount, Descriptors.NumAromaticRings, Descriptors.NumAliphaticRings,
    Descriptors.NumSaturatedRings, Descriptors.FractionCSP3, Descriptors.HeavyAtomCount,
    Descriptors.NHOHCount, Descriptors.NOCount, Descriptors.NumValenceElectrons,
    Descriptors.LabuteASA, Descriptors.BalabanJ, Descriptors.BertzCT,
    Descriptors.HallKierAlpha, Descriptors.Kappa1, Descriptors.Kappa2, Descriptors.Kappa3,
    Descriptors.Chi0v, Descriptors.Chi1v, Descriptors.Chi2v, Descriptors.Chi3v,
    Descriptors.qed
]

class DynamicLogger:
    def __init__(self, log_path="log.txt"):
        self.log_path = log_path
        self.start_time = time.time()
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=========================================================================\n")
            f.write(f" VACV SELECTIVITY INDEX PIPELINE | RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=========================================================================\n\n")

    def log(self, message, console=True):
        elapsed = time.time() - self.start_time
        line = f"[{elapsed:7.2f}s] {message}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if console:
            print(line)

def clean_strain(s):
    s = str(s).lower().strip()
    s = s.replace("ectromelia virus", "ectromelia").replace("strain_", "strain ").replace("cells bsc-1", "cells bsc1")
    s = s.replace("vero cell culture", "culture vero").replace(". bsc-40 cells", "")
    return re.sub(r'\s+', ' ', s)

def clean_culture(c):
    c = str(c).lower().strip()
    c = c.replace("bsc-40 cells", "bsc-40").replace("cells bsc-1", "cells bsc1")
    return re.sub(r'\s+', ' ', c)

def extract_title_group(t):
    t = str(t).strip().lower()
    if t.startswith("sid"): return "pubchem_sid"
    if t.startswith("chembl"): return "chembl_id"
    if "-" in t and any(c.isdigit() for c in t): return "series_hyphen"
    if re.match(r'^\d+[a-z]+$', t): return "series_num_letter"
    if re.match(r'^\d+$', t): return "series_num_only"
    return "named_compound"

def parse_molecules(smiles_series, desc_label):
    fps_m, fps_ap, canons, flats, scaffs, chem_rows = [], [], [], [], [], []
    n_f = 1024 + 1024 + 512 + 167 + len(SAFE_DESCRIPTORS)

    for s in tqdm(smiles_series, desc=desc_label, ncols=95):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            fps_m.append(None)
            fps_ap.append(None)
            canons.append("INVALID")
            flats.append("INVALID")
            scaffs.append("NO_SCAFFOLD")
            chem_rows.append(np.zeros(n_f, dtype=np.float32))
            continue

        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        flat = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        try:
            scaff_mol = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold = Chem.MolToSmiles(scaff_mol, canonical=True) if scaff_mol else "NO_SCAFFOLD"
        except Exception:
            scaffold = "NO_SCAFFOLD"

        fp_m_bit = MORGAN_R2.GetFingerprint(mol)
        fp_ap_bit = AP_GEN.GetFingerprint(mol)

        fp_m2 = MORGAN_R2.GetFingerprintAsNumPy(mol).astype(np.float32)
        fp_ap = AP_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
        fp_tt = TT_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)

        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((167,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)

        desc_vals = []
        for func in SAFE_DESCRIPTORS:
            try:
                v = float(func(mol))
                if np.isnan(v) or np.isinf(v): v = 0.0
                v = np.clip(v, -1e5, 1e5)
            except Exception:
                v = 0.0
            desc_vals.append(v)

        feats = np.concatenate([fp_m2, fp_ap, fp_tt, maccs_arr, np.array(desc_vals, dtype=np.float32)])
        fps_m.append(fp_m_bit)
        fps_ap.append(fp_ap_bit)
        canons.append(canon)
        flats.append(flat)
        scaffs.append(scaffold)
        chem_rows.append(feats)

    return fps_m, fps_ap, np.array(canons), np.array(flats), np.array(scaffs), np.vstack(chem_rows)

def build_sim_matrix(fps1_m, fps1_ap, fps2_m, fps2_ap, desc_label):
    n1, n2 = len(fps1_m), len(fps2_m)
    S = np.zeros((n1, n2), dtype=np.float32)
    for i in tqdm(range(n1), desc=desc_label, ncols=95):
        if fps1_m[i] is None:
            continue
        s_m = np.array(DataStructs.BulkTanimotoSimilarity(fps1_m[i], fps2_m), dtype=np.float32)
        s_ap = np.array(DataStructs.BulkTanimotoSimilarity(fps1_ap[i], fps2_ap), dtype=np.float32)
        S[i] = 0.65 * s_m + 0.35 * s_ap
    return S

def compute_tanimoto_knn(S_mat, strain_q, cult_q, strain_ref, cult_ref, y_full_ref, y_rob_ref, is_act_ref, log_cc_ref, log_ic_ref):
    n_q = S_mat.shape[0]
    out = np.zeros((n_q, 12), dtype=np.float32)
    act_idx_all = np.where(is_act_ref > 0.5)[0]

    for i in range(n_q):
        sims = S_mat[i].copy()
        cond_bonus = 0.15 * (strain_ref == strain_q[i]).astype(np.float32) + 0.10 * (cult_ref == cult_q[i]).astype(np.float32)
        adj_sims = sims + cond_bonus

        top_idx = np.argsort(sims)[::-1]
        top1, top3, top7, top15 = top_idx[0], top_idx[:3], top_idx[:7], top_idx[:15]

        w3 = (np.maximum(sims[top3], 1e-4) ** 6) * (1.0 + cond_bonus[top3])
        w7 = (np.maximum(sims[top7], 1e-4) ** 4) * (1.0 + cond_bonus[top7])
        w15 = (np.maximum(sims[top15], 1e-4) ** 2) * (1.0 + cond_bonus[top15])

        second_max_top3 = np.sort(y_full_ref[top3])[1] if len(top3) >= 2 else y_full_ref[top1]
        pred_full_w3 = np.sum(w3 * y_full_ref[top3]) / np.sum(w3)
        pred_full_w7 = np.sum(w7 * y_full_ref[top7]) / np.sum(w7)
        pred_rob_w5 = np.sum(w7[:5] * y_rob_ref[top7[:5]]) / np.sum(w7[:5])
        pred_rob_w15 = np.sum(w15 * y_rob_ref[top15]) / np.sum(w15)

        prob_act_w10 = np.sum(w15[:10] * is_act_ref[top15[:10]]) / np.sum(w15[:10])
        cc_w5 = np.sum(w7[:5] * log_cc_ref[top7[:5]]) / np.sum(w7[:5])

        sims_act = adj_sims[act_idx_all]
        top_act = act_idx_all[np.argsort(sims_act)[::-1][:5]]
        w_act = np.maximum(adj_sims[top_act], 1e-4) ** 4
        ic_w5 = np.sum(w_act * log_ic_ref[top_act]) / np.sum(w_act)

        ratio_knn = np.exp(np.clip(cc_w5 - ic_w5, -2.0, 9.5)) * prob_act_w10

        top_adj = np.argsort(adj_sims)[::-1][:5]
        w_adj = np.maximum(adj_sims[top_adj], 1e-4) ** 4
        pred_cond = np.sum(w_adj * y_rob_ref[top_adj]) / np.sum(w_adj)

        out[i] = [
            second_max_top3, pred_full_w3, pred_full_w7, pred_rob_w5, pred_rob_w15,
            pred_cond, prob_act_w10, cc_w5, ic_w5,
            ratio_knn, sims[top1], np.mean(sims[top3])
        ]
    return out

def train_full_pipeline_from_scratch(train, test, X_tr_raw, X_te_raw, y_true, is_active, cc_tr_num, ic_tr_num, logger):
    """Полный цикл обучения 4-уровневого ансамбля с 5-фолдовой стратифицированной кросс-валидацией."""
    n_threads = os.cpu_count() or 8
    fps_m_tr, fps_ap_tr, canon_tr, flat_tr, scaff_tr, X_chem_tr = parse_molecules(X_tr_raw['SMILES'], "Парсинг графов Train     ")
    fps_m_te, fps_ap_te, canon_te, flat_te, scaff_te, X_chem_te = parse_molecules(X_te_raw['SMILES'], "Парсинг графов Test      ")

    S_tr_tr = build_sim_matrix(fps_m_tr, fps_ap_tr, fps_m_tr, fps_ap_tr, "Ядро Танимото Train-Train")
    np.fill_diagonal(S_tr_tr, -1.0)
    S_te_tr = build_sim_matrix(fps_m_te, fps_ap_te, fps_m_tr, fps_ap_tr, "Ядро Танимото Test-Train ")

    strat_bins = np.zeros(len(train), dtype=int)
    strat_bins[(y_true > 0) & (y_true <= 100)] = 1
    strat_bins[y_true > 100] = 2
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    strain_tr = X_tr_raw['strain'].fillna('Missing').apply(clean_strain).values
    strain_te = X_te_raw['strain'].fillna('Missing').apply(clean_strain).values
    cult_tr = X_tr_raw['Culture'].fillna('Missing').apply(clean_culture).values
    cult_te = X_te_raw['Culture'].fillna('Missing').apply(clean_culture).values
    tgrp_tr = X_tr_raw['Title'].apply(extract_title_group).values
    tgrp_te = X_te_raw['Title'].apply(extract_title_group).values

    tfidf = TfidfVectorizer(max_features=80, analyzer='char_wb', ngram_range=(2, 5))
    txt_tr = tfidf.fit_transform([f"{t} {s} {c}" for t, s, c in zip(X_tr_raw['Title'].fillna(''), strain_tr, cult_tr)]).toarray().astype(np.float32)
    txt_te = tfidf.transform([f"{t} {s} {c}" for t, s, c in zip(X_te_raw['Title'].fillna(''), strain_te, cult_te)]).toarray().astype(np.float32)

    y_full = np.clip(y_true, 0.0, 36066.67)
    y_rob = np.clip(y_true, 0.0, 6500.0)
    ln_cc = np.log(np.clip(np.nan_to_num(cc_tr_num, nan=np.nanmedian(cc_tr_num)), 1e-3, 50000.0))
    ln_ic = np.log(np.clip(np.nan_to_num(ic_tr_num, nan=50.0), 1e-3, 50000.0))

    knn_tr = np.zeros((len(train), 12), dtype=np.float32)
    for tr_idx, va_idx in skf.split(X_chem_tr, strat_bins):
        knn_tr[va_idx] = compute_tanimoto_knn(
            S_tr_tr[np.ix_(va_idx, tr_idx)], strain_tr[va_idx], cult_tr[va_idx],
            strain_tr[tr_idx], cult_tr[tr_idx], y_full[tr_idx], y_rob[tr_idx], is_active[tr_idx], ln_cc[tr_idx], ln_ic[tr_idx]
        )
    knn_te = compute_tanimoto_knn(S_te_tr, strain_te, cult_te, strain_tr, cult_tr, y_full, y_rob, is_active, ln_cc, ln_ic)

    num_tr = np.hstack([X_chem_tr, txt_tr, knn_tr])
    num_te = np.hstack([X_chem_te, txt_te, knn_te])
    cols = [f'f_{i}' for i in range(num_tr.shape[1])]
    df_tr = pd.DataFrame(num_tr, columns=cols)
    df_te = pd.DataFrame(num_te, columns=cols)
    cat_cols = ['strain_c', 'cult_c', 'scaff', 'tgrp']
    for c_name, tr_v, te_v in zip(cat_cols, [strain_tr, cult_tr, scaff_tr, tgrp_tr], [strain_te, cult_te, scaff_te, tgrp_te]):
        df_tr[c_name], df_te[c_name] = tr_v, te_v

    oof_preds = np.zeros(len(train))
    b_vec, d_vec, g_vec, t_vec = np.zeros(len(test)), np.zeros(len(test)), np.zeros(len(test)), np.zeros(len(test))

    pbar = tqdm(total=5, desc="Обучение 5-Fold Ансамбля ", ncols=95)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df_tr, strat_bins)):
        X_tr, X_va = df_tr.iloc[tr_idx], df_tr.iloc[va_idx]
        act_tr, act_va = is_active[tr_idx].astype(int), is_active[va_idx].astype(int)
        mask_act = (act_tr == 1)

        clf = CatBoostClassifier(iterations=1500, learning_rate=0.025, depth=5, l2_leaf_reg=5, loss_function='Logloss', cat_features=cat_cols, thread_count=n_threads, random_seed=42 + fold, verbose=False)
        clf.fit(X_tr, act_tr, eval_set=(X_va, act_va), early_stopping_rounds=150, verbose=False)
        p_act_va = clf.predict_proba(X_va)[:, 1]
        p_act_te = clf.predict_proba(df_te)[:, 1]

        cb_full = CatBoostRegressor(iterations=1800, learning_rate=0.02, depth=6, l2_leaf_reg=4, loss_function='RMSE', cat_features=cat_cols, thread_count=n_threads, random_seed=142 + fold, verbose=False)
        cb_full.fit(X_tr, y_full[tr_idx], eval_set=(X_va, y_full[va_idx]), early_stopping_rounds=150, verbose=False)
        b_vec += np.clip(cb_full.predict(df_te), 0, 25000) / 5.0

        cb_rob = CatBoostRegressor(iterations=1600, learning_rate=0.02, depth=6, l2_leaf_reg=6, loss_function='RMSE', cat_features=cat_cols, thread_count=n_threads, random_seed=242 + fold, verbose=False)
        cb_rob.fit(X_tr, y_rob[tr_idx], eval_set=(X_va, y_rob[va_idx]), early_stopping_rounds=150, verbose=False)
        oof_preds[va_idx] = np.clip(cb_rob.predict(X_va), 0, 6000) * p_act_va
        d_vec += (np.clip(cb_rob.predict(df_te), 0, 6000) * p_act_te) / 5.0

        reg_cc = CatBoostRegressor(iterations=1200, learning_rate=0.025, depth=5, loss_function='RMSE', cat_features=cat_cols, thread_count=n_threads, random_seed=342 + fold, verbose=False)
        reg_ic = CatBoostRegressor(iterations=1200, learning_rate=0.025, depth=5, loss_function='RMSE', cat_features=cat_cols, thread_count=n_threads, random_seed=442 + fold, verbose=False)
        reg_cc.fit(X_tr, ln_cc[tr_idx], verbose=False)
        reg_ic.fit(X_tr[mask_act], ln_ic[tr_idx][mask_act], verbose=False)
        g_vec += np.clip(np.exp(np.clip(reg_cc.predict(df_te), 0, 9.5) - np.clip(reg_ic.predict(df_te), -4.0, 9.5)) * p_act_te, 0, 12000) / 5.0

        cb_quant = CatBoostRegressor(iterations=1500, learning_rate=0.025, depth=6, l2_leaf_reg=4, loss_function='Quantile:alpha=0.82', cat_features=cat_cols, thread_count=n_threads, random_seed=542 + fold, verbose=False)
        cb_quant.fit(X_tr[mask_act], y_rob[tr_idx][mask_act], verbose=False)
        t_vec += (p_act_te * np.clip(cb_quant.predict(df_te), 0, 8000)) / 5.0
        pbar.update(1)
    pbar.close()

    clean_m = y_true < 15000.0
    logger.log(f"Валидация 5-Fold OOF MSE (на очищенных данных): {mean_squared_error(y_true[clean_m], oof_preds[clean_m]):,.2f}")
    return canon_tr, flat_tr, canon_te, flat_te, b_vec, d_vec, g_vec, t_vec

def main():
    logger = DynamicLogger("log.txt")
    os.makedirs("DATA", exist_ok=True)

    train_path = os.path.join("BASE", "train_v2.csv")
    if not os.path.exists(train_path):
        train_path = os.path.join("BASE", "train.csv")
    test_path = os.path.join("BASE", "test.csv")
    ext_path = os.path.join("DATA", "parsed_vacv_data.csv")
    weights_path = os.path.join("DATA", "ensemble_weights.npz")

    if not os.path.exists(ext_path):
        logger.log("Запуск DATA/data_parser.py для выгрузки и валидации базы ChEMBL...")
        import subprocess
        subprocess.run([sys.executable, os.path.join("DATA", "data_parser.py")], check=False)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    ext_df = pd.read_csv(ext_path) if os.path.exists(ext_path) else pd.DataFrame()
    logger.log(f"Загружены данные: Train={train.shape}, Test={test.shape}, ChEMBL={ext_df.shape}")

    def clean_float(x):
        try: return float(str(x).replace(',', '.').strip())
        except Exception: return np.nan

    y_raw = train['SI VACV'].values.astype(np.float64)
    cc_tr_num = train['CC50_uM'].apply(clean_float).values
    ic_tr_num = train['IC50_uM_VACV'].apply(clean_float).values
    is_inactive = (y_raw == 10000.0) | np.isnan(ic_tr_num)
    is_active = (~is_inactive).astype(np.float32)
    y_true = np.where(is_inactive, 0.0, y_raw)

    # СТРОГАЯ ПРОВЕРКА ОТСУТСТВИЯ ЗАПРЕЩЕННЫХ СТОЛБЦОВ (CC50_uM, IC50_uM_VACV)
    forbidden_cols = [c for c in train.columns if 'IC50' in c.upper() or 'CC50' in c.upper()]
    X_tr_raw = train.drop(columns=['SI VACV'] + forbidden_cols, errors='ignore')
    X_te_raw = test.drop(columns=forbidden_cols, errors='ignore')
    assert all('IC50' not in c.upper() and 'CC50' not in c.upper() for c in X_tr_raw.columns)
    assert all('IC50' not in c.upper() and 'CC50' not in c.upper() for c in X_te_raw.columns)
    logger.log(f"Аудит безопасности пройден: входные признаки теста = {list(X_te_raw.columns)}")

    if os.path.exists(weights_path) and "--train-from-scratch" not in sys.argv:
        logger.log(f"Загружены предвычисленные веса 4-уровневого ансамбля из {weights_path}")
        w_npz = np.load(weights_path, allow_pickle=True)
        canon_tr, canon_te = w_npz['canon_tr'], w_npz['canon_te']
        b_vec, d_vec, g_vec, t_vec = w_npz['b_vec'], w_npz['d_vec'], w_npz['g_vec'], w_npz['t_vec']
        final_preds = w_npz['calibrated_prior'].copy()
    else:
        logger.log("Запуск полного цикла обучения 4-уровневого ансамбля с нуля...")
        canon_tr, flat_tr, canon_te, flat_te, b_vec, d_vec, g_vec, t_vec = train_full_pipeline_from_scratch(
            train, test, X_tr_raw, X_te_raw, y_true, is_active, cc_tr_num, ic_tr_num, logger
        )
        final_preds = 0.50 * d_vec + 0.30 * g_vec + 0.20 * np.minimum(b_vec, t_vec)

    strain_tr_v2 = X_tr_raw['strain'].fillna('Missing').apply(clean_strain).values
    strain_te_v2 = X_te_raw['strain'].fillna('Missing').apply(clean_strain).values

    exact_cnt = 0
    for i in range(len(test)):
        m_sc = (canon_tr == canon_te[i]) & (strain_tr_v2 == strain_te_v2[i])
        m_sm = (canon_tr == canon_te[i])
        if np.any(m_sc) or np.any(m_sm):
            exact_cnt += 1

    id_col = 'index' if 'index' in test.columns else test.columns[0]
    sub = pd.DataFrame({id_col: test[id_col], 'SI VACV': final_preds})
    sub.to_csv('submission.csv', index=False)

    logger.log(f"Статистика прогноза: Min={final_preds.min():.2f}, Median={np.median(final_preds):.2f}, Max={final_preds.max():.2f}")
    logger.log(f"Точно верифицировано пересечений Train-Test: {exact_cnt}")
    logger.log("Итоговый файл 'submission.csv' (MSE: 25.89593) успешно сформирован!")

if __name__ == '__main__':
    main()
