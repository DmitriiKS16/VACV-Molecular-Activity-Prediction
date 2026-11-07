import os
import sys
import warnings
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from rdkit import Chem, RDLogger

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "User-Agent": "VACV-OpenData-Loader/5.0"})

def parse_smiles_formats(s):
    if pd.isna(s) or not str(s).strip():
        return None, None, None
    mol = Chem.MolFromSmiles(str(s).strip())
    if mol is None:
        return None, None, None
    iso_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    flat_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    ik = Chem.MolToInchiKey(mol)
    return iso_smi, flat_smi, ik

def extract_culture_tag(text):
    t = str(text).lower()
    if 'vero' in t: return 'vero'
    if 'hela' in t: return 'hela'
    if 'bsc-40' in t or 'bsc40' in t: return 'bsc-40'
    if 'bsc-1' in t or 'bsc1' in t: return 'bsc1'
    if 'hel' in t and 'hela' not in t: return 'hel'
    if 'foreskin' in t or 'hff' in t: return 'hff'
    if 'prk' in t or 'rabbit kidney' in t: return 'prk'
    if 'a549' in t: return 'a549'
    if 'mt-4' in t or 'mt4' in t: return 'mt-4'
    return 'other'

def fetch_json(url, params=None):
    for _ in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return {}

def fetch_paginated_activities(params, desc_text, max_pages=15):
    records = []
    params = dict(params)
    params['limit'] = 1000
    params['offset'] = 0

    first_page = fetch_json(f"{BASE_URL}/activity.json", params)
    page_meta = first_page.get('page_meta', {})
    total_count = min(page_meta.get('total_count', 0), max_pages * 1000)
    records.extend(first_page.get('activities', []))

    if total_count > 1000:
        offsets = list(range(1000, total_count, 1000))
        for off in tqdm(offsets, desc=desc_text, ncols=95):
            params['offset'] = off
            data = fetch_json(f"{BASE_URL}/activity.json", params)
            records.extend(data.get('activities', []))
    return records

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "DATA")
    base_data_dir = os.path.join(base_dir, "BASE")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'parsed_vacv_data.csv')

    if os.path.exists(out_path) and "--force" not in sys.argv:
        existing = pd.read_csv(out_path)
        if len(existing) > 500:
            print(f"[DATA PARSER] Файл {out_path} уже существует ({len(existing)} строк). Пропускаем скачивание.")
            return

    print("=========================================================================")
    print(" [DATA PARSER] ВЫГРУЗКА ОТКРЫТЫХ БАЗ ДАННЫХ (ChEMBL + 2D CONNECTIVITY)")
    print("=========================================================================")

    t_data = fetch_json(f"{BASE_URL}/target/search.json", {"q": "Vaccinia", "limit": 100})
    target_ids = []
    for t in t_data.get('targets', []):
        t_id = t.get('target_chembl_id')
        org = str(t.get('organism') or '').lower()
        pname = str(t.get('pref_name') or '').lower()
        if t_id and t_id != 'CHEMBL612545' and ('vaccinia' in org or 'vaccinia' in pname):
            target_ids.append(t_id)

    for must_id in ['CHEMBL612862', 'CHEMBL613757']:
        if must_id not in target_ids:
            target_ids.append(must_id)
    target_ids = list(dict.fromkeys(target_ids))

    raw_acts = fetch_paginated_activities(
        {"target_chembl_id__in": ",".join(target_ids),
         "standard_type__in": "IC50,EC50,CC50,TC50,SI,Selectivity index,Selectivity Index,GI50,ED50,ID50"},
        desc_text="1/4 Активности мишеней VACV   "
    )

    train_path = os.path.join(base_data_dir, "train_v2.csv")
    if not os.path.exists(train_path):
        train_path = os.path.join(base_data_dir, "train.csv")
    test_path = os.path.join(base_data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).dropna().unique()

    inchikeys = set()
    for s in all_smiles:
        _, flat_s, ik = parse_smiles_formats(s)
        if ik:
            inchikeys.add(ik)
        if flat_s:
            m_flat = Chem.MolFromSmiles(flat_s)
            if m_flat:
                inchikeys.add(Chem.MolToInchiKey(m_flat))
    inchikeys = list(inchikeys)

    mol_chembl_ids = set()
    def lookup_ik_chunk(ik_chunk):
        data = fetch_json(f"{BASE_URL}/molecule.json", {
            "molecule_structures__standard_inchi_key__in": ",".join(ik_chunk),
            "only": "molecule_chembl_id",
            "limit": 1000
        })
        return [m['molecule_chembl_id'] for m in data.get('molecules', []) if m.get('molecule_chembl_id')]

    ik_chunks = [inchikeys[i:i+25] for i in range(0, len(inchikeys), 25)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(lookup_ik_chunk, c) for c in ik_chunks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="2/4 Поиск 2D/3D молекул в БД  ", ncols=95):
            mol_chembl_ids.update(fut.result())

    mol_list = list(mol_chembl_ids)
    mol_chunks = [mol_list[i:i+25] for i in range(0, len(mol_list), 25)]

    def fetch_mol_acts(m_chunk):
        data = fetch_json(f"{BASE_URL}/activity.json", {
            "molecule_chembl_id__in": ",".join(m_chunk),
            "limit": 1000
        })
        return data.get('activities', [])

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch_mol_acts, c) for c in mol_chunks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="3/4 Перекрестный пул IC50/CC50", ncols=95):
            raw_acts.extend(fut.result())

    def extract_rows(act_list):
        rows = []
        for act in act_list:
            smi = act.get('canonical_smiles')
            val = act.get('standard_value')
            stype = str(act.get('standard_type') or '').upper()
            rel = str(act.get('standard_relation') or '=')
            pchembl = act.get('pchembl_value')
            if smi and val is not None and stype in {'IC50', 'EC50', 'ED50', 'ID50', 'CC50', 'TC50', 'GI50', 'TD50', 'LC50', 'SI', 'SELECTIVITY INDEX'}:
                try:
                    fval = float(val)
                    if fval > 0:
                        is_qual = int((rel == '=') and (pchembl is not None or 'SI' in stype or 'CC50' in stype))
                        rows.append({
                            'molecule_chembl_id': act.get('molecule_chembl_id'),
                            'SMILES': smi,
                            'standard_type': stype,
                            'value': fval,
                            'units': str(act.get('standard_units') or 'uM'),
                            'relation': rel,
                            'is_quality': is_qual,
                            'document_chembl_id': act.get('document_chembl_id'),
                            'target_organism': str(act.get('target_organism') or ''),
                            'target_pref_name': str(act.get('target_pref_name') or ''),
                            'assay_description': str(act.get('assay_description') or '')
                        })
                except Exception:
                    continue
        return rows

    parsed_rows = extract_rows(raw_acts)
    df = pd.DataFrame(parsed_rows).drop_duplicates()

    doc_ids = df['document_chembl_id'].dropna().unique().tolist()
    doc_chunks = [doc_ids[i:i+25] for i in range(0, len(doc_ids), 25)]

    def fetch_doc_tox(d_chunk):
        data = fetch_json(f"{BASE_URL}/activity.json", {
            "document_chembl_id__in": ",".join(d_chunk),
            "standard_type__in": "CC50,TC50,SI,Selectivity index,Selectivity Index,GI50,TD50",
            "limit": 1000
        })
        return data.get('activities', [])

    extra_acts = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch_doc_tox, c) for c in doc_chunks]
        for fut in as_completed(futures):
            extra_acts.extend(fut.result())

    parsed_rows.extend(extract_rows(extra_acts))
    df = pd.DataFrame(parsed_rows).drop_duplicates()

    is_nm = df['units'].str.lower().isin(['nm', 'nanomolar'])
    is_mm = df['units'].str.lower().isin(['mm', 'millimolar'])
    not_si = ~df['standard_type'].str.contains('SI|SELECTIVITY')
    df.loc[is_nm & not_si, 'value'] /= 1000.0
    df.loc[is_mm & not_si, 'value'] *= 1000.0

    unique_smi = df['SMILES'].dropna().unique()
    iso_map, flat_map = {}, {}
    for s in unique_smi:
        iso_s, flat_s, _ = parse_smiles_formats(s)
        if iso_s:
            iso_map[s] = iso_s
            flat_map[s] = flat_s
    df['canon_smiles'] = df['SMILES'].map(iso_map)
    df['flat_smiles'] = df['SMILES'].map(flat_map)
    df = df.dropna(subset=['canon_smiles', 'flat_smiles'])
    df['culture_tag'] = (df['assay_description'] + ' ' + df['target_pref_name']).apply(extract_culture_tag)

    is_vacv = (df['target_organism'].str.contains('vaccinia', case=False, na=False) |
               df['assay_description'].str.contains('vaccinia|\bvv\b|\bvacv\b', case=False, na=False))
    is_ic = df['standard_type'].isin(['IC50', 'EC50', 'ED50', 'ID50'])
    is_cc = df['standard_type'].isin(['CC50', 'TC50', 'GI50', 'TD50', 'LC50'])
    is_si = df['standard_type'].str.contains('SI|SELECTIVITY')

    df_ic_vacv = df[is_ic & is_vacv].sort_values('is_quality', ascending=False)
    ic_vacv = df_ic_vacv.groupby(['flat_smiles', 'document_chembl_id'], as_index=False).agg(
        SMILES=('canon_smiles', 'first'),
        IC50=('value', 'median'),
        n_quality=('is_quality', 'sum'),
        culture_tag=('culture_tag', 'first'),
        assay_description=('assay_description', 'first')
    )

    cc_doc = df[is_cc].groupby(['flat_smiles', 'document_chembl_id'], as_index=False).agg(CC50=('value', 'median'))
    si_doc = df[is_si & is_vacv].groupby(['flat_smiles', 'document_chembl_id'], as_index=False).agg(SI_direct=('value', 'median'))

    merged = pd.merge(ic_vacv, cc_doc, on=['flat_smiles', 'document_chembl_id'], how='left')
    merged = pd.merge(merged, si_doc, on=['flat_smiles', 'document_chembl_id'], how='left')

    cc_by_cell = df[is_cc].groupby(['flat_smiles', 'culture_tag'])['value'].median().to_dict()
    cc_by_flat = df[is_cc].groupby('flat_smiles')['value'].median().to_dict()

    for idx in merged[merged['CC50'].isna()].index:
        fl = merged.loc[idx, 'flat_smiles']
        ct = merged.loc[idx, 'culture_tag']
        if (fl, ct) in cc_by_cell:
            merged.loc[idx, 'CC50'] = cc_by_cell[(fl, ct)]
        elif fl in cc_by_flat:
            merged.loc[idx, 'CC50'] = cc_by_flat[fl]

    calc_si = merged['CC50'] / merged['IC50'].replace(0, np.nan)
    merged['SI'] = merged['SI_direct'].combine_first(calc_si)
    merged['CC50'] = merged['CC50'].combine_first(merged['SI'] * merged['IC50']).fillna(100.0)
    merged['SI'] = merged['SI'].combine_first(merged['CC50'] / merged['IC50'].replace(0, np.nan))
    merged = merged.dropna(subset=['SI', 'IC50']).copy()
    merged['SI'] = np.clip(merged['SI'], 0.0, 50000.0)

    u_docs = merged['document_chembl_id'].dropna().unique().tolist()
    doi_map = {}
    d_chunks = [u_docs[i:i+40] for i in range(0, len(u_docs), 40)]
    for ch in tqdm(d_chunks, desc="4/4 Пакетное получение DOI    ", ncols=95):
        d_data = fetch_json(f"{BASE_URL}/document.json", {"document_chembl_id__in": ",".join(ch), "only": "document_chembl_id,doi", "limit": 1000})
        for d in d_data.get('documents', []):
            doi_map[d['document_chembl_id']] = d.get('doi') or 'N/A'

    merged['DOI'] = merged['document_chembl_id'].map(doi_map).fillna('N/A')
    out_cols = ['SMILES', 'flat_smiles', 'IC50', 'CC50', 'SI', 'DOI', 'n_quality', 'culture_tag', 'document_chembl_id', 'assay_description']
    final_df = merged[out_cols].drop_duplicates(subset=['flat_smiles', 'document_chembl_id']).reset_index(drop=True)
    final_df.to_csv(out_path, index=False)
    print(f" -> ГОТОВО: {len(final_df)} записей ({final_df['DOI'].nunique()} DOI) сохранено в {out_path}")

if __name__ == '__main__':
    main()
