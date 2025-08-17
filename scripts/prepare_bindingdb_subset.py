# scripts/prepare_bindingdb_subset.py
# Convert a BindingDB TSV (zipped) into the 4 CSVs your pipeline expects under data/real/.
# - Reads Ki (nM) records, aggregates median per (drug, target)
# - Maps drugs to PubChem CID or MonomerID; targets to UniProt Primary ID when available
# - Builds simple gene & "cell" embeddings via PCA
# - Creates labels: positive if Ki <= 100 nM, negative if Ki >= 10000 nM (sampled)

import re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ---- CONFIG ----
IN_ZIP = Path("data/bindingdb/raw/BindingDB_PDSPKi_202508_tsv.zip")   # change if you picked another file
OUT = Path("data/real"); OUT.mkdir(parents=True, exist_ok=True)

# Column name helpers (BindingDB TSV schema)
COL_KI = "Ki (nM)"
COL_DRUG_NAME = "BindingDB Ligand Name"
COL_MONOMER = "BindingDB MonomerID"
COL_PUBCHEM = "PubChem CID"
COL_TARGET_NAME = "Target Name"
COL_ORG = "Target Source Organism According to Curator or DataSource"

# UniProt Primary ID can repeat per chain; pick the first such column we find
def find_uniprot_primary(cols):
    cands = [c for c in cols if c.startswith("UniProt (SwissProt) Primary ID of Target Chain")]
    return cands[0] if cands else None

import re
num_re = re.compile(r"([0-9]*\.?[0-9]+)")

def to_float_nM(x):
    """
    Parse Ki strings like '12.3', '>10000', '<0.5', etc. Return float(nM) or NaN.
    We ignore inequality signs and just take the numeric part for MVP.
    """
    if pd.isna(x): return np.nan
    s = str(x).strip()
    m = num_re.search(s)
    return float(m.group(1)) if m else np.nan

def load_bindingdb_zip(in_zip: Path) -> pd.DataFrame:
    with zipfile.ZipFile(in_zip) as z:
        tsv_name = [n for n in z.namelist() if n.endswith(".tsv")][0]
        with z.open(tsv_name) as f:
            df = pd.read_csv(f, sep="\t", low_memory=False)
    return df

def main():
    if not IN_ZIP.exists():
        raise FileNotFoundError(f"Cannot find {IN_ZIP}. Upload the TSV zip there or edit IN_ZIP at top of this script.")

    df = load_bindingdb_zip(IN_ZIP)
    # Locate UniProt column if present
    col_uniprot = find_uniprot_primary(df.columns)

    # Keep human targets where possible (ok to relax later)
    if COL_ORG in df.columns:
        df = df[df[COL_ORG].fillna("").str.contains("Homo sapiens", case=False, na=False)]

    # Parse Ki
    if COL_KI not in df.columns:
        raise RuntimeError(f"Column '{COL_KI}' not found. Did you pick a *_tsv.zip file?")
    df["Ki_nM"] = df[COL_KI].apply(to_float_nM)
    df = df.dropna(subset=["Ki_nM"])

    # Build IDs
    # drug_id: prefer PubChem CID, else MonomerID, else fallback to ligand name
    def drug_id(row):
        if pd.notna(row.get(COL_PUBCHEM, np.nan)):
            try:
                return f"CID{int(float(row[COL_PUBCHEM]))}"
            except Exception:
                return f"CID{str(row[COL_PUBCHEM])}"
        if pd.notna(row.get(COL_MONOMER, np.nan)):
            try:
                return f"BDB{int(float(row[COL_MONOMER]))}"
            except Exception:
                return f"BDB{str(row[COL_MONOMER])}"
        name = row.get(COL_DRUG_NAME, "")
        return f"DRUG_{str(name)[:32].replace(' ','_')}" if name else None

    df["drug_id"] = df.apply(drug_id, axis=1)

    # gene/protein id: prefer UniProt primary, else Target Name
    if col_uniprot and col_uniprot in df.columns:
        df["gene"] = df[col_uniprot].fillna(df.get(COL_TARGET_NAME, "UNKNOWN"))
    else:
        df["gene"] = df.get(COL_TARGET_NAME, "UNKNOWN")
    df = df.dropna(subset=["drug_id","gene"])

    # Aggregate median Ki per (drug, gene)
    agg = (df.groupby(["drug_id","gene"])["Ki_nM"]
           .median()
           .reset_index()
           .rename(columns={"Ki_nM":"Ki_nM_median"}))

    # Make pKi (M) = -log10(Ki in molar); Ki(nM) -> Ki(M) = Ki(nM) * 1e-9
    agg["pKi"] = -np.log10(agg["Ki_nM_median"] * 1e-9)

    # Build a manageable panel (optional: cap sizes for Git friendliness)
    # Keep up to 500 drugs and 2000 genes with the most data
    d_counts = agg["drug_id"].value_counts()
    g_counts = agg["gene"].value_counts()
    keep_drugs = set(d_counts.head(500).index)
    keep_genes = set(g_counts.head(2000).index)
    agg = agg[agg["drug_id"].isin(keep_drugs) & agg["gene"].isin(keep_genes)]

    # Pivot gene x drug matrix of pKi
    mat = agg.pivot_table(index="gene", columns="drug_id", values="pKi", aggfunc="median")
    mat = mat.fillna(0.0)

    # ---- gene_embeddings.csv (PCA to 64D) ----
    g = mat.to_numpy(dtype=np.float32)
    k_g = min(64, g.shape[1]) if g.shape[1] > 0 else 1
    pca_g = PCA(n_components=k_g)
    G = pca_g.fit_transform(g)  # n_genes x k_g
    if G.shape[1] < 64:
        G = np.pad(G, ((0,0),(0,64-G.shape[1])), constant_values=0.0)
    ge = pd.DataFrame(G, columns=[f"g{i}" for i in range(64)])
    ge.insert(0, "gene", mat.index.to_list())
    ge.to_csv(OUT/"gene_embeddings.csv", index=False)

    # ---- cell_embeddings.csv (single “context” row) ----
    # Define one pseudo-context “BDB” with 32D vector from PCA over drug space mean
    drug_means = mat.mean(axis=0).to_numpy().reshape(1,-1).astype(np.float32)
    k_c = min(32, drug_means.shape[1]) if drug_means.shape[1] > 0 else 1
    pca_c = PCA(n_components=k_c)
    C = pca_c.fit_transform(drug_means)
    if C.shape[1] < 32:
        C = np.pad(C, ((0,0),(0,32-C.shape[1])), constant_values=0.0)
    ce = pd.DataFrame(C, columns=[f"c{i}" for i in range(32)])
    ce.insert(0, "cell_id", ["BDB"])
    ce["disease"] = 1
    ce.to_csv(OUT/"cell_embeddings.csv", index=False)

    # ---- drug_catalog.csv ----
    name_map = (df.dropna(subset=["drug_id"])
                  .groupby("drug_id")[COL_DRUG_NAME]
                  .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else "")
                  .to_dict())
    drugs = pd.DataFrame({"drug_id": sorted(keep_drugs)})
    drugs["drug_name"] = drugs["drug_id"].map(name_map).fillna("")
    drugs["approved"] = 0
    drugs["clinical_phase"] = 0
    drugs.to_csv(OUT/"drug_catalog.csv", index=False)

    # ---- train_triples.csv ----
    # Positive if Ki <= 100 nM, negative if Ki >= 10000 nM
    pos = agg[agg["Ki_nM_median"] <= 100].copy()
    neg = agg[agg["Ki_nM_median"] >= 10000].copy()
    if len(neg) > len(pos):
        neg = neg.sample(len(pos), random_state=42)

    def mk_rows(frame, label):
        g_med = frame.groupby("gene")["pKi"].transform("median")
        return pd.DataFrame({
            "gene": frame["gene"].values,
            "cell_id": ["BDB"]*len(frame),
            "drug_id": frame["drug_id"].values,
            "label": [label]*len(frame),
            "de_logfc": (frame["pKi"] - g_med).values
        })

    triples = pd.concat([mk_rows(pos,1), mk_rows(neg,0)], ignore_index=True)
    triples.to_csv(OUT/"train_triples.csv", index=False)

    print("✅ Wrote:")
    for f in ["gene_embeddings.csv","cell_embeddings.csv","drug_catalog.csv","train_triples.csv"]:
        print("   ", OUT/f)

if __name__ == "__main__":
    main()
