# scripts/prepare_bindingdb_subset.py  (robust)
import re, zipfile, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

SEARCH_ROOTS = [Path("data"), Path("data/bindingdb"), Path("data/bindingdb/raw")]
PATTERNS = ["BindingDB_PDSPKi*.tsv.zip","BindingDB_PDSPKi*_tsv.zip","*PDSPKi*.tsv.zip","*PDSPKi*_tsv.zip"]

def _cap(nc, ns, nf): return int(max(1, min(nc, ns, nf)))

def find_input():
    preferred = Path("data/BindingDB_PDSPKi_202508_tsv.zip")
    if preferred.exists(): return preferred
    for r in SEARCH_ROOTS:
        if not r.exists(): continue
        for pat in PATTERNS:
            for p in r.glob(pat): return p
    raise FileNotFoundError("No PDSP Ki TSV zip found under data/. Accepts *.tsv.zip or *_tsv.zip")

def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower()==".zip":
        with zipfile.ZipFile(path) as z:
            tsvs=[n for n in z.namelist() if n.lower().endswith(".tsv")]
            if not tsvs: raise RuntimeError(f"No .tsv inside {path.name}")
            with z.open(tsvs[0]) as f: return pd.read_csv(f, sep="\t", low_memory=False)
    elif path.suffix.lower()==".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    else: raise RuntimeError(f"Unsupported file: {path}")

def to_float_nM(x):
    if pd.isna(x): return np.nan
    m=re.search(r"([0-9]*\.?[0-9]+)", str(x)); return float(m.group(1)) if m else np.nan

def main():
    src = find_input()
    out = Path("data/real"); out.mkdir(parents=True, exist_ok=True)
    df = load_any(src)

    COL_KI="Ki (nM)"; COL_DN="BindingDB Ligand Name"; COL_MON="BindingDB MonomerID"
    COL_PC="PubChem CID"; COL_TN="Target Name"
    COL_ORG="Target Source Organism According to Curator or DataSource"
    uni=[c for c in df.columns if c.startswith("UniProt (SwissProt) Primary ID of Target Chain")]
    COL_UP = uni[0] if uni else None

    # Keep human if column exists
    if COL_ORG in df.columns:
        df = df[df[COL_ORG].fillna("").str.contains("Homo sapiens", case=False, na=False)]

    if COL_KI not in df.columns:
        raise RuntimeError(f"Column '{COL_KI}' missing")

    df["Ki_nM"] = df[COL_KI].apply(to_float_nM)
    df = df.dropna(subset=["Ki_nM"])

    def drug_id(r):
        pc=r.get(COL_PC, np.nan)
        if pd.notna(pc):
            try: return f"CID{int(float(pc))}"
            except: return f"CID{str(pc)}"
        mon=r.get(COL_MON, np.nan)
        if pd.notna(mon):
            try: return f"BDB{int(float(mon))}"
            except: return f"BDB{str(mon)}"
        nm=r.get(COL_DN,"")
        return f"DRUG_{str(nm)[:32].replace(' ','_')}" if nm else None

    df["drug_id"]=df.apply(drug_id,axis=1)
    df["gene"] = df[COL_UP].fillna(df.get(COL_TN,"UNKNOWN")) if (COL_UP and COL_UP in df.columns) else df.get(COL_TN,"UNKNOWN")
    df = df.dropna(subset=["drug_id","gene"])

    # Aggregate median Ki per (drug,gene), compute pKi
    agg = (df.groupby(["drug_id","gene"])["Ki_nM"].median().reset_index().rename(columns={"Ki_nM":"Ki_nM_median"}))
    if agg.empty: raise RuntimeError("No (drug,gene) rows after aggregation")
    agg["pKi"] = -np.log10(agg["Ki_nM_median"]*1e-9)

    # Soft caps; if empty after caps, fall back to uncapped
    d_counts = agg["drug_id"].value_counts()
    g_counts = agg["gene"].value_counts()
    keep_drugs = set(d_counts.head(1000).index)
    keep_genes = set(g_counts.head(5000).index)
    capped = agg[agg["drug_id"].isin(keep_drugs) & agg["gene"].isin(keep_genes)]
    if capped.empty: capped = agg.copy()

    # Pivot matrix gene x drug
    mat = capped.pivot_table(index="gene", columns="drug_id", values="pKi", aggfunc="median").fillna(0.0)
    G = mat.to_numpy(dtype=np.float32)
    if G.ndim!=2 or G.shape[0]==0 or G.shape[1]==0:
        raise RuntimeError("Pivot produced empty matrix")

    # Gene embeddings 64D (cap by rows/cols, then pad)
    k_g = _cap(64, G.shape[0], G.shape[1])
    GE = PCA(n_components=k_g).fit_transform(G)
    if GE.shape[1] < 64: GE = np.pad(GE, ((0,0),(0,64-GE.shape[1])), constant_values=0.0)
    ge = pd.DataFrame(GE, columns=[f"g{i}" for i in range(64)]); ge.insert(0,"gene",mat.index.to_list())
    ge.to_csv(out/"gene_embeddings.csv", index=False)

    # Single pseudo-cell → components must be 1, then pad to 32
    dm = mat.mean(axis=0).to_numpy().reshape(1,-1).astype(np.float32)
    k_c = _cap(32, dm.shape[0], dm.shape[1])  # -> 1
    CE = PCA(n_components=k_c).fit_transform(dm)
    if CE.shape[1] < 32: CE = np.pad(CE, ((0,0),(0,32-CE.shape[1])), constant_values=0.0)
    ce = pd.DataFrame(CE, columns=[f"c{i}" for i in range(32)]); ce.insert(0,"cell_id",["BDB"]); ce["disease"]=1
    ce.to_csv(out/"cell_embeddings.csv", index=False)

    # Drug catalog
    name_map = (df.dropna(subset=["drug_id"]).groupby("drug_id")[COL_DN]
                  .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else "")
                  .to_dict())
    drugs = pd.DataFrame({"drug_id": sorted(set(capped["drug_id"]))})
    drugs["drug_name"]=drugs["drug_id"].map(name_map).fillna(""); drugs["approved"]=0; drugs["clinical_phase"]=0
    drugs.to_csv(out/"drug_catalog.csv", index=False)

    # Labels: try strict, then relaxed, then quantiles
    def label_pairs(th_lo=100.0, th_hi=10000.0):
        pos = capped[capped["Ki_nM_median"] <= th_lo].copy()
        neg = capped[capped["Ki_nM_median"] >= th_hi].copy()
        return pos, neg

    pos, neg = label_pairs(100.0, 10000.0)
    if min(len(pos),len(neg)) < 100:
        pos, neg = label_pairs(1000.0, 100000.0)
    if min(len(pos),len(neg)) < 50:
        # quantile fallback (top 30% vs bottom 30% by pKi)
        q70, q30 = capped["pKi"].quantile(0.70), capped["pKi"].quantile(0.30)
        pos = capped[capped["pKi"] >= q70].copy()
        neg = capped[capped["pKi"] <= q30].copy()

    # balance classes (avoid empty)
    n = min(len(pos), len(neg))
    if n == 0:
        raise RuntimeError("No positive/negative pairs after labeling")
    pos = pos.sample(n, random_state=42); neg = neg.sample(n, random_state=42)

    def mk_rows(frame, label):
        g_med = frame.groupby("gene")["pKi"].transform("median")
        return pd.DataFrame({
            "gene": frame["gene"].values, "cell_id": ["BDB"]*len(frame),
            "drug_id": frame["drug_id"].values, "label": [label]*len(frame),
            "de_logfc": (frame["pKi"] - g_med).values
        })
    triples = pd.concat([mk_rows(pos,1), mk_rows(neg,0)], ignore_index=True)
    triples.to_csv(out/"train_triples.csv", index=False)

    # summary
    (out/"_summary.json").write_text(json.dumps({
        "src": str(src), "genes": int(G.shape[0]), "drugs": int(G.shape[1]),
        "pairs_total": int(len(capped)), "pos": int(len(pos)), "neg": int(len(neg))
    }, indent=2))
    print("✅ Wrote data/real/: gene_embeddings.csv, cell_embeddings.csv, drug_catalog.csv, train_triples.csv")

if __name__=="__main__": main()
