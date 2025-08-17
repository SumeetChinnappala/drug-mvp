import json, math, re, io
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
ZIP_PATH = HERE / "data" / "BindingDB_PDSPKi_202508_tsv.zip"
ART = HERE / "artifacts"
VAL = ART / "validation"
ART.mkdir(parents=True, exist_ok=True)
VAL.mkdir(parents=True, exist_ok=True)

def pick_col(cols, candidates):
    cols_low = {c.lower(): c for c in cols}
    for name in candidates:
        if name.lower() in cols_low:
            return cols_low[name.lower()]
    return None

_unit_pat = re.compile(r"^\s*([<>≈~]?\s*)?([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*([munpkμ]?M|nM|uM|pM|fM|mM)?\s*$")

def to_nM(val):
    """
    Parse a Ki-like string/number into nM (float).
    Accepts things like '35', '35 nM', '0.12 uM', '< 10 nM', '1e-9 M'. Returns np.nan if not parseable.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        # assume already nM if no units supplied in dataset numeric column
        # to be safer, treat plain numeric as nM unless absurd
        if val <= 0:
            return np.nan
        return float(val)
    s = str(val).strip()
    if not s:
        return np.nan
    m = _unit_pat.match(s.replace("μ", "u"))
    if not m:
        # try to detect raw Molar like '1e-8'
        try:
            x = float(s)
            # assume M
            return x * 1e9
        except:
            return np.nan
    num = float(m.group(2))
    unit = (m.group(3) or "").lower()
    if unit in ("nm",):
        mult = 1.0
    elif unit in ("um",):
        mult = 1e3
    elif unit in ("pm",):
        mult = 1e-3
    elif unit in ("fm",):
        mult = 1e-6
    elif unit in ("mm",):
        mult = 1e6
    elif unit in ("m",):
        mult = 1e9
    else:
        # no unit: assume nM
        mult = 1.0
    nm = num * mult
    if nm <= 0 or not np.isfinite(nm):
        return np.nan
    return nm

def ece_score(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-9, 1-1e-9)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    total = len(y_true)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            rows.append((np.nan, np.nan, 0))
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        w = mask.mean()
        ece += w * abs(acc - conf)
        rows.append((acc, conf, mask.sum()))
    calib = pd.DataFrame(rows, columns=["prob_true", "prob_pred", "n"])
    return float(ece), calib

def load_pdspki():
    assert ZIP_PATH.exists(), f"Missing {ZIP_PATH}"
    with ZipFile(ZIP_PATH) as z:
        # pick the first .tsv inside
        tsv_name = next((n for n in z.namelist() if n.lower().endswith(".tsv")), None)
        assert tsv_name, "No .tsv found inside PDSP Ki zip"
        with z.open(tsv_name) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="ignore"), sep="\t", low_memory=False)
    # map flexible column names
    col_cid = pick_col(df.columns, ["PubChem CID", "CID", "Ligand PubChem CID"])
    col_db  = pick_col(df.columns, ["DrugBank ID of Ligand", "DrugBank ID"])
    col_name= pick_col(df.columns, ["Ligand Name", "Ligand", "Drug Name"])
    col_uni = pick_col(df.columns, ["UniProt (Target) Primary ID", "Uniprot", "UniProt"])
    # prefer Ki; fall back to IC50 or Kd if needed
    col_ki  = pick_col(df.columns, ["Ki (nM)", "Ki", "Ki (nM) (Mean)", "Ki (mean)"])
    if col_ki is None:
        col_ki = pick_col(df.columns, ["IC50 (nM)", "IC50"])
    if col_ki is None:
        col_ki = pick_col(df.columns, ["Kd (nM)", "Kd"])
    for c in [col_cid, col_db, col_name, col_uni, col_ki]:
        if c is None:
            pass
    # keep only sensible rows
    keep = df
    if col_uni: keep = keep[keep[col_uni].notna()]
    if col_ki:  keep = keep[keep[col_ki].notna()]
    keep = keep.copy()
    # build drug_id: prefer CID, else DrugBank, else drop
    def mk_drug_id(row):
        cid = str(row[col_cid]).strip() if col_cid and not pd.isna(row[col_cid]) else ""
        db  = str(row[col_db]).strip()  if col_db and not pd.isna(row[col_db])  else ""
        if cid and cid.lower() not in ("nan", "none", "") and cid != "0":
            # BindingDB may include lists like "1234; 5678" — pick the first numeric
            cid_first = re.split(r"[;,\s]+", cid)[0]
            if cid_first.isdigit():
                return cid_first  # use plain numeric CID as string
        if db and db.lower() not in ("nan", "none", ""):
            db_first = re.split(r"[;,\s]+", db)[0]
            return db_first
        return None
    keep["drug_id"] = keep.apply(mk_drug_id, axis=1)
    if col_name:
        keep["drug_name"] = keep[col_name].astype(str)
    else:
        keep["drug_name"] = np.nan
    if col_uni:
        keep["gene"] = keep[col_uni].astype(str).str.strip()
    # Ki to nM, then to pKi
    keep["Ki_nM"] = keep[col_ki].map(to_nM) if col_ki else np.nan
    keep = keep[keep["drug_id"].notna() & keep["gene"].notna() & keep["Ki_nM"].notna()]
    keep = keep[keep["Ki_nM"] > 0]
    keep["pKi"] = 9.0 - np.log10(keep["Ki_nM"])  # since pKi = -log10(Ki[M]) and Ki_nM * 1e-9 M
    # deduplicate per (drug_id, gene): keep best (max pKi)
    keep = keep.sort_values("pKi", ascending=False).drop_duplicates(["drug_id", "gene"])
    return keep[["drug_id", "drug_name", "gene", "Ki_nM", "pKi"]]

def build_edges_and_candidates(df):
    # per drug normalization of weights to [0,1]
    def norm_group(g):
        p = g["pKi"].values
        if len(p) == 0:
            g["weight"] = 0.0
            return g
        lo, hi = np.nanpercentile(p, 5), np.nanpercentile(p, 95)
        if hi <= lo:
            w = np.zeros_like(p, float)
        else:
            w = np.clip((p - lo) / (hi - lo), 0, 1)
        g["weight"] = w
        return g
    edges = df.groupby("drug_id", group_keys=False).apply(norm_group)
    edges["direction"] = np.where(edges["pKi"] >= edges["pKi"].median(), 1, -1)
    edges["source"] = "BindingDB PDSP Ki"

    # candidates
    agg = edges.groupby("drug_id").agg(
        net_strength=("weight", "mean"),
        num_targets=("gene", "nunique"),
        drug_name=("drug_name", "first")
    ).reset_index()
    # proxy "betweenness" = normalized target count
    cnt = agg["num_targets"].values.astype(float)
    cnt_norm = (cnt - cnt.min()) / (cnt.max() - cnt.min() + 1e-9)
    agg["net_betweenness"] = cnt_norm
    agg["candidate_score"] = 0.7 * agg["net_strength"] + 0.3 * agg["net_betweenness"]
    return edges[["drug_id", "gene", "weight", "direction", "source"]], agg[["drug_id", "drug_name", "candidate_score", "net_strength", "net_betweenness"]]

def build_valpreds_and_metrics(edges):
    # Positives: observed pairs
    pos = edges[["drug_id", "gene", "weight"]].copy()
    pos["y_true"] = 1
    # Negatives: sample for each drug genes not observed
    rng = np.random.default_rng(7)
    drugs = pos["drug_id"].unique()
    genes_all = pos["gene"].unique()
    neg_rows = []
    for d in drugs:
        seen = set(pos.loc[pos["drug_id"] == d, "gene"])
        cand = [g for g in genes_all if g not in seen]
        if not cand:
            continue
        k = min(len(seen), len(cand))
        pick = rng.choice(cand, size=k, replace=False)
        for g in pick:
            neg_rows.append((d, g, 0.0, 0))
    neg = pd.DataFrame(neg_rows, columns=["drug_id", "gene", "weight", "y_true"])
    df = pd.concat([pos, neg], ignore_index=True)
    # simple features: weight only (0 for negatives)
    X = df[["weight"]].values
    y = df["y_true"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(X_tr, y_tr)
    y_prob = lr.predict_proba(X_te)[:, 1]
    # metrics
    roc = roc_auc_score(y_te, y_prob)
    ap = average_precision_score(y_te, y_prob)
    acc = float((y_prob >= 0.5).astype(int).mean() == y_te.mean())  # not ideal; report true acc@0.5
    acc = float(((y_prob >= 0.5).astype(int) == y_te).mean())
    brier = brier_score_loss(y_te, y_prob)
    ece, calib = ece_score(y_te, y_prob, n_bins=10)
    # write validation helpers
    calib.to_csv(VAL / "calibration.csv", index=False)
    plt.figure()
    plt.hist(y_prob, bins=30)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Prediction distribution")
    plt.tight_layout()
    plt.savefig(VAL / "pred_dist.png", dpi=150)
    plt.close()
    # write val_preds.csv
    val_preds = pd.DataFrame({"y_true": y_te, "y_prob": y_prob})
    val_preds.to_csv(ART / "val_preds.csv", index=False)
    # write metrics.json
    metrics = {
        "AUC": float(roc),
        "AP": float(ap),
        "acc@0.5": float(acc),
        "Brier": float(brier),
        "ECE": float(ece),
        "run_id": f"pdspki-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

def main():
    print(">> Loading PDSP Ki from zip …")
    df = load_pdspki()
    if len(df) < 100:
        raise RuntimeError(f"Too few PDSP Ki rows parsed ({len(df)}). Check columns in the TSV inside the zip.")
    print(f"   rows parsed: {len(df):,}  drugs: {df['drug_id'].nunique():,}  genes: {df['gene'].nunique():,}")

    print(">> Building edges and candidates …")
    edges, cands = build_edges_and_candidates(df)
    # keep a manageable top set for MVP (optional)
    cands = cands.sort_values("candidate_score", ascending=False)
    top_drugs = cands["drug_id"].head(50).tolist()
    edges_top = edges[edges["drug_id"].isin(top_drugs)]

    edges_top.to_csv(ART / "edges.csv", index=False)
    cands.to_csv(ART / "candidates.csv", index=False)

    print(">> Building validation (val_preds, metrics, calibration, pred_dist) …")
    metrics = build_valpreds_and_metrics(edges_top)

    print("== DONE ==")
    print(" artifacts/candidates.csv :", (ART / 'candidates.csv').exists())
    print(" artifacts/edges.csv      :", (ART / 'edges.csv').exists())
    print(" artifacts/val_preds.csv  :", (ART / 'val_preds.csv').exists())
    print(" artifacts/metrics.json   :", (ART / 'metrics.json').exists())
    print(" validation/calibration.csv:", (VAL / 'calibration.csv').exists())
    print(" validation/pred_dist.png :", (VAL / 'pred_dist.png').exists())
    print(" metrics:", metrics)

if __name__ == "__main__":
    main()
