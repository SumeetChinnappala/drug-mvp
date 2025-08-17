# scripts/make_artifacts_from_real.py
# v1 builder: parse BindingDB PDSP Ki, write all app + validation artifacts.
# Inputs  : data/BindingDB_PDSPKi_202508_tsv.zip  (TSV inside the zip)
# Outputs : artifacts/{candidates.csv,edges.csv,val_preds.csv,metrics.json}
#           artifacts/validation/{summary.json,calibration.csv,pred_dist.png}

import io, json, re
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

# Headless plotting for servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent.parent
ZIP_PATH = HERE / "data" / "BindingDB_PDSPKi_202508_tsv.zip"
ART = HERE / "artifacts"
VAL = ART / "validation"
ART.mkdir(parents=True, exist_ok=True)
VAL.mkdir(parents=True, exist_ok=True)

# -------------------------- helpers --------------------------

def pick_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for name in candidates:
        if name.lower() in low:
            return low[name.lower()]
    return None

_unit_pat = re.compile(r"^\s*([<>≈~]?\s*)?([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*([munpkμ]?M|nM|uM|pM|fM|mM)?\s*$")

def to_nM(val):
    """Parse a Ki-like value into nM."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val) if val > 0 else np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    m = _unit_pat.match(s.replace("μ", "u"))
    if not m:
        # bare number => assume M
        try:
            return float(s) * 1e9
        except:
            return np.nan
    num = float(m.group(2))
    unit = (m.group(3) or "").lower()
    mult = (
        1.0 if unit in ("nm",) else
        1e3 if unit in ("um",) else
        1e-3 if unit in ("pm",) else
        1e-6 if unit in ("fm",) else
        1e6 if unit in ("mm",) else
        1e9 if unit in ("m",) else
        1.0
    )
    nm = num * mult
    return nm if (nm > 0 and np.isfinite(nm)) else np.nan

def ece_score(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-9, 1-1e-9)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
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

# -------------------------- pipeline --------------------------

def load_pdspki():
    assert ZIP_PATH.exists(), f"Missing {ZIP_PATH}"
    with ZipFile(ZIP_PATH) as z:
        tsv_name = next((n for n in z.namelist() if n.lower().endswith(".tsv")), None)
        assert tsv_name, "No .tsv found inside PDSP Ki zip"
        with z.open(tsv_name) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="ignore"),
                             sep="\t", low_memory=False)

    col_cid  = pick_col(df.columns, ["PubChem CID", "CID", "Ligand PubChem CID"])
    col_db   = pick_col(df.columns, ["DrugBank ID of Ligand", "DrugBank ID"])
    col_name = pick_col(df.columns, ["Ligand Name", "Ligand", "Drug Name"])
    col_uni  = pick_col(df.columns, ["UniProt (Target) Primary ID", "Uniprot", "UniProt"])
    col_ki   = pick_col(df.columns, ["Ki (nM)", "Ki", "Ki (nM) (Mean)", "Ki (mean)"]) \
               or pick_col(df.columns, ["IC50 (nM)", "IC50"]) \
               or pick_col(df.columns, ["Kd (nM)", "Kd"])

    keep = df
    if col_uni: keep = keep[keep[col_uni].notna()]
    if col_ki:  keep = keep[keep[col_ki].notna()]
    keep = keep.copy()

    def mk_drug_id(row):
        cid = str(row[col_cid]).strip() if col_cid and not pd.isna(row[col_cid]) else ""
        db  = str(row[col_db]).strip()  if col_db  and not pd.isna(row[col_db])  else ""
        if cid and cid.lower() not in ("nan", "none", "") and cid != "0":
            first = re.split(r"[;,\s]+", cid)[0]
            if first.isdigit():
                return first  # PubChem CID (string)
        if db and db.lower() not in ("nan", "none", ""):
            return re.split(r"[;,\s]+", db)[0]  # DrugBank ID
        return None

    keep["drug_id"]   = keep.apply(mk_drug_id, axis=1)
    keep["drug_name"] = keep[col_name].astype(str) if col_name else np.nan
    keep["gene"]      = keep[col_uni].astype(str).str.strip() if col_uni else np.nan
    keep["Ki_nM"]     = keep[col_ki].map(to_nM) if col_ki else np.nan

    keep = keep[keep["drug_id"].notna() & keep["gene"].notna() & keep["Ki_nM"].notna()]
    keep = keep[keep["Ki_nM"] > 0]

    # pKi: pKi = -log10(Ki [M]) = -(log10(Ki_nM) - 9) = 9 - log10(Ki_nM)
    keep["pKi"] = 9.0 - np.log10(keep["Ki_nM"])

    # deduplicate best (max pKi) per (drug_id, gene)
    keep = keep.sort_values("pKi", ascending=False).drop_duplicates(["drug_id", "gene"])
    return keep[["drug_id", "drug_name", "gene", "Ki_nM", "pKi"]]

def build_edges_and_candidates(df, topN=50):
    # normalize pKi within each drug to [0,1] => weight
    def norm_group(g):
        p = g["pKi"].values
        if len(p) == 0:
            g["weight"] = 0.0
            return g
        lo, hi = np.nanpercentile(p, 5), np.nanpercentile(p, 95)
        w = np.zeros_like(p, float) if hi <= lo else np.clip((p - lo) / (hi - lo), 0, 1)
        g["weight"] = w
        return g

    edges = df.groupby("drug_id", group_keys=False).apply(norm_group)
    edges["direction"] = np.where(edges["pKi"] >= edges["pKi"].median(), 1, -1)
    edges["source"]    = "BindingDB PDSP Ki"

    # candidates
    agg = edges.groupby("drug_id").agg(
        net_strength=("weight", "mean"),
        num_targets=("gene", "nunique"),
        drug_name=("drug_name", "first"),
    ).reset_index()

    cnt = agg["num_targets"].astype(float).values
    cnt_norm = (cnt - cnt.min()) / (cnt.max() - cnt.min() + 1e-9)
    agg["net_betweenness"] = cnt_norm
    agg["candidate_score"] = 0.7 * agg["net_strength"] + 0.3 * agg["net_betweenness"]

    # choose topN drugs for edges to keep UI snappy
    agg = agg.sort_values("candidate_score", ascending=False)
    top_drugs = set(agg["drug_id"].head(topN))
    edges = edges[edges["drug_id"].isin(top_drugs)]

    return edges[["drug_id", "gene", "weight", "direction", "source"]], agg[
        ["drug_id", "drug_name", "candidate_score", "net_strength", "net_betweenness"]
    ]

def build_valpreds_and_metrics(edges):
    # positives: observed pairs
    pos = edges[["drug_id", "gene", "weight"]].copy()
    pos["y_true"] = 1

    # negatives: per drug, sample unseen genes
    rng = np.random.default_rng(7)
    genes_all = pos["gene"].unique()
    neg_rows = []
    for d, gdf in pos.groupby("drug_id"):
        seen = set(gdf["gene"])
        cand = [g for g in genes_all if g not in seen]
        if not cand:
            continue
        k = min(len(seen), len(cand))
        pick = rng.choice(cand, size=k, replace=False)
        for g in pick:
            neg_rows.append((d, g, 0.0, 0))
    neg = pd.DataFrame(neg_rows, columns=["drug_id", "gene", "weight", "y_true"])

    df = pd.concat([pos, neg], ignore_index=True)

    # simple 1-feature classifier (weight)
    X = df[["weight"]].values
    y = df["y_true"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr = LogisticRegression(solver="lbfgs")
    lr.fit(X_tr, y_tr)
    y_prob = lr.predict_proba(X_te)[:, 1]

    # metrics
    auc = roc_auc_score(y_te, y_prob)
    ap  = average_precision_score(y_te, y_prob)
    acc = float(((y_prob >= 0.5).astype(int) == y_te).mean())
    brier = brier_score_loss(y_te, y_prob)
    ece, calib = ece_score(y_te, y_prob, n_bins=10)

    # write artifacts used by UI
    pd.DataFrame({"y_true": y_te, "y_prob": y_prob}).to_csv(ART / "val_preds.csv", index=False)
    (ART / "metrics.json").write_text(json.dumps({
        "AUC": float(auc),
        "AP": float(ap),
        "acc@0.5": float(acc),
        "Brier": float(brier),
        "ECE": float(ece),
        "run_id": f"pdspki-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }, indent=2))

    # write validation bundle for Patch 1 panel
    calib.to_csv(VAL / "calibration.csv", index=False)
    try:
        plt.figure()
        plt.hist(y_prob, bins=30)
        plt.xlabel("Predicted probability")
        plt.ylabel("Count")
        plt.title("Prediction distribution")
        plt.tight_layout()
        plt.savefig(VAL / "pred_dist.png", dpi=150)
        plt.close()
    except Exception:
        # fine if plotting isn't available; app doesn't require the PNG
        pass

    # confusion summary for summary.json
    y_hat = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, y_hat, labels=[0, 1]).ravel()
    vsummary = {
        "version": "v1",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "class_balance": {"positives": int(y_te.sum()), "negatives": int((1 - y_te).sum())},
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "links": [
            "artifacts/metrics.json",
            "artifacts/val_preds.csv",
            "artifacts/validation/calibration.csv",
            "artifacts/validation/pred_dist.png",
        ],
    }
    (VAL / "summary.json").write_text(json.dumps(vsummary, indent=2))

def main():
    print(">> Loading PDSP Ki …")
    df = load_pdspki()
    if len(df) < 100:
        raise RuntimeError(f"Too few PDSP Ki rows parsed ({len(df)}).")
    print(f"   rows: {len(df):,}  drugs: {df['drug_id'].nunique():,}  genes: {df['gene'].nunique():,}")

    print(">> Building edges & candidates …")
    edges, cands = build_edges_and_candidates(df, topN=50)
    edges.to_csv(ART / "edges.csv", index=False)
    cands.to_csv(ART / "candidates.csv", index=False)

    print(">> Building validation & metrics …")
    build_valpreds_and_metrics(edges)

    print("== DONE ==")
    for p in [
        ART / "candidates.csv",
        ART / "edges.csv",
        ART / "val_preds.csv",
        ART / "metrics.json",
        VAL / "summary.json",
        VAL / "calibration.csv",
        VAL / "pred_dist.png",
    ]:
        print(f"  - {p.relative_to(HERE)}  exists={p.exists()} size={p.stat().st_size if p.exists() else 0}")

if __name__ == "__main__":
    main()
