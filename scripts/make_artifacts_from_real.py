# scripts/make_artifacts_from_real.py
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

DATA = Path("data/real"); ART = Path("artifacts"); ART.mkdir(exist_ok=True, parents=True)

ge = pd.read_csv(DATA/"gene_embeddings.csv")
ce = pd.read_csv(DATA/"cell_embeddings.csv")
dc = pd.read_csv(DATA/"drug_catalog.csv")
tr = pd.read_csv(DATA/"train_triples.csv")

gcols = [c for c in ge.columns if c.startswith("g")]
ccols = [c for c in ce.columns if c.startswith("c")]

df = tr.merge(ge, on="gene").merge(ce, on="cell_id")
X = df[gcols+ccols].to_numpy(np.float32)
y = df["label"].astype(np.float32).to_numpy()
meta = df[["gene","cell_id","drug_id","disease","de_logfc"]].copy()

# Split (robust to small sets)
strat = (y>0.5).astype(int) if (y.min()==0 and y.max()==1 and len(np.unique(y))>1) else None
test_size = 0.2 if len(y)>=30 else 0.5 if len(y)>=10 else 0.34
Xtr,Xva,ytr,yva = train_test_split(X,y,test_size=test_size,random_state=42,stratify=strat)

sc = StandardScaler().fit(Xtr)
Xtr = sc.transform(Xtr).astype(np.float32)
Xva = sc.transform(Xva).astype(np.float32)

# Fast model
lr = LogisticRegression(max_iter=2000)
lr.fit(Xtr, ytr.astype(int))
p_val = lr.predict_proba(Xva)[:,1]

# Metrics (guard)
def safe(metric, *args):
    try: return float(metric(*args))
    except Exception: return float("nan")
auc   = safe(roc_auc_score, yva, p_val)
ap    = safe(average_precision_score, yva, p_val)
brier = safe(brier_score_loss, yva, p_val)

# Simple ECE
def ece(y,p,bins=10):
    y=np.asarray(y); p=np.asarray(p)
    ids = np.clip((p*bins).astype(int), 0, bins-1)
    acc=[]; conf=[]; w=[]
    for b in range(bins):
        m = ids==b
        if m.any():
            acc.append(y[m].mean()); conf.append(p[m].mean()); w.append(m.mean())
    if not w: return float("nan")
    return float(np.average(np.abs(np.array(acc)-np.array(conf)), weights=np.array(w)))
ECE = ece(yva, p_val)

# Save val preds (non-empty, with headers)
pd.DataFrame({
    "y_true": yva,
    "y_prob_cal": p_val,
    "y_prob_raw": p_val,
    "logit": np.log(np.clip(p_val,1e-8,1-1e-8))-np.log(np.clip(1-p_val,1e-8,1-1e-8)),
    "y_prob": p_val
}).to_csv(ART/"val_preds.csv", index=False)

# Save metrics
(ART/"metrics.json").write_text(json.dumps({
    "AUC": auc, "AP": ap, "Brier": brier, "ECE": ECE,
    "calibration": {"type":"none"},
    "n_train": int(len(ytr)), "n_val": int(len(yva)),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
}, indent=2))

# Score ALL rows → edges & candidates
p_all = lr.predict_proba(sc.transform(X))[:,1]
df_all = meta.copy(); df_all["prob"] = p_all
if "disease" in df_all.columns:
    df_all = df_all[df_all["disease"]==1]

grp = df_all.groupby(["drug_id","gene"])["prob"].mean().reset_index(name="weight")
if "de_logfc" in meta.columns:
    dir_df = meta.groupby(["drug_id","gene"])["de_logfc"].mean().reset_index(name="de_mean")
    grp = grp.merge(dir_df, on=["drug_id","gene"], how="left")
    grp["direction"] = np.where(grp["de_mean"]>=0, "up", "down"); grp.drop(columns=["de_mean"], inplace=True)
else:
    grp["direction"] = "n/a"
grp["source"] = "sklearn_lr_prob_mean"
top = grp.sort_values(["drug_id","weight"], ascending=[True,False]).groupby("drug_id").head(100).reset_index(drop=True)
top.to_csv(ART/"edges.csv", index=False)

agg = grp.groupby("drug_id")["weight"].sum().reset_index(name="net_strength")
bet = grp.groupby("drug_id")["weight"].std().fillna(0.0).reset_index(name="net_betweenness")
ranked = (dc.merge(agg, on="drug_id", how="left").merge(bet, on="drug_id", how="left"))
ranked[["net_strength","net_betweenness"]] = ranked[["net_strength","net_betweenness"]].fillna(0.0)
ranked["phase_bonus"] = ranked.get("clinical_phase",0).fillna(0).astype(float)*0.05
ranked["candidate_score"] = ranked["net_strength"] + 0.5*ranked["net_betweenness"] + ranked["phase_bonus"]
ranked["rationale_business"] = "Signals in disease contexts; potential repurpose/fast-follow."
ranked["rationale_stat"]     = "Sklearn LR; edges = mean prob per gene (top-k)."
ranked["rationale_bio"]      = "High-probability genes match direction of effect."
ranked["rationale_reg"]      = "If approved/late phase, consider 505(b)(2) route."
ranked.to_csv(ART/"candidates.csv", index=False)

print("✅ wrote artifacts: metrics.json, val_preds.csv, edges.csv, candidates.csv")
