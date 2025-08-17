import json, math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ART = Path("artifacts"); ART.mkdir(exist_ok=True, parents=True)
DATA = Path("data")

# ---------------- Base utils ----------------
class NPDs(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float(); self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i], self.y[i]

class FFNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d,256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128,1)
        )
    def forward(self,x): return self.net(x).squeeze(-1)

# --------------- Data loaders ---------------

def load_real_subset(root: Path):
    ge = pd.read_csv(root/"gene_embeddings.csv")   # gene,g0..g63
    ce = pd.read_csv(root/"cell_embeddings.csv")   # cell_id,c0..c31,disease
    dc = pd.read_csv(root/"drug_catalog.csv")      # drug_id,drug_name,approved,clinical_phase
    tr = pd.read_csv(root/"train_triples.csv")     # gene,cell_id,drug_id,label,de_logfc
    gcols = [c for c in ge.columns if c.startswith("g")]
    ccols = [c for c in ce.columns if c.startswith("c")]
    df = tr.merge(ge,on="gene").merge(ce,on="cell_id")
    X = df[gcols+ccols].to_numpy(np.float32)
    y = df["label"].to_numpy(np.float32)
    meta = df[["gene","cell_id","drug_id","disease","de_logfc"]].copy()
    return X,y,meta,dc


def make_synthetic(n_genes=800, n_cells=400, n_drugs=25, d_g=64, d_c=32, rows=30000):
    rng=np.random.default_rng(123)
    genes=[f"G{i:05d}" for i in range(n_genes)]; cells=[f"CELL{i:05d}" for i in range(n_cells)]; drugs=[f"DRUG{i:03d}" for i in range(n_drugs)]
    GE=rng.normal(0,1,(n_genes,d_g)).astype(np.float32); CE=rng.normal(0,1,(n_cells,d_c)).astype(np.float32); disease=rng.integers(0,2,size=n_cells)
    ge=pd.DataFrame(np.c_[genes,GE], columns=["gene"]+[f"g{i}" for i in range(d_g)])
    ce=pd.DataFrame(np.c_[cells,CE], columns=["cell_id"]+[f"c{i}" for i in range(d_c)])
    ce["disease"]=disease
    appr=rng.integers(0,2,size=n_drugs); phase=[4 if a==1 else int(rng.choice([0,1,2,3],p=[.2,.3,.3,.2])) for a in appr]
    dc=pd.DataFrame({"drug_id":drugs,"drug_name":[f"Compound-{i:03d}" for i in range(n_drugs)],"approved":appr,"clinical_phase":phase})
    rows_out=[]
    wg=rng.normal(0,0.6,d_g); wc=rng.normal(0,0.6,d_c); deff=rng.normal(0.4,0.15)
    for _ in range(rows):
        gi,ci,di=rng.integers(0,n_genes),rng.integers(0,n_cells),rng.integers(0,n_drugs)
        gv,cv=GE[gi],CE[ci]; dz=disease[ci]
        logit=gv.dot(wg)+cv.dot(wc)+dz*deff+rng.normal(0,0.7)
        prob=1/(1+math.exp(-logit)); label=int(prob>0.65 and dz==1); de=gv.dot(wg)*0.4+(prob-0.5)+rng.normal(0,0.5)
        rows_out.append((genes[gi], cells[ci], drugs[di], label, de))
    tr=pd.DataFrame(rows_out, columns=["gene","cell_id","drug_id","label","de_logfc"])
    gcols=[c for c in ge.columns if c.startswith("g")]; ccols=[c for c in ce.columns if c.startswith("c")]
    df=tr.merge(ge,on="gene").merge(ce,on="cell_id")
    X=df[gcols+ccols].to_numpy(np.float32); y=df["label"].to_numpy(np.float32)
    meta=df[["gene","cell_id","drug_id","disease","de_logfc"]].copy()
    return X,y,meta,dc

# --------------- Train / evaluate ---------------

def train_eval(X,y):
    Xtr,Xva,ytr,yva=train_test_split(X,y,test_size=0.2,random_state=42,stratify=(y>0.5))
    sc=StandardScaler().fit(Xtr)
    Xtr=sc.transform(Xtr).astype(np.float32); Xva=sc.transform(Xva).astype(np.float32)

    m=FFNN(X.shape[1])
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss()
    dtr=DataLoader(NPDs(Xtr,ytr),batch_size=512,shuffle=True)
    dva=DataLoader(NPDs(Xva,yva),batch_size=512)
    best=1e9; bad=0
    for ep in range(1,11):
        m.train();
        for xb,yb in dtr:
            opt.zero_grad(); lo=m(xb); loss=lossf(lo,yb); loss.backward(); opt.step()
        # val
        m.eval(); vl=0.0; pr=[]
        with torch.no_grad():
            for xb,yb in dva:
                lo=m(xb); loss=lossf(lo,yb); vl+=float(loss)*len(xb); pr+=torch.sigmoid(lo).tolist()
        vl/=len(dva.dataset)
        if vl<best-1e-4: torch.save(m.state_dict(), ART/"model.pt"); best=vl; bad=0
        else: bad+=1
        if bad>=3: break

    # final preds on val set
    with torch.no_grad():
        logits = m(torch.from_numpy(Xva)).numpy().reshape(-1)
        probs  = 1/(1+np.exp(-logits))
    auc = float(roc_auc_score(yva, probs))
    ap  = float(average_precision_score(yva, probs))
    brier = float(brier_score_loss(yva, probs))
    prob_true, prob_pred = calibration_curve(yva, probs, n_bins=10, strategy='uniform')
    ece = float(np.average(np.abs(prob_true - prob_pred)))

    pd.DataFrame({"y_true":yva, "y_prob":probs}).to_csv(ART/"val_preds.csv", index=False)
    json.dump({"AUC":auc, "AP":ap, "Brier":brier, "ECE":ece}, open(ART/"metrics.json","w"))

# --------------- Simple network + candidates ---------------

def build_candidates(meta: pd.DataFrame, drug_df: pd.DataFrame):
    # proxy: use de_logfc to score edges; real run would use calibrated probs
    df=meta.copy()
    df["prob"] = (df["de_logfc"]-df["de_logfc"].min())/(df["de_logfc"].max()-df["de_logfc"].min()+1e-8)

    grp = df[df["disease"]==1].groupby(["drug_id","gene"]).agg(weight=("prob","mean")).reset_index()
    grp = grp.sort_values(["drug_id","weight"],ascending=[True,False]).groupby("drug_id").head(100)
    grp.to_csv(ART/"edges.csv", index=False)  # for app evidence table

    # aggregate to drug level
    agg = grp.groupby("drug_id").agg(net_strength=("weight","sum")).reset_index()
    # rough betweenness proxy via weight variance across genes
    betw = grp.groupby("drug_id").agg(net_betweenness=("weight","std")).fillna(0).reset_index()

    ranked = drug_df.merge(agg, on="drug_id", how="left").merge(betw, on="drug_id", how="left")
    ranked[["net_strength","net_betweenness"]]=ranked[["net_strength","net_betweenness"]].fillna(0.0)
    ranked["phase_bonus"]=ranked["clinical_phase"].fillna(0).astype(float)*0.05
    ranked["candidate_score"]=ranked["net_strength"]+0.5*ranked["net_betweenness"]+ranked["phase_bonus"]

    ranked["rationale_business"]= "Impacts diseased contexts; potential repurpose."
    ranked["rationale_stat"] = "Prototype scoring; calibration next."
    ranked["rationale_bio"]  = "Top genes in diseased contexts."
    ranked["rationale_reg"]  = "Late‑phase/approved faster via 505(b)(2) if applicable."

    ranked.to_csv(ART/"candidates.csv", index=False)

# --------------- Main ---------------
if __name__ == "__main__":
    real_root = DATA/"real"
    if real_root.exists():
        X,y,meta,drug_df = load_real_subset(real_root)
    else:
        X,y,meta,drug_df = make_synthetic()
    train_eval(X,y)
    build_candidates(meta, drug_df)
    print("Done. Wrote artifacts: candidates.csv, metrics.json, val_preds.csv, edges.csv")
