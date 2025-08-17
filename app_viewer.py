import json
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Drug Discovery — MVP", layout="wide")
st.title("Drug Discovery — MVP")

ART = Path("artifacts")

# --- Load candidates ---
p_cand = ART / "candidates.csv"
if not p_cand.exists():
    st.error("No candidates found. Add **artifacts/candidates.csv** to this repo.")
    st.stop()

cands = pd.read_csv(p_cand)
if cands.empty:
    st.warning("candidates.csv is empty.")
    st.stop()

# --- Optional metrics panel ---
metrics = None
p_metrics = ART / "metrics.json"
if p_metrics.exists():
    try:
        metrics = json.loads(p_metrics.read_text())
    except Exception as e:
        st.warning(f"metrics.json present but could not be parsed: {e}")

if metrics:
    st.markdown("### Model evaluation")
    c1, c2, c3 = st.columns(3)
    if "AUC" in metrics:      c1.metric("AUC", f"{float(metrics['AUC']):.3f}")
    if "AP" in metrics:       c2.metric("AP", f"{float(metrics['AP']):.3f}")
    if "acc@0.5" in metrics:  c3.metric("Acc@0.5", f"{float(metrics['acc@0.5']):.3f}")
    c4, c5 = st.columns(2)
    if "ECE" in metrics:      c4.metric("ECE", f"{float(metrics['ECE']):.3f}")
    if "Brier" in metrics:    c5.metric("Brier", f"{float(metrics['Brier']):.3f}")
    st.caption(f"run_id: {metrics.get('run_id','n/a')} • timestamp: {metrics.get('timestamp','n/a')}")
else:
    st.info("Add **artifacts/metrics.json** to display AUC/AP and calibration metrics.")

# --- Sensitivity control for ranking ---
st.sidebar.markdown("### Ranking sensitivity")
w_strength = st.sidebar.slider("Weight: net_strength", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
w_betw    = 1.0 - w_strength

# Recompute a simple composite score on-the-fly for display
cview = cands.copy()
if {"net_strength", "net_betweenness"}.issubset(cview.columns):
    cview["score_sensitivity"] = w_strength * cview["net_strength"].fillna(0) + w_betw * cview["net_betweenness"].fillna(0)
else:
    cview["score_sensitivity"] = cview.get("candidate_score", 0)

label_col = "drug_name" if "drug_name" in cview.columns else "drug_id"
sel = st.sidebar.selectbox("Candidate", cview.sort_values("score_sensitivity", ascending=False)[label_col].astype(str).tolist(), index=0)
row = cview[cview[label_col].astype(str) == str(sel)].iloc[0]

st.subheader(f"Selected: {row.get('drug_name', row.get('drug_id', 'N/A'))}")

# --- Metric cards ---
def safe_num(series, key, cast=float):
    try:
        return cast(series.get(key, 0))
    except Exception:
        return series.get(key, 0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Candidate score", f"{safe_num(row,'candidate_score'):.3f}" if 'candidate_score' in row else "N/A")
k2.metric("Network strength", f"{safe_num(row,'net_strength'):.3f}" if 'net_strength' in row else "N/A")
k3.metric("Betweenness", f"{safe_num(row,'net_betweenness'):.3f}" if 'net_betweenness' in row else "N/A")
phase = int(safe_num(row,'clinical_phase', int)) if 'clinical_phase' in row else None
appr  = int(safe_num(row,'approved', int)) if 'approved' in row else None
k4.metric("Phase / Approved", f"{phase if phase is not None else 'N/A'} / {appr if appr is not None else 'N/A'}")

# --- Why this drug? (evidence table) ---
p_edges = ART / "edges.csv"
if p_edges.exists():
    try:
        edges = pd.read_csv(p_edges)
        edges_drug = edges[edges["drug_id"].astype(str) == str(row.get("drug_id", row.get("drug_name")))]
        st.markdown("### Evidence: top genes")
        show_cols = [c for c in ["gene","weight","direction","source"] if c in edges_drug.columns]
        if not show_cols:
            show_cols = edges_drug.columns.tolist()
        st.dataframe(edges_drug.sort_values("weight", ascending=False).head(20)[show_cols], use_container_width=True)
    except Exception as e:
        st.warning(f"edges.csv present but could not be parsed: {e}")
else:
    st.info("Add **artifacts/edges.csv** with columns like drug_id,gene,weight,direction,source to show evidence.")

# --- Reliability diagram from val_preds.csv ---
p_preds = ART / "val_preds.csv"
if p_preds.exists():
    try:
        vp = pd.read_csv(p_preds)
        if {"y_true","y_prob"}.issubset(vp.columns):
            st.markdown("### Calibration (Reliability diagram)")
            # compute bins
            bins = st.slider("Bins", 5, 20, 10, 1)
            vp = vp.dropna(subset=["y_true","y_prob"]).clip({"y_prob": (0,1)})
            vp["bin"] = pd.cut(vp["y_prob"], bins=bins, labels=False, include_lowest=True)
            grp = vp.groupby("bin").agg(
                conf=("y_prob","mean"),
                acc=("y_true","mean"),
                n=("y_true","size")
            ).reset_index()
            fig = plt.figure()
            plt.plot([0,1],[0,1])
            plt.plot(grp["conf"], grp["acc"], marker="o")
            plt.xlabel("Predicted probability (bin mean)")
            plt.ylabel("Observed frequency")
            plt.title("Reliability diagram")
            st.pyplot(fig)
        else:
            st.info("val_preds.csv should have columns y_true,y_prob.")
    except Exception as e:
        st.warning(f"val_preds.csv present but could not be parsed: {e}")
else:
    st.info("Add **artifacts/val_preds.csv** to enable the reliability diagram.")

# --- DoE helper ---
st.markdown("### Export a mini DoE plan")
cell_lines = st.multiselect("Choose 2–4 cell contexts for a quick assay", ["A","B","C","D"], default=["A","B"])
doses = st.multiselect("Doses (µM)", [0.1, 0.5, 1, 5], default=[0.5,1])
times = st.multiselect("Timepoints (h)", [24, 48, 72], default=[48])
assays = st.multiselect("Assays", ["viability","marker"], default=["viability","marker"])

if st.button("Download DoE CSV"):
    import io
    import itertools as it
    rows = []
    for c,d,t,a in it.product(cell_lines, doses, times, assays):
        rows.append({
            "drug_id": row.get("drug_id", row.get("drug_name","")),
            "context": c, "dose_uM": d, "time_h": t, "assay": a
        })
    df = pd.DataFrame(rows)
    st.download_button("DoE_plan.csv", data=df.to_csv(index=False), file_name="DoE_plan.csv")

# --- All candidates table ---
st.markdown("### All Candidates (sensitivity score on the right)")
st.dataframe(cview.assign(score_sensitivity=cview["score_sensitivity"].round(3)), use_container_width=True)

a = cview[[label_col,"score_sensitivity"]].sort_values("score_sensitivity", ascending=False).head(10)
st.download_button("Download Top10 (sensitivity)", data=a.to_csv(index=False), file_name="top10_sensitivity.csv")
