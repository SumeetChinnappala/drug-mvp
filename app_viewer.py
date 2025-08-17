import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Drug Discovery — MVP", layout="wide")
st.title("Drug Discovery — MVP")

p = Path("artifacts") / "candidates.csv"
if not p.exists():
    st.error("No candidates found. Add artifacts/candidates.csv to this repo.")
    st.info("GitHub → Add file → Create new file → artifacts/candidates.csv (paste a few rows).")
    st.stop()

cands = pd.read_csv(p)
if cands.empty:
    st.warning("candidates.csv is empty.")
    st.stop()

label_col = "drug_name" if "drug_name" in cands.columns else "drug_id"
sel = st.sidebar.selectbox("Candidate", cands[label_col].astype(str).tolist(), index=0)
row = cands[cands[label_col].astype(str) == str(sel)].iloc[0]

st.subheader(f"Selected: {row.get('drug_name', row.get('drug_id', 'N/A'))}")

def safe(k, cast=float, default="N/A"):
    try: return cast(row.get(k, default))
    except Exception: return default

c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidate score", f"{safe('candidate_score'):.3f}" if 'candidate_score' in cands.columns else "N/A")
c2.metric("Network strength", f"{safe('net_strength'):.3f}" if 'net_strength' in cands.columns else "N/A")
c3.metric("Betweenness", f"{safe('net_betweenness'):.3f}" if 'net_betweenness' in cands.columns else "N/A")
c4.metric("Phase / Approved",
          f"{int(safe('clinical_phase', int, 0)) if 'clinical_phase' in cands.columns else 'N/A'} / "
          f"{int(safe('approved', int, 0)) if 'approved' in cands.columns else 'N/A'}")

st.markdown("### Rationales")
for k, title in [("rationale_business","Business"),("rationale_stat","Statistical"),
                 ("rationale_bio","Biological"),("rationale_reg","Regulatory")]:
    if k in cands.columns:
        st.markdown(f"**{title}:** {row.get(k, '')}")

st.markdown("### All Candidates")
st.dataframe(cands, use_container_width=True)

st.download_button("Download candidates.csv", data=cands.to_csv(index=False), file_name="candidates.csv")
