# Real data subset (small)

This project reads *embeddings*, not raw matrices, to keep size small.
Prepare four CSVs and place them in `data/real/`:

1) `gene_embeddings.csv`
   - Columns: `gene,g0,g1,...,g63` (64 dims)
   - Example first lines:
     gene,g0,g1,...,g63
     TP53,0.12,-0.03,...,0.04
     EGFR,-0.22,0.11,...,0.08

2) `cell_embeddings.csv`
   - Columns: `cell_id,c0,c1,...,c31,disease`
   - `disease` is 0/1 for control vs disease context

3) `drug_catalog.csv`
   - Columns: `drug_id,drug_name,approved,clinical_phase`

4) `train_triples.csv`
   - Columns: `gene,cell_id,drug_id,label,de_logfc`
   - `label` is a binary supervision signal (e.g., positive if the gene×cell shows desired direction under the drug).

> You can derive these from public resources by: selecting a small cohort, computing PCA/UMAP or using precomputed vectors, and exporting to the schemas above. Keep the subset **<25 MB** so it’s easy to version and view in Streamlit.
