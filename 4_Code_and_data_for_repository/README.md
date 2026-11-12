# Code and data for "Adaptation strategy outweighs backbone choice in transfer learning for paddy disease recognition"

- run_experiments.py: the exact script that produced all results (run on Kaggle Notebooks, GPU T4 x2).
  Stages: train, embed, submit, report. Data: Kaggle competition "paddy-disease-classification".
- results/splits.csv: the fixed stratified 70/15/15 split (seed 2026) used by every model.
- results/runs/*.json: per-run training histories, test metrics, confusion matrices, and predictions (5 configurations x 3 seeds).
- results/emb/*.npy: test-split embeddings used for the feature-space analysis.
- results/tab_*.tex, results_summary.json, fig_*.png: generated tables, summary values, and figures.
- results/config_used.json: all hyperparameters.
