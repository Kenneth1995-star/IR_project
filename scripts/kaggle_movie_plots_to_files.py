# scripts/kaggle_movie_plots_to_files.py
# Here, we are converting the Kaggle Wikipedia Movies CSV file into one text file per movie plot.
# This ensures compatibility with run_demo.py and Indexer (each .txt = one document).

import pandas as pd
import os

# here, we are defining input and output paths
raw_csv = "data/raw/wikipedia-movies.csv"
out_dir = "data/sample_docs"
os.makedirs(out_dir, exist_ok=True)

print("=== Loading Wikipedia Movies dataset from Kaggle ===")
df = pd.read_csv(raw_csv, encoding="utf8", low_memory=False)

# here, we are checking which columns exist
cols = df.columns
plot_col = "Plot" if "Plot" in cols else "plot"
title_col = "Title" if "Title" in cols else "title"

count = 0
for i, row in df.iterrows():
    plot = str(row.get(plot_col, "")).strip()
    title = str(row.get(title_col, f"movie_{i}"))
    if not plot:
        continue
    fname = f"movie_{i+1:05d}.txt"
    with open(os.path.join(out_dir, fname), "w", encoding="utf8", errors="ignore") as f:
        # here, we are writing both title and plot
        f.write(f"{title}\n\n{plot}")
    count += 1

print(f" Created {count} individual .txt documents in {out_dir}")
print("You can now run 'python run_demo.py' to build the index.")
