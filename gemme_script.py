"""Use this script to create an input dataframe for FSFP in the correct format required by the code in Pro-FSFP repo.

`gemme_dir` should point to the directory containing GEMME scores for your dataset.
`target_dir` should point to the directory where you want the final dataframe to be saved; it should contain raw ProteinGym data for the task.
Generally, this is a duckt-tape solution necessary because ProteinGym's own df merging scripts currently don't work.
"""
from pathlib import Path

import pandas as pd
from tqdm import tqdm

gemme_dir = "..."
target_dir = "..."

gemme_files = set(x.name for x in Path(gemme_dir).iterdir())
target_files = set(x.name for x in Path(target_dir).iterdir())
assert gemme_files == target_files

for fname in tqdm(gemme_files):
    target_df = pd.read_csv(Path(target_dir) / fname)
    gemme_df = pd.read_csv(Path(gemme_dir) / fname)
    target_df["GEMME"] = gemme_df["GEMME_score"]
    target_df.to_csv(Path(target_dir) / fname, index=False)
