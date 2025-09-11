#!/usr/bin/env python

import os.path as op
import pandas as pd
import numpy as np
import os


# List of relevant files
bp = './data'

# List of all local parquet filepaths for each competitor
f_submission_files = sorted([
                                op.join(bp, 'submissions', fl)
                                for fl in os.listdir(op.join(bp, 'submissions'))
                            ])
f_solution = op.join(bp, 'solution.csv')  # True solution for each sequence
f_test_sequences = op.join(bp, 'test.csv')  # Test CSV file that competitors used
f_aggregated = op.join(bp, 'top20results.parquet')  # File where we will store results

# For every submission, load the file and add it to a list.
df_subs = [pd.read_parquet(sub)
           for sub in f_submission_files] 

# Seed the table we'll merge the results into
df_agg = df_subs[0]
for idx, tdf in enumerate(df_subs[1:]):
    df_agg = df_agg.merge(tdf, on="sequence_id", suffixes=["", f"{idx+1}"])  # Merge on sequence ID, add index to predictions

# Relabel the first index to make it similar to the rest
df_agg.rename(columns={"gesture": "gesture0"}, inplace=True)

# Load the solutions table, and add it into the aggregate table
df_sol = pd.read_csv(f_solution)
df_agg = df_agg.merge(df_sol, on="sequence_id")

# Relabel the gesture column to clearly indicate true value
df_agg.rename(columns={"gesture": "gesture_true"}, inplace=True)

# Load the complete series of test data
df_test = pd.read_csv(f_test_sequences)

# For every sequence ID, evaluate if the thermopile data (thm_1) were real or NaNs
seq_all_sensors = []
for seq in df_test['sequence_id'].unique():
    tdf = df_test.query(f'sequence_id == "{seq}"')
    tthm_mn = tdf['thm_1'].mean()
    seq_all_sensors += [{"sequence_id": seq,
                         "all_sensors": bool(tthm_mn < np.inf)}]

# Add data completeness 
df_agg = df_agg.merge(pd.DataFrame.from_dict(seq_all_sensors), on="sequence_id")

# Save to file
df_agg.to_parquet(f_aggregated)
