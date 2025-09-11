#/usr/bin/env python

import pandas as pd
import numpy as np

from metric import CompetitionMetric


# Create relevant column handles
c_sol = "gesture_true"
c_subs = [f"gesture{_}" for _ in range(20)]
dataset = 'Usage'
sensors = 'all_sensors'

# Create utility function that applies competition metric in a loop
def score(df):
    metric = CompetitionMetric()
    return np.array([metric.calculate_hierarchical_f1(df, c_sol, c_sub) for c_sub in c_subs])

# Define and load aggregated table
f_aggregated = '/data/gkiar/kaggle/top20results.parquet'
df = pd.read_parquet(f_aggregated)

# Group dataframe by the the columns of interest and score each grouping
results = (df
           .groupby([dataset, sensors])
           .apply(score, include_groups=False)
           .reset_index(name="scores"))

# Compute mean and standard deviation
results['mean'] = results['scores'].apply(np.mean)
results['std'] = results['scores'].apply(np.std)

# Show the results
print(results[[dataset, sensors, 'mean', 'std']])