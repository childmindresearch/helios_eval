#/usr/bin/env python

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

from metric import CompetitionMetric

matplotlib.use('TkAgg')

# Convenience variables for the columns
c_sol = "gesture_true"
c_subs = [f"gesture{_}" for _ in range(20)]

# Create utility function that applies competition metric in a loop
def score(df):
    metric = CompetitionMetric()
    return np.array([metric.calculate_hierarchical_f1(df, c_sol, c_sub) for c_sub in c_subs])

# Create utility for plotting and saving confusion matrices
def plot_cm(cm, fname):
    fig, ax = plt.subplots(figsize=(9, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=False, ax=ax)
    plt.title("Confusion Matrix")
    plt.xticks([])
    disp.im_.set_clim([0, 250])
    plt.savefig(fname)
    plt.close('all')

# Define and load aggregated table, vars for relevant columns
f_aggregated = './data/top20results.parquet'
dataset = 'Usage'
sensors = 'all_sensors'
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

# Compute confusion matrices (and again the mean and standard deviations thereof)
y_true = df[c_sol]
metric = CompetitionMetric()
labels = metric.target_gestures + metric.non_target_gestures

# Compute confusion matrices for every submission and plot them
cms = []
for idx, c_sub in enumerate(c_subs):
    ty_pred = df[c_sub]
    cm = confusion_matrix(y_true, ty_pred)
    cms += [cm]
    plot_cm(cm, f'./data/figs/submission_{idx}.png')

# Compute mean confusion matrix
cm_agg_mn = np.mean(np.stack(cms), axis=0)
plot_cm(cm_agg_mn, './data/figs/mean.png')

# Compute coefficient of variation confusion matrix
cm_agg_std = np.std(np.stack(cms), axis=0)
plot_cm(cm_agg_std, './data/figs/std.png')