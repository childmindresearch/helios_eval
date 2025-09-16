import marimo

__generated_with = "0.15.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Competition Metric Evaluation Dashboard""")
    return


@app.cell
def _():
    import altair as alt
    import pandas as pd

    import marimo as mo
    from src.metric import CompetitionMetric

    return CompetitionMetric, alt, mo, pd


@app.cell
def _(pd):
    # Load the data
    df = pd.read_parquet("./data/top20results.parquet")

    # Define submission columns (gesture0 through gesture19)
    submission_cols = [f"gesture{i}" for i in range(20)]
    truth_col = "gesture_true"

    df
    return df, submission_cols, truth_col


@app.cell
def _(CompetitionMetric, df, pd, submission_cols, truth_col):
    # Calculate competition metric for each submission
    competition_metric = CompetitionMetric()

    score_results = []
    for i, sub_col in enumerate(submission_cols):
        metric_score = competition_metric.calculate_hierarchical_f1(
            df, truth_col, sub_col
        )
        metric_score = round(metric_score, 5)
        score_results.append({"submission_id": i, "score": metric_score})

    results_df = pd.DataFrame(score_results)
    return (results_df,)


@app.cell
def _(results_df):
    # Create a rank dictionary that maps submission_id to rank
    rank_dict = {
        i: rank + 1
        for rank, i in enumerate(
            results_df.sort_values("score", ascending=False).submission_id
        )
    }
    return (rank_dict,)


@app.cell
def _(mo, results_df):
    # Create an interactive table
    results_table = mo.ui.table(results_df, page_size=20, selection="single")

    mo.vstack(
        [
            mo.md("### Detailed Results Table"),
            mo.md("Click on a row to see submission details below:"),
            results_table,
        ]
    )
    return (results_table,)


@app.cell
def _(alt, results_df, results_table):
    # Determine the selected submission ID from the results_table
    _selected_submission_id = -1
    if results_table.value is not None and not results_table.value.empty:
        _selected_submission_id = int(results_table.value.iloc[0]["submission_id"])

    # Define the color encoding based on whether a submission is selected
    if _selected_submission_id != -1:
        # A submission is selected, highlight it and grey out others
        color_encoding = alt.condition(
            alt.datum.submission_id == _selected_submission_id,
            alt.Color("score:Q", scale=alt.Scale(scheme="redblue")),
            alt.value("lightgrey"),
        )
    else:
        # No submission is selected, color all bars by their score
        color_encoding = alt.Color("score:Q", scale=alt.Scale(scheme="redblue"))

    # Create a bar chart of scores by submission
    score_chart = (
        alt.Chart(results_df)
        .mark_bar(clip=True)
        .encode(
            x=alt.X("submission_id:N", title="Submission ID"),
            y=alt.Y(
                "score:Q",
                title="Hierarchical Macro F1 Score",
                scale=alt.Scale(domain=[0.85, 0.9]),
            ),
            color=color_encoding,
            tooltip=["submission_id:N", "score:Q"],
        )
        .properties(
            width=800, height=400, title="Competition Metric Scores by Submission"
        )
    )

    score_chart
    return


@app.cell
def _(mo, rank_dict, results_table):
    # Show details for selected submission
    if results_table.value is not None and not results_table.value.empty:
        selected_row = results_table.value.iloc[0]
        submission_id = selected_row["submission_id"]
        submission_score = selected_row["score"]

        submission_details = mo.md(f"""
        ### Selected Submission Details

        **Submission ID:** {int(submission_id)}  
        **Competition Metric Score:** {submission_score}  
        **Rank:** {rank_dict[selected_row.name]} out of 20 submissions
        """)
    else:
        submission_details = mo.md(
            "*Select a submission from the table above to see details*"
        )

    submission_details
    return


@app.cell
def _(CompetitionMetric, df, mo, pd, results_table, truth_col):
    # Show accuracy breakdown for selected submission
    if results_table.value is not None and not results_table.value.empty:
        selected_row_data = results_table.value.iloc[0]
        _submission_id = int(selected_row_data["submission_id"])
        submission_col = "gesture" + str(_submission_id)

        # Create accuracy breakdown data
        y_true = df[truth_col]
        y_pred = df[submission_col]

        # Get unique labels
        accuracy_metric = CompetitionMetric()
        all_labels = (
            accuracy_metric.target_gestures + accuracy_metric.non_target_gestures
        )

        # Create a comprehensive accuracy breakdown
        eval_data = []
        for label in all_labels:
            # For a given label, consider it as the "positive" class
            # and all other labels as the "negative" class.

            # True Positives: actual is label, predicted is label
            tp = ((y_true == label) & (y_pred == label)).sum()

            # False Negatives: actual is label, predicted is NOT label
            fn = ((y_true == label) & (y_pred != label)).sum()

            # False Positives: actual is NOT label, predicted is label
            fp = ((y_true != label) & (y_pred == label)).sum()

            # True Negatives: actual is NOT label, predicted is NOT label
            tn = ((y_true != label) & (y_pred != label)).sum()

            # Calculate derived metrics, handling division by zero
            # Sensitivity/Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            # Precision/Positive Predictive Value
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            # False Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            # False Negative Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            # Specificity/True Negative Rate
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            # F1-Score
            f1_score = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            # Accuracy (overall for this binary classification problem)
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            # Negative Predictive Value
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            eval_data.append(
                {
                    "gesture": label,
                    "gesture_type": "Target"
                    if label in accuracy_metric.target_gestures
                    else "Non-Target",
                    "TP": tp,
                    "FN": fn,
                    "FP": fp,
                    "TN": tn,
                    "Recall": recall,
                    "Precision": precision,
                    "FPR": fpr,
                    "FNR": fnr,
                    "Specificity": specificity,
                    "F1-Score": f1_score,
                    "Accuracy": accuracy,
                    "NPV": npv,
                }
            )

        eval_df = pd.DataFrame(eval_data)

        # Sort by gesture_type and gesture
        eval_df = eval_df.sort_values(
            by=["gesture_type", "gesture"], ascending=[False, True]
        )

        # Drop the index for cleaner display
        eval_df.reset_index(drop=True, inplace=True)

        eval_breakdown = mo.vstack(
            [
                mo.md(
                    f"### Per-Gesture Evaluation Metrics for Submission {_submission_id}"
                ),
                mo.ui.table(eval_df, page_size=20, selection=None),
            ]
        )
    else:
        eval_breakdown = mo.md(
            "*Select a submission to see per-gesture performance metrics breakdown*"
        )

    eval_breakdown
    return (eval_df,)


@app.cell
def _(alt, eval_df, mo, results_table):
    if results_table.value is not None and not results_table.value.empty:
        f1_score_chart = (
            alt.Chart(eval_df)
            .mark_bar()
            .encode(
                x=alt.X("gesture:N", sort="-y", title="Gesture"),
                y=alt.Y("F1-Score:Q", title="F1 Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "gesture_type:N",
                    title="Gesture Type",
                    scale=alt.Scale(range=["#1f77b4", "#ff7f0e"]),
                ),
                tooltip=[
                    "gesture:N",
                    "gesture_type:N",
                    alt.Tooltip("F1-Score:Q", format=".3f"),
                    alt.Tooltip("Precision:Q", format=".3f"),
                    alt.Tooltip("Recall:Q", format=".3f"),
                ],
            )
            .properties(
                title=f"Per-Gesture F1 Scores for Submission {int(results_table.value.iloc[0]['submission_id'])}"
            )
            .interactive()
        )
    else:
        f1_score_chart = mo.md("*Select a submission to see per-gesture F1 scores*")

    f1_score_chart
    return


if __name__ == "__main__":
    app.run()
