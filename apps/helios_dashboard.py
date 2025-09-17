import marimo

__generated_with = "0.15.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Helios Kaggle Competition - Evaluation Dashboard""")
    return


@app.cell
def _():
    import altair as alt
    import pandas as pd
    from sklearn.metrics import f1_score

    import marimo as mo
    return alt, f1_score, mo, pd


@app.cell
def _(f1_score, pd):
    class CompetitionMetric:
        """Hierarchical macro F1 for the CMI 2025 challenge."""
        def __init__(self):
            self.target_gestures = [
                'Above ear - pull hair',
                'Cheek - pinch skin',
                'Eyebrow - pull hair',
                'Eyelash - pull hair',
                'Forehead - pull hairline',
                'Forehead - scratch',
                'Neck - pinch skin',
                'Neck - scratch',
            ]
            self.non_target_gestures = [
                'Write name on leg',
                'Wave hello',
                'Glasses on/off',
                'Text on phone',
                'Write name in air',
                'Feel around in tray and pull out an object',
                'Scratch knee/leg skin',
                'Pull air toward your face',
                'Drink from bottle/cup',
                'Pinch knee/leg skin'
            ]
            self.all_classes = self.target_gestures + self.non_target_gestures

        def calculate_hierarchical_f1(
            self,
            df: pd.DataFrame,
            sol_col: str,
            sub_col: str
        ) -> float:

            # Validate gestures
            invalid_types = {i for i in df[sub_col].unique() if i not in self.all_classes}
            if invalid_types:
                raise ValueError(
                    f"Invalid gesture values in submission: {invalid_types}"
                )

            # Compute binary F1 (Target vs Non-Target)
            y_true_bin = df[sol_col].isin(self.target_gestures).values
            y_pred_bin = df[sub_col].isin(self.target_gestures).values
            f1_binary = f1_score(
                y_true_bin,
                y_pred_bin,
                pos_label=True,
                zero_division=0,
                average='binary'
            )

            # Build multi-class labels for gestures
            y_true_mc = df[sol_col].apply(lambda x: x if x in self.target_gestures else 'non_target')
            y_pred_mc = df[sub_col].apply(lambda x: x if x in self.target_gestures else 'non_target')

            # Compute macro F1 over all gesture classes
            f1_macro = f1_score(
                y_true_mc,
                y_pred_mc,
                average='macro',
                zero_division=0
            )

            return 0.5 * f1_binary + 0.5 * f1_macro
    return (CompetitionMetric,)


@app.function
def calculate_metrics_from_confusion_matrix(tp, fn, fp, tn):
    """Calculate all evaluation metrics from confusion matrix values."""
    # Calculate derived metrics, handling division by zero
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "Recall": round(recall, 5),
        "Precision": round(precision, 5),
        "FPR": round(fpr, 5),
        "FNR": round(fnr, 5),
        "Specificity": round(specificity, 5),
        "F1-Score": round(f1_score, 5),
        "Accuracy": round(accuracy, 5),
        "NPV": round(npv, 5),
    }


@app.function
def calculate_binary_metrics(y_true, y_pred, positive_class):
    """Calculate binary classification metrics for a given positive class."""
    # True Positives: actual is positive_class, predicted is positive_class
    tp = ((y_true == positive_class) & (y_pred == positive_class)).sum()
    # False Negatives: actual is positive_class, predicted is NOT positive_class
    fn = ((y_true == positive_class) & (y_pred != positive_class)).sum()
    # False Positives: actual is NOT positive_class, predicted is positive_class
    fp = ((y_true != positive_class) & (y_pred == positive_class)).sum()
    # True Negatives: actual is NOT positive_class, predicted is NOT positive_class
    tn = ((y_true != positive_class) & (y_pred != positive_class)).sum()

    return calculate_metrics_from_confusion_matrix(tp, fn, fp, tn)


@app.function
def calculate_collapsed_non_target_metrics(y_true, y_pred, non_target_gestures):
    """Calculate metrics for all non-target gestures treated as one class."""
    actual_non_target = y_true.isin(non_target_gestures)
    pred_non_target = y_pred.isin(non_target_gestures)

    tp = (actual_non_target & pred_non_target).sum()
    fn = (actual_non_target & ~pred_non_target).sum()
    fp = (~actual_non_target & pred_non_target).sum()
    tn = (~actual_non_target & ~pred_non_target).sum()

    return calculate_metrics_from_confusion_matrix(tp, fn, fp, tn)


@app.function
def apply_data_filters(
    df, public_filter, private_filter, all_sensors_filter, imu_sensors_filter
):
    """Apply the data filters to the dataframe."""
    filtered_df = df.copy()

    # Filter by public/private
    public_conditions = []
    if public_filter.value:
        public_conditions.append(filtered_df["public"])
    if private_filter.value:
        public_conditions.append(~filtered_df["public"])

    # Filter by sensor completeness
    sensor_conditions = []
    if all_sensors_filter.value:
        sensor_conditions.append(filtered_df["all_sensors"])
    if imu_sensors_filter.value:
        sensor_conditions.append(~filtered_df["all_sensors"])

    # Combine conditions - must match at least one from each category
    if public_conditions and sensor_conditions:
        public_mask = public_conditions[0]
        for cond in public_conditions[1:]:
            public_mask = public_mask | cond

        sensor_mask = sensor_conditions[0]
        for cond in sensor_conditions[1:]:
            sensor_mask = sensor_mask | cond

        filtered_df = filtered_df[public_mask & sensor_mask]
    elif public_conditions:
        public_mask = public_conditions[0]
        for cond in public_conditions[1:]:
            public_mask = public_mask | cond
        filtered_df = filtered_df[public_mask]
    elif sensor_conditions:
        sensor_mask = sensor_conditions[0]
        for cond in sensor_conditions[1:]:
            sensor_mask = sensor_mask | cond
        filtered_df = filtered_df[sensor_mask]

    return filtered_df


@app.function
def create_filter_summary(
    public_filter,
    private_filter,
    all_sensors_filter,
    imu_sensors_filter,
    collapse_non_target_filter,
):
    """Create a human-readable summary of the applied filters."""
    filter_parts = []

    if public_filter.value and private_filter.value:
        filter_parts.append("Public + Private")
    elif public_filter.value:
        filter_parts.append("Public only")
    elif private_filter.value:
        filter_parts.append("Private only")

    if all_sensors_filter.value and imu_sensors_filter.value:
        filter_parts.append("All sensors + only IMU sensors")
    elif all_sensors_filter.value:
        filter_parts.append("All sensors only")
    elif imu_sensors_filter.value:
        filter_parts.append("IMU sensors only")

    if collapse_non_target_filter.value:
        filter_parts.append("Non-target gestures collapsed")

    return (
        " | ".join(filter_parts)
        if filter_parts
        else "No filter selected, including the whole dataset"
    )


@app.cell
def _(mo, pd):
    # Load the data
    df = pd.read_parquet("./data/top20results_anonymized.parquet")

    # Convert Usage column from object to boolean (True for public, False for private)
    df["public"] = df["Usage"] == "Public"
    df = df.drop(columns=["Usage"])

    mo.vstack(
        [
            mo.md("## Top 20 Submission Results"),
            df,
        ]
    )
    return (df,)


@app.cell
def _(CompetitionMetric, df, pd):
    # Define submission columns (gesture0 through gesture19)
    submission_cols = [f"gesture{i}" for i in range(20)]
    truth_col = "gesture_true"

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

    # Create a rank dictionary that maps submission_id to rank
    rank_dict = {
        i: rank + 1
        for rank, i in enumerate(
            results_df.sort_values("score", ascending=False).submission_id
        )
    }
    return competition_metric, rank_dict, results_df, truth_col


@app.cell
def _(alt, mo, results_df):
    # Create an interactive bar chart with selection capabilities
    click_selector = alt.selection_point()

    # Create a bar chart of scores by submission with interactive selection
    score_chart = mo.ui.altair_chart(
        alt.Chart(results_df)
        .mark_bar(clip=True)
        .add_params(click_selector)
        .encode(
            x=alt.X("submission_id:N", title="Submission ID"),
            y=alt.Y(
                "score:Q",
                title="Hierarchical Macro F1 Score",
                scale=alt.Scale(domain=[0.85, 0.9]),
            ),
            color=alt.condition(
                click_selector,
                alt.Color("score:Q", scale=alt.Scale(scheme="redblue")),
                alt.value("lightgrey"),
            ),
            stroke=alt.condition(click_selector, alt.value("black"), alt.value(None)),
            strokeWidth=alt.condition(click_selector, alt.value(1), alt.value(0)),
            tooltip=["submission_id:N", "score:Q"],
        )
        .properties(
            width=800,
            height=400,
            title="Competition Metric Scores by Submission",
        )
    )

    mo.vstack(
        [
            mo.md("## Hierarchical Macro F1 Scores by Submission"),
            mo.md("Click on a bar to see submission details below:"),
            score_chart,
        ]
    )
    return (score_chart,)


@app.cell
def _(mo, rank_dict, results_df, score_chart):
    # Show details for selected submission from chart
    if score_chart.value is not None and not score_chart.value.empty:
        _submission_id = int(score_chart.value.iloc[0]["submission_id"])
        submission_score = results_df[results_df["submission_id"] == _submission_id][
            "score"
        ].iloc[0]

        submission_details = mo.md(f"""
        ### Selected Submission Details

        **Submission ID:** {_submission_id}  
        **Competition Metric Score:** {submission_score}  
        **Rank:** {rank_dict[_submission_id]} out of 20 submissions
        """)
    else:
        submission_details = mo.md(
            "*Select a submission from the chart above to see details*"
        )

    submission_details
    return


@app.cell
def _(mo, score_chart):
    # Filter controls for evaluation breakdown
    public_filter = mo.ui.switch(value=True, label="Include Public dataset")
    private_filter = mo.ui.switch(value=True, label="Include Private dataset")
    all_sensors_filter = mo.ui.switch(value=True, label="Include All sensors data")
    imu_sensors_filter = mo.ui.switch(value=True, label="Include only IMU sensors data")
    collapse_non_target_filter = mo.ui.switch(
        value=True, label="Collapse all non-target gestures into one row"
    )

    # Only show filters if a submission is selected
    if score_chart.value is not None and not score_chart.value.empty:
        filters_display = mo.vstack(
            [
                mo.md("### Evaluation Filters"),
                mo.md(
                    "Select which data subsets to include in the per-gesture evaluation breakdown:"
                ),
                mo.hstack(
                    [public_filter, private_filter], justify="start", widths=[1, 2]
                ),
                mo.hstack(
                    [all_sensors_filter, imu_sensors_filter],
                    justify="start",
                    widths=[1, 2],
                ),
                collapse_non_target_filter,
            ]
        )
    else:
        filters_display = mo.md(
            "*Select a submission from the chart above to configure evaluation filters*"
        )

    filters_display
    return (
        all_sensors_filter,
        collapse_non_target_filter,
        imu_sensors_filter,
        private_filter,
        public_filter,
    )


@app.cell
def _(
    all_sensors_filter,
    collapse_non_target_filter,
    competition_metric,
    df,
    imu_sensors_filter,
    mo,
    pd,
    private_filter,
    public_filter,
    score_chart,
    truth_col,
):
    # Show accuracy breakdown for selected submission from chart
    if score_chart.value is not None and not score_chart.value.empty:
        _submission_id = int(score_chart.value.iloc[0]["submission_id"])
        submission_col = f"gesture{_submission_id}"

        # Apply filters to the dataframe
        filtered_df = apply_data_filters(
            df, public_filter, private_filter, all_sensors_filter, imu_sensors_filter
        )

        # Create accuracy breakdown data from filtered data
        y_true = filtered_df[truth_col]
        y_pred = filtered_df[submission_col]

        eval_data = []
        if collapse_non_target_filter.value:
            # Calculate metrics for target gestures individually
            for label in competition_metric.target_gestures:
                metrics = calculate_binary_metrics(y_true, y_pred, label)
                eval_data.append(
                    {"gesture": label, "gesture_type": "Target", **metrics}
                )

            # Calculate collapsed metrics for all non-target gestures
            collapsed_metrics = calculate_collapsed_non_target_metrics(
                y_true, y_pred, competition_metric.non_target_gestures
            )
            eval_data.append(
                {
                    "gesture": "All non-target gestures",
                    "gesture_type": "Non-Target",
                    **collapsed_metrics,
                }
            )
        else:
            # Original behavior - calculate metrics for each gesture individually
            all_labels = (
                competition_metric.target_gestures
                + competition_metric.non_target_gestures
            )
            for label in all_labels:
                metrics = calculate_binary_metrics(y_true, y_pred, label)
                gesture_type = (
                    "Target"
                    if label in competition_metric.target_gestures
                    else "Non-Target"
                )
                eval_data.append(
                    {"gesture": label, "gesture_type": gesture_type, **metrics}
                )

        eval_df = pd.DataFrame(eval_data)

        # Sort by gesture_type and gesture
        eval_df = eval_df.sort_values(
            by=["gesture_type", "gesture"], ascending=[False, True]
        )
        eval_df.reset_index(drop=True, inplace=True)

        # Create display elements
        _filter_summary = create_filter_summary(
            public_filter,
            private_filter,
            all_sensors_filter,
            imu_sensors_filter,
            collapse_non_target_filter,
        )
        sample_count = len(filtered_df)

        eval_breakdown = mo.vstack(
            [
                mo.md(
                    f"### Per-Gesture Evaluation Metrics for Submission {_submission_id}"
                ),
                mo.md(
                    f"**Filters applied:** {_filter_summary} | **Samples:** {sample_count}"
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
def _(
    all_sensors_filter,
    alt,
    collapse_non_target_filter,
    eval_df,
    imu_sensors_filter,
    mo,
    private_filter,
    public_filter,
    score_chart,
):
    if score_chart.value is not None and not score_chart.value.empty:
        _submission_id = int(score_chart.value.iloc[0]["submission_id"])

        # Create filter summary for chart title
        _filter_summary = create_filter_summary(
            public_filter,
            private_filter,
            all_sensors_filter,
            imu_sensors_filter,
            collapse_non_target_filter,
        )

        # Calculate min and max F1 scores for auto-scaling
        min_f1 = eval_df["F1-Score"].min()
        max_f1 = eval_df["F1-Score"].max()

        # Add some padding (5% on each side)
        padding = (max_f1 - min_f1) * 0.05
        y_min = max(0, min_f1 - padding)  # Don't go below 0
        y_max = min(1, max_f1 + padding)  # Don't go above 1

        f1_score_chart = (
            alt.Chart(eval_df)
            .mark_bar(clip=True)
            .encode(
                x=alt.X("gesture:N", sort="-y", title="Gesture"),
                y=alt.Y(
                    "F1-Score:Q",
                    title="F1 Score",
                    scale=alt.Scale(domain=[y_min, y_max]),
                ),
                color=alt.Color(
                    "gesture_type:N",
                    title="Gesture Type",
                    scale=alt.Scale(range=["#1f77b4", "#ff7f0e"]),
                    sort=["Target", "Non-Target"],
                ),
                tooltip=[
                    alt.Tooltip("gesture:N", title="Gesture"),
                    alt.Tooltip("gesture_type:N", title="Gesture Type"),
                    alt.Tooltip("F1-Score:Q", format=".3f"),
                    alt.Tooltip("Accuracy:Q", format=".3f"),
                    alt.Tooltip("Precision:Q", format=".3f"),
                    alt.Tooltip("Recall:Q", format=".3f"),
                    alt.Tooltip("Specificity:Q", format=".3f"),
                    alt.Tooltip("NPV:Q", format=".3f", title="Negative Predictive Value"),
                    alt.Tooltip("FNR:Q", title="False Negative Rate"),
                    alt.Tooltip("FPR:Q", title="False Positive Rate"),
                ],
            )
            .properties(
                title=f"Per-Gesture F1 Scores for Submission {_submission_id} ({_filter_summary})"
            )
            .interactive()
        )
    else:
        f1_score_chart = mo.md("*Select a submission to see per-gesture F1 scores*")

    f1_score_chart
    return


@app.cell
def _(alt, competition_metric, df, mo, pd, score_chart, truth_col):
    if score_chart.value is not None and not score_chart.value.empty:
        _submission_id = int(score_chart.value.iloc[0]["submission_id"])
        _submission_col = f"gesture{_submission_id}"

        # Create comprehensive dataset for all combinations
        boxplot_data = []

        # Define the combinations for 2x2 grid
        data_combinations = [
            ("Public", "All sensors", True, True),
            ("Public", "IMU only", True, False),
            ("Private", "All sensors", False, True),
            ("Private", "IMU only", False, False),
        ]

        for dataset_label, sensor_label, is_public, is_all_sensors in data_combinations:
            # Filter data for this combination
            subset_df = df[
                (df["public"] == is_public) & (df["all_sensors"] == is_all_sensors)
            ]

            if len(subset_df) == 0:
                continue

            _y_true = subset_df[truth_col]
            _y_pred = subset_df[_submission_col]

            # Calculate metrics for all gestures individually
            _all_labels = (
                competition_metric.target_gestures
                + competition_metric.non_target_gestures
            )

            for _label in _all_labels:
                _metrics = calculate_binary_metrics(_y_true, _y_pred, _label)
                _gesture_type = (
                    "Target"
                    if _label in competition_metric.target_gestures
                    else "Non-Target"
                )

                # Extract only the calculated metrics (excluding TP, TN, FP, FN)
                calculated_metrics = {
                    "Recall": _metrics["Recall"],
                    "Precision": _metrics["Precision"],
                    "FPR": _metrics["FPR"],
                    "FNR": _metrics["FNR"],
                    "Specificity": _metrics["Specificity"],
                    "F1-Score": _metrics["F1-Score"],
                    "Accuracy": _metrics["Accuracy"],
                    "NPV": _metrics["NPV"],
                }

                for metric_name, metric_value in calculated_metrics.items():
                    boxplot_data.append(
                        {
                            "dataset": dataset_label,
                            "sensor_type": sensor_label,
                            "gesture": _label,
                            "gesture_type": _gesture_type,
                            "metric": metric_name,
                            "value": metric_value,
                        }
                    )

        boxplot_df = pd.DataFrame(boxplot_data)

        if len(boxplot_df) > 0:
            # Create the 2x2 grid of boxplots
            boxplot_chart = (
                alt.Chart(boxplot_df)
                .mark_boxplot(size=15, outliers={"size": 10, "opacity": 0.6})
                .encode(
                    x=alt.X(
                        "metric:N",
                        title="Evaluation Metric",
                        axis=alt.Axis(labelAngle=45),
                        sort=[
                            "F1-Score",
                            "Accuracy",
                            "Precision",
                            "Recall",
                            "Specificity",
                            "NPV",
                            "FNR",
                            "FPR",
                        ],
                    ),
                    xOffset=alt.XOffset(
                        "gesture_type:N",
                        title="Gesture Type",
                        sort=["Target", "Non-Target"],
                    ),
                    y=alt.Y(
                        "value:Q", title="Metric Value", scale=alt.Scale(domain=[0, 1])
                    ),
                    color=alt.Color(
                        "gesture_type:N",
                        title="Gesture Type",
                        scale=alt.Scale(range=["#1f77b4", "#ff7f0e"]),
                        sort=["Target", "Non-Target"],
                    ),
                    column=alt.Column(
                        "sensor_type:N",
                        title="Sensor Configuration",
                        header=alt.Header(titleFontSize=14, labelFontSize=12),
                    ),
                    row=alt.Row(
                        "dataset:N",
                        title="Dataset Split",
                        header=alt.Header(titleFontSize=14, labelFontSize=12),
                    ),
                    tooltip=[
                        alt.Tooltip("dataset:N", title="Dataset"),
                        alt.Tooltip("sensor_type:N", title="Sensor Type"),
                        alt.Tooltip("metric:N", title="Metric"),
                        alt.Tooltip("gesture_type:N", title="Gesture Type"),
                        alt.Tooltip("value:Q", format=".3f", title="Value"),
                    ],
                )
                .properties(
                    width=400,
                    height=300,
                )
                .resolve_scale(y="independent")
            )

            boxplot_display = mo.vstack(
                [
                    mo.md(
                        f"### Metric Distribution Analysis for Submission {_submission_id}"
                    ),
                    mo.md(
                        "Boxplots showing the distribution of evaluation metrics across different data subsets and gesture types:"
                    ),
                    boxplot_chart,
                ]
            )
        else:
            boxplot_display = mo.md("No data available for boxplot analysis")
    else:
        boxplot_display = mo.md(
            "*Select a submission to see metric distribution analysis*"
        )

    boxplot_display
    return


if __name__ == "__main__":
    app.run()
