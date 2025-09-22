import marimo

__generated_with = "0.15.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Helios Kaggle Competition - Evaluation Dashboard""")
    return


@app.cell
def _():
    # Standard library imports
    # Third-party imports
    import altair as alt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score

    import marimo as mo
    return alt, f1_score, mo, np, pd


@app.cell
def _():
    # Constants and configuration
    DATA_URL = "https://raw.githubusercontent.com/childmindresearch/helios_eval/refs/heads/main/data/top20results_anonymized.csv"

    # Chart dimensions
    CHART_WIDTH = 800
    CHART_HEIGHT = 400
    BOXPLOT_WIDTH = 300
    BOXPLOT_HEIGHT = 300

    # Bootstrap configuration
    DEFAULT_BOOTSTRAP_SAMPLES = 100
    CONFIDENCE_INTERVAL_PERCENTILES = [2.5, 97.5]

    # Submission configuration
    NUM_SUBMISSIONS = 20

    # Chart color schemes
    CHART_COLOR_SCHEME = "redblue"
    DELTA_COLOR_SCHEME = "reds"
    GESTURE_TYPE_COLORS = ["#1f77b4", "#ff7f0e"]  # Target, Non-Target

    # Metric display order
    METRIC_DISPLAY_ORDER = [
        "F1-Score",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "NPV",
    ]

    # Precision for rounding metrics
    METRIC_PRECISION = 5
    return (
        BOXPLOT_HEIGHT,
        BOXPLOT_WIDTH,
        CHART_COLOR_SCHEME,
        CHART_HEIGHT,
        CHART_WIDTH,
        CONFIDENCE_INTERVAL_PERCENTILES,
        DATA_URL,
        DEFAULT_BOOTSTRAP_SAMPLES,
        DELTA_COLOR_SCHEME,
        GESTURE_TYPE_COLORS,
        METRIC_DISPLAY_ORDER,
        METRIC_PRECISION,
        NUM_SUBMISSIONS,
    )


@app.cell
def _(f1_score, mo):
    class CompetitionMetric:
        """Hierarchical macro F1 for the CMI 2025 challenge."""

        def __init__(self):
            self.target_gestures = [
                "Above ear - pull hair",
                "Cheek - pinch skin",
                "Eyebrow - pull hair",
                "Eyelash - pull hair",
                "Forehead - pull hairline",
                "Forehead - scratch",
                "Neck - pinch skin",
                "Neck - scratch",
            ]
            self.non_target_gestures = [
                "Write name on leg",
                "Wave hello",
                "Glasses on/off",
                "Text on phone",
                "Write name in air",
                "Feel around in tray and pull out an object",
                "Scratch knee/leg skin",
                "Pull air toward your face",
                "Drink from bottle/cup",
                "Pinch knee/leg skin",
            ]
            self.all_classes = self.target_gestures + self.non_target_gestures

        @mo.persistent_cache
        def calculate_hierarchical_f1(self, df, sol_col: str, sub_col: str) -> float:
            # Validate gestures
            invalid_types = {
                i for i in df[sub_col].unique() if i not in self.all_classes
            }
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
                average="binary",
            )

            # Build multi-class labels for gestures
            y_true_mc = df[sol_col].apply(
                lambda x: x if x in self.target_gestures else "non_target"
            )
            y_pred_mc = df[sub_col].apply(
                lambda x: x if x in self.target_gestures else "non_target"
            )

            # Compute macro F1 over all gesture classes
            f1_macro = f1_score(y_true_mc, y_pred_mc, average="macro", zero_division=0)

            return 0.5 * f1_binary + 0.5 * f1_macro
    return (CompetitionMetric,)


@app.cell
def _(mo):
    @mo.persistent_cache
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
    return (calculate_metrics_from_confusion_matrix,)


@app.cell
def _(calculate_metrics_from_confusion_matrix, mo):
    @mo.persistent_cache
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
    return (calculate_binary_metrics,)


@app.cell
def _(calculate_metrics_from_confusion_matrix, mo):
    @mo.persistent_cache
    def calculate_collapsed_metrics(y_true, y_pred, gesture_list):
        """Calculate metrics for a list of gestures treated as one class."""
        actual_positive = y_true.isin(gesture_list)
        pred_positive = y_pred.isin(gesture_list)

        tp = (actual_positive & pred_positive).sum()
        fn = (actual_positive & ~pred_positive).sum()
        fp = (~actual_positive & pred_positive).sum()
        tn = (~actual_positive & ~pred_positive).sum()

        return calculate_metrics_from_confusion_matrix(tp, fn, fp, tn)
    return (calculate_collapsed_metrics,)


@app.cell
def _(calculate_binary_metrics, calculate_metrics_from_confusion_matrix, mo):
    @mo.persistent_cache
    def calculate_macro_averaged_metrics(y_true, y_pred, gesture_list):
        """Calculate true macro-averaged metrics across a list of gestures.

        This computes metrics for each individual gesture separately, then averages
        ALL resulting metrics (including confusion matrix counts), giving equal
        importance to each gesture regardless of sample count.

        Note: True macro-averaging averages the final calculated metrics, not the
        confusion matrix counts. The averaged confusion matrix counts are provided
        for reference but should be interpreted carefully as they represent the
        average per-gesture counts, not the total counts.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            gesture_list: List of gesture labels to include in macro-average

        Returns:
            Dict with macro-averaged metrics
        """
        individual_metrics = []

        # Calculate metrics for each gesture individually
        for gesture in gesture_list:
            gesture_metrics = calculate_binary_metrics(y_true, y_pred, gesture)
            individual_metrics.append(gesture_metrics)

        if not individual_metrics:
            # Return zeros if no valid gestures
            return calculate_metrics_from_confusion_matrix(0, 0, 0, 0)

        # True macro-average: average all final calculated metrics across gestures
        # Do NOT sum confusion matrix counts - that would be micro-averaging
        macro_metrics = {}
        metric_keys = individual_metrics[0].keys()
        n_gestures = len(individual_metrics)

        for key in metric_keys:
            # Average ALL metrics across gestures for true macro-averaging
            macro_metrics[key] = sum(m[key] for m in individual_metrics) / n_gestures

        # Round the averaged metrics to maintain precision
        for key in macro_metrics:
            if key not in ["TP", "FN", "FP", "TN"]:
                macro_metrics[key] = round(macro_metrics[key], 5)
            else:
                # Round confusion matrix counts to integers
                macro_metrics[key] = round(macro_metrics[key])

        return macro_metrics
    return (calculate_macro_averaged_metrics,)


@app.cell
def _(mo):
    @mo.persistent_cache
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
            for _cond in public_conditions[1:]:
                public_mask = public_mask | _cond

            sensor_mask = sensor_conditions[0]
            for _cond in sensor_conditions[1:]:
                sensor_mask = sensor_mask | _cond

            filtered_df = filtered_df[public_mask & sensor_mask]
        elif public_conditions:
            public_mask = public_conditions[0]
            for _cond in public_conditions[1:]:
                public_mask = public_mask | _cond
            filtered_df = filtered_df[public_mask]
        elif sensor_conditions:
            sensor_mask = sensor_conditions[0]
            for _cond in sensor_conditions[1:]:
                sensor_mask = sensor_mask | _cond
            filtered_df = filtered_df[sensor_mask]

        return filtered_df

    @mo.persistent_cache
    def build_filter_masks(df, public_filter, private_filter):
        """Helper function to build filter masks for public/private data splits."""
        masks = []
        if public_filter.value:
            masks.append(df["public"])
        if private_filter.value:
            masks.append(~df["public"])

        if masks:
            mask = masks[0]
            for _m in masks[1:]:
                mask = mask | _m
            return df[mask]
        else:
            return df
    return apply_data_filters, build_filter_masks


@app.cell
def _(mo):
    @mo.persistent_cache
    def create_filter_summary(
        public_filter,
        private_filter,
        all_sensors_filter,
        imu_sensors_filter,
        collapse_non_target_filter,
        collapse_target_filter,
        collapse_submissions_filter,
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
            filter_parts.append("All-sensors + Only-IMU-sensors")
        elif all_sensors_filter.value:
            filter_parts.append("All-sensors only")
        elif imu_sensors_filter.value:
            filter_parts.append("Only-IMU-sensors only")

        if collapse_non_target_filter.value:
            filter_parts.append("Non-target gestures collapsed")

        if collapse_target_filter.value:
            filter_parts.append("Target gestures collapsed")

        if collapse_submissions_filter.value:
            filter_parts.append("Submissions collapsed")

        return (
            " | ".join(filter_parts)
            if filter_parts
            else "No filter selected, including the whole dataset"
        )
    return (create_filter_summary,)


@app.cell
def _(mo, pd):
    @mo.persistent_cache
    def stack_submissions(df, truth_col, submission_cols):
        """Stack all submissions into a long-form dataframe for collapsed analysis.

        Args:
            df: DataFrame with data
            truth_col: Column name for ground truth
            submission_cols: List of submission column names

        Returns:
            Long-form DataFrame with columns: truth, prediction, submission_id, public, all_sensors
        """
        stacked_rows = []

        for idx, row in df.iterrows():
            truth_value = row[truth_col]
            public_value = row["public"]
            all_sensors_value = row["all_sensors"]

            for sub_col in submission_cols:
                submission_id = int(sub_col.replace("gesture", ""))
                prediction_value = row[sub_col]

                stacked_rows.append(
                    {
                        "truth": truth_value,
                        "prediction": prediction_value,
                        "submission_id": submission_id,
                        "public": public_value,
                        "all_sensors": all_sensors_value,
                    }
                )

        return pd.DataFrame(stacked_rows)
    return (stack_submissions,)


@app.cell
def _(
    calculate_binary_metrics,
    calculate_collapsed_metrics,
    calculate_macro_averaged_metrics,
    mo,
):
    @mo.persistent_cache
    def compute_f1_score(df, truth_col, pred_col, metric_type, target_param=None):
        """
        Unified function to compute F1 scores for different metric types.

        Args:
            df: DataFrame with data
            truth_col: Column name for ground truth
            pred_col: Column name for predictions
            metric_type: Type of metric ('gesture', 'collapsed_target', 'collapsed_non_target')
            target_param: Additional parameter (gesture label for 'gesture', gesture lists for collapsed types)
        """
        if metric_type == "gesture":
            metrics = calculate_binary_metrics(
                df[truth_col], df[pred_col], target_param
            )
        elif metric_type == "collapsed_target":
            # Use macro-averaging for target gestures
            metrics = calculate_macro_averaged_metrics(
                df[truth_col], df[pred_col], target_param
            )
        elif metric_type == "collapsed_non_target":
            # Use pooling for non-target gestures
            metrics = calculate_collapsed_metrics(
                df[truth_col], df[pred_col], target_param
            )
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        return metrics["F1-Score"]
    return


@app.cell
def _(
    METRIC_PRECISION,
    calculate_binary_metrics,
    calculate_collapsed_metrics,
    calculate_macro_averaged_metrics,
    mo,
):
    @mo.persistent_cache
    def compute_simple_delta(df, truth_col, pred_col, metric_type, target_param=None):
        """
        Unified function to compute simple delta of metrics (All-sensors - Only-IMU) without bootstrapping.

        Args:
            df: DataFrame with data
            truth_col: Column name for ground truth
            pred_col: Column name for predictions
            metric_type: Type of metric to compute ('gesture', 'collapsed_target', 'collapsed_non_target', 'competition')
            target_param: Additional parameter needed for some metric types (gesture label, gesture lists, or competition_metric object)
        """
        df_all = df[df["all_sensors"]]
        df_imu = df[~df["all_sensors"]]

        n_all = len(df_all)
        n_imu = len(df_imu)
        if n_all == 0 or n_imu == 0:
            return None

        # Compute metric values based on type
        if metric_type == "gesture":
            metric_all = calculate_binary_metrics(
                df_all[truth_col], df_all[pred_col], target_param
            )["F1-Score"]
            metric_imu = calculate_binary_metrics(
                df_imu[truth_col], df_imu[pred_col], target_param
            )["F1-Score"]
            key_prefix = "f1"
        elif metric_type == "collapsed_target":
            metric_all = calculate_macro_averaged_metrics(
                df_all[truth_col], df_all[pred_col], target_param
            )["F1-Score"]
            metric_imu = calculate_macro_averaged_metrics(
                df_imu[truth_col], df_imu[pred_col], target_param
            )["F1-Score"]
            key_prefix = "f1"
        elif metric_type == "collapsed_non_target":
            metric_all = calculate_collapsed_metrics(
                df_all[truth_col], df_all[pred_col], target_param
            )["F1-Score"]
            metric_imu = calculate_collapsed_metrics(
                df_imu[truth_col], df_imu[pred_col], target_param
            )["F1-Score"]
            key_prefix = "f1"
        elif metric_type == "competition":
            metric_all = target_param.calculate_hierarchical_f1(
                df_all, truth_col, pred_col
            )
            metric_imu = target_param.calculate_hierarchical_f1(
                df_imu, truth_col, pred_col
            )
            key_prefix = "score"
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        delta_point = metric_all - metric_imu

        return {
            f"{key_prefix}_all": round(float(metric_all), METRIC_PRECISION),
            f"{key_prefix}_imu": round(float(metric_imu), METRIC_PRECISION),
            "delta": round(float(delta_point), METRIC_PRECISION),
            "n_all": int(n_all),
            "n_imu": int(n_imu),
        }
    return (compute_simple_delta,)


@app.cell
def _(
    CONFIDENCE_INTERVAL_PERCENTILES,
    DEFAULT_BOOTSTRAP_SAMPLES,
    METRIC_PRECISION,
    calculate_binary_metrics,
    calculate_collapsed_metrics,
    calculate_macro_averaged_metrics,
    mo,
):
    @mo.persistent_cache
    def bootstrap_delta_ci(
        df,
        np,
        truth_col: str,
        pred_col: str,
        metric_type: str,
        target_param,
        n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
        seed: int = 0,
    ):
        """
        Unified bootstrap 95% CI for delta metrics (All-sensors - Only-IMU).

        Args:
            df: DataFrame with data
            np: numpy module
            truth_col: Column name for ground truth
            pred_col: Column name for predictions
            metric_type: Type of metric ('gesture', 'collapsed_target', 'collapsed_non_target', 'competition')
            target_param: Additional parameter for metric computation
            n_boot: Number of bootstrap samples
            seed: Random seed

        Returns:
            Dict with point estimates and CI, or None if data insufficient
        """
        df_all = df[df["all_sensors"]]
        df_imu = df[~df["all_sensors"]]

        n_all = len(df_all)
        n_imu = len(df_imu)
        if n_all == 0 or n_imu == 0:
            return None

        # Point estimates on full data
        if metric_type == "gesture":
            metric_all = calculate_binary_metrics(
                df_all[truth_col], df_all[pred_col], target_param
            )["F1-Score"]
            metric_imu = calculate_binary_metrics(
                df_imu[truth_col], df_imu[pred_col], target_param
            )["F1-Score"]
            key_prefix = "f1"
        elif metric_type == "collapsed_target":
            metric_all = calculate_macro_averaged_metrics(
                df_all[truth_col], df_all[pred_col], target_param
            )["F1-Score"]
            metric_imu = calculate_macro_averaged_metrics(
                df_imu[truth_col], df_imu[pred_col], target_param
            )["F1-Score"]
            key_prefix = "f1"
        elif metric_type == "collapsed_non_target":
            metric_all = calculate_collapsed_metrics(
                df_all[truth_col], df_all[pred_col], target_param
            )["F1-Score"]
            metric_imu = calculate_collapsed_metrics(
                df_imu[truth_col], df_imu[pred_col], target_param
            )["F1-Score"]
            key_prefix = "f1"
        elif metric_type == "competition":
            metric_all = target_param.calculate_hierarchical_f1(
                df_all, truth_col, pred_col
            )
            metric_imu = target_param.calculate_hierarchical_f1(
                df_imu, truth_col, pred_col
            )
            key_prefix = "score"
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        delta_point = metric_all - metric_imu

        # Bootstrap
        rng = np.random.default_rng(seed)
        deltas = []
        all_idx = np.arange(n_all)
        imu_idx = np.arange(n_imu)

        for _ in range(n_boot):
            # Resample with replacement
            boot_all_idx = rng.choice(all_idx, size=n_all, replace=True)
            boot_imu_idx = rng.choice(imu_idx, size=n_imu, replace=True)
            df_boot_all = df_all.iloc[boot_all_idx]
            df_boot_imu = df_imu.iloc[boot_imu_idx]

            # Compute metrics on bootstrap samples
            boot_metric_all = 0
            boot_metric_imu = 0
            if metric_type == "gesture":
                metrics_boot_all = calculate_binary_metrics(
                    df_boot_all[truth_col], df_boot_all[pred_col], target_param
                )
                metrics_boot_imu = calculate_binary_metrics(
                    df_boot_imu[truth_col], df_boot_imu[pred_col], target_param
                )
                boot_metric_all = metrics_boot_all["F1-Score"]
                boot_metric_imu = metrics_boot_imu["F1-Score"]
            elif metric_type == "collapsed_target":
                metrics_boot_all = calculate_macro_averaged_metrics(
                    df_boot_all[truth_col], df_boot_all[pred_col], target_param
                )
                metrics_boot_imu = calculate_macro_averaged_metrics(
                    df_boot_imu[truth_col], df_boot_imu[pred_col], target_param
                )
                boot_metric_all = metrics_boot_all["F1-Score"]
                boot_metric_imu = metrics_boot_imu["F1-Score"]
            elif metric_type == "collapsed_non_target":
                metrics_boot_all = calculate_collapsed_metrics(
                    df_boot_all[truth_col], df_boot_all[pred_col], target_param
                )
                metrics_boot_imu = calculate_collapsed_metrics(
                    df_boot_imu[truth_col], df_boot_imu[pred_col], target_param
                )
                boot_metric_all = metrics_boot_all["F1-Score"]
                boot_metric_imu = metrics_boot_imu["F1-Score"]
            elif metric_type == "competition":
                boot_metric_all = target_param.calculate_hierarchical_f1(
                    df_boot_all, truth_col, pred_col
                )
                boot_metric_imu = target_param.calculate_hierarchical_f1(
                    df_boot_imu, truth_col, pred_col
                )

            deltas.append(boot_metric_all - boot_metric_imu)

        lo, hi = np.percentile(deltas, CONFIDENCE_INTERVAL_PERCENTILES).tolist()

        return {
            f"{key_prefix}_all": round(float(metric_all), METRIC_PRECISION),
            f"{key_prefix}_imu": round(float(metric_imu), METRIC_PRECISION),
            "delta": round(float(delta_point), METRIC_PRECISION),
            "ci_lo": round(float(lo), METRIC_PRECISION),
            "ci_hi": round(float(hi), METRIC_PRECISION),
            "n_all": int(n_all),
            "n_imu": int(n_imu),
        }
    return (bootstrap_delta_ci,)


@app.cell
def _(DATA_URL, mo, pd):
    # Load the data from GitHub raw URL
    df = pd.read_csv(DATA_URL)

    # Convert Usage column from object to boolean (True for public, False for private)
    df["public"] = df["Usage"] == "Public"
    df = df.drop(columns=["Usage"])

    # Create data dictionary explaining the columns
    data_dictionary = mo.md("""
    ### Dataset Column Descriptions

    - **sequence_id** (string): Anonymized unique identifier for each gesture sequence in the test set (format: ANON_XXXXXX)

    - **gesture0-gesture19** (string): Predicted gesture labels from each of the 20 top-performing competition submissions

    - **gesture_true** (string): Ground truth gesture label from the competition test set

    - **all_sensors** (boolean): Whether the sequence includes data from all available sensors (True) or only IMU sensors (False)

    - **public** (boolean): Whether the sequence belongs to the public test set (True) or private test set (False)
    """)

    mo.vstack([mo.md("## Top 20 Submission Results"), data_dictionary, df])
    return (df,)


@app.cell
def _(CompetitionMetric, METRIC_PRECISION, NUM_SUBMISSIONS, df, pd):
    # Define submission columns (gesture0 through gesture19)
    submission_cols = [f"gesture{i}" for i in range(NUM_SUBMISSIONS)]
    truth_col = "gesture_true"

    # Calculate competition metric for each submission
    competition_metric = CompetitionMetric()

    score_results = []
    for _i, sub_col in enumerate(submission_cols):
        metric_score = competition_metric.calculate_hierarchical_f1(
            df, truth_col, sub_col
        )
        metric_score = round(metric_score, METRIC_PRECISION)
        score_results.append({"submission_id": _i, "score": metric_score})

    results_df = pd.DataFrame(score_results)

    # Create a rank dictionary that maps submission_id to rank
    rank_dict = {
        _i: _rank + 1
        for _rank, _i in enumerate(
            results_df.sort_values("score", ascending=False).submission_id
        )
    }
    return (
        competition_metric,
        rank_dict,
        results_df,
        submission_cols,
        truth_col,
    )


@app.cell
def _(DEFAULT_BOOTSTRAP_SAMPLES, mo):
    # Filter controls for evaluation breakdown
    public_filter = mo.ui.switch(value=True, label="Include Public data subset")
    private_filter = mo.ui.switch(value=True, label="Include Private data subset")
    all_sensors_filter = mo.ui.switch(value=True, label="Include all-sensors subset")
    imu_sensors_filter = mo.ui.switch(
        value=True, label="Include only-IMU-sensors subset"
    )
    collapse_non_target_filter = mo.ui.switch(
        value=True, label="Collapse all non-target gestures into one row (pooling)"
    )
    collapse_target_filter = mo.ui.switch(
        value=False, label="Collapse all target gestures into one row (macro-averaging)"
    )
    collapse_submissions_filter = mo.ui.switch(
        value=False, label="Collapse all submissions for overall evaluation (pooling)"
    )
    bootstrap_toggle = mo.ui.switch(
        value=False,
        label="Enable bootstrap confidence intervals for delta plots (slower)",
    )
    bootstrap_samples = mo.ui.text(
        value=str(DEFAULT_BOOTSTRAP_SAMPLES),
        label="Number of bootstrap samples",
        placeholder=str(DEFAULT_BOOTSTRAP_SAMPLES),
    )

    filters_display = mo.vstack(
        [
            mo.md("### Evaluation Filters"),
            mo.md(
                "Select which data subsets and sensor configurations to include in the evaluation:"
            ),
            mo.md("**Visibility Subsets:**"),
            mo.hstack([public_filter, private_filter], justify="start", widths=[1, 2]),
            mo.md("**Sensor Subsets:**"),
            mo.hstack(
                [all_sensors_filter, imu_sensors_filter],
                justify="start",
                widths=[1, 2],
            ),
            mo.md("**Collapsing Options:**"),
            mo.hstack(
                [collapse_non_target_filter, collapse_target_filter],
                justify="start",
                widths=[1, 2],
            ),
            mo.hstack([collapse_submissions_filter], justify="start", widths=[1]),
            mo.md(
                "<u>Macro-averaging</u>: Equal weighting regardless of sample count  \n"
                "<u>Pooling</u>: Sample-weighted averaging"
            ),
            mo.md("**Bootstrapped Estimation:**"),
            mo.hstack(
                [bootstrap_toggle, bootstrap_samples], justify="start", widths=[1, 2]
            ),
        ]
    )

    filters_display
    return (
        all_sensors_filter,
        bootstrap_samples,
        bootstrap_toggle,
        collapse_non_target_filter,
        collapse_submissions_filter,
        collapse_target_filter,
        imu_sensors_filter,
        private_filter,
        public_filter,
    )


@app.cell
def _(
    CHART_COLOR_SCHEME,
    CHART_HEIGHT,
    CHART_WIDTH,
    alt,
    collapse_submissions_filter,
    mo,
    pd,
    results_df,
):
    # Create an interactive bar chart with selection capabilities
    if collapse_submissions_filter.value:
        # When collapsed, show overall statistics as boxplot and table
        # Create boxplot data from individual submission scores
        boxplot_data = pd.DataFrame(
            {
                "submission_id": results_df["submission_id"],
                "score": results_df["score"],
                "category": "All Submissions",  # Single category for the boxplot
            }
        )

        # Create boxplot
        score_chart = mo.ui.altair_chart(
            alt.Chart(boxplot_data)
            .mark_boxplot(size=40, color="green", outliers={"size": 8})
            .encode(
                x=alt.X("category:N", title="", axis=alt.Axis(labelAngle=0)),
                y=alt.Y(
                    "score:Q",
                    title="Competition Metric Score",
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip("category:N", title="Category"),
                    alt.Tooltip("score:Q", format=".4f", title="Score"),
                ],
            )
            .properties(
                width=CHART_WIDTH // 2,  # Make it narrower since it's just one box
                height=CHART_HEIGHT,
                title="Distribution of Submission Scores",
            )
        )

        # Create summary statistics table
        _overall_stats = {
            "Statistic": [
                "Mean Score",
                "Median Score",
                "Min Score",
                "Max Score",
                "Std Dev",
                "Total Submissions",
            ],
            "Value": [
                f"{results_df['score'].mean():.4f}",
                f"{results_df['score'].median():.4f}",
                f"{results_df['score'].min():.4f}",
                f"{results_df['score'].max():.4f}",
                f"{results_df['score'].std():.4f}",
                str(len(results_df)),
            ],
        }
        _overall_df = pd.DataFrame(_overall_stats)

        chart_title = "## Overall Submission Statistics"
        chart_description = "Boxplot showing the distribution of scores across all submissions, with summary statistics below."

        # Display both chart and table side by side
        display_content = mo.vstack(
            [
                mo.md(chart_title),
                mo.md(chart_description),
                mo.hstack(
                    [
                        score_chart,
                        mo.ui.table(_overall_df, page_size=10, selection=None),
                    ],
                    justify="start",
                ),
            ]
        )
    else:
        # Original behavior: individual submission scores
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
                    alt.Color("score:Q", scale=alt.Scale(scheme=CHART_COLOR_SCHEME)),
                    alt.value("lightgrey"),
                ),
                stroke=alt.condition(
                    click_selector, alt.value("black"), alt.value(None)
                ),
                strokeWidth=alt.condition(click_selector, alt.value(1), alt.value(0)),
                tooltip=["submission_id:N", "score:Q"],
            )
            .properties(
                width=CHART_WIDTH,
                height=CHART_HEIGHT,
                title="Competition Metric Scores by Submission",
            )
        )

        chart_title = "## Competition Metric Scores by Submission"
        chart_description = "Click on a bar to see submission details below:"

        display_content = mo.vstack(
            [
                mo.md(chart_title),
                mo.md(chart_description),
                score_chart,
            ]
        )

    display_content
    return (score_chart,)


@app.cell
def _(
    CHART_HEIGHT,
    CHART_WIDTH,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DELTA_COLOR_SCHEME,
    NUM_SUBMISSIONS,
    alt,
    bootstrap_delta_ci,
    bootstrap_samples,
    bootstrap_toggle,
    collapse_submissions_filter,
    competition_metric,
    compute_simple_delta,
    df,
    mo,
    np,
    pd,
    private_filter,
    public_filter,
    score_chart,
    stack_submissions,
    submission_cols,
    truth_col,
):
    # Per-submission overview: shows delta comparison across all submissions
    # This ignores submission selection and sensor toggles, always comparing both sensor types

    # Filter by public/private only; always include both sensor types for delta
    overview_masks = []
    if public_filter.value:
        overview_masks.append(df["public"])
    if private_filter.value:
        overview_masks.append(~df["public"])
    if overview_masks:
        overview_mask = overview_masks[0]
        for _overview_m in overview_masks[1:]:
            overview_mask = overview_mask | _overview_m
        overview_filtered_df = df[overview_mask]
    else:
        overview_filtered_df = df

    # Ensure both sensor subsets exist
    if (overview_filtered_df["all_sensors"].sum() == 0) or (
        (~overview_filtered_df["all_sensors"]).sum() == 0
    ):
        overview_display = mo.md(
            "Per-submission overview requires both all-sensors and only-IMU samples in the selected splits. Adjust Public/Private filters above."
        )
    else:
        if collapse_submissions_filter.value:
            # Handle collapsed submissions mode - show overall delta only
            # Get number of bootstrap samples, default to DEFAULT_BOOTSTRAP_SAMPLES if invalid
            try:
                _overview_n_bootstrap = (
                    int(bootstrap_samples.value)
                    if bootstrap_samples.value
                    else DEFAULT_BOOTSTRAP_SAMPLES
                )
            except (ValueError, TypeError):
                _overview_n_bootstrap = DEFAULT_BOOTSTRAP_SAMPLES

            # Stack all submissions for aggregated analysis
            _overview_stacked_df = stack_submissions(
                overview_filtered_df, truth_col, submission_cols
            )

            # Calculate overall competition metric delta across all submissions
            if bootstrap_toggle.value:
                overall_res = bootstrap_delta_ci(
                    _overview_stacked_df,
                    np,
                    "truth",  # Use the renamed truth column from stacked data
                    "prediction",  # Use the stacked prediction column
                    "competition",
                    competition_metric,
                    n_boot=_overview_n_bootstrap,
                    seed=42,
                )
            else:
                overall_res = compute_simple_delta(
                    _overview_stacked_df,
                    "truth",  # Use the renamed truth column from stacked data
                    "prediction",
                    "competition",
                    competition_metric,
                )

            if overall_res is None:
                overview_display = mo.md("No valid data for overall delta computation.")
            else:
                # Create summary statistics table
                _overview_stats = {
                    "Metric": [
                        "Competition Metric (All-sensors)",
                        "Competition Metric (Only-IMU)",
                        "Delta (All - IMU)",
                        "Sample Count (All-sensors)",
                        "Sample Count (Only-IMU)",
                    ],
                    "Value": [
                        f"{overall_res['score_all']:.4f}",
                        f"{overall_res['score_imu']:.4f}",
                        f"{overall_res['delta']:.4f}",
                        str(overall_res["n_all"]),
                        str(overall_res["n_imu"]),
                    ],
                }

                if bootstrap_toggle.value and "ci_lo" in overall_res:
                    _overview_stats["Metric"].extend(
                        [
                            "95% CI Lower Bound",
                            "95% CI Upper Bound",
                        ]
                    )
                    _overview_stats["Value"].extend(
                        [
                            f"{overall_res['ci_lo']:.4f}",
                            f"{overall_res['ci_hi']:.4f}",
                        ]
                    )

                _overview_df = pd.DataFrame(_overview_stats)

                # Create a simple bar chart showing only the delta
                delta_chart_data = pd.DataFrame(
                    {
                        "metric": ["Delta"],
                        "value": [overall_res["delta"]],
                        "type": ["Delta"],
                        "all_sensors_score": [overall_res["score_all"]],
                        "imu_score": [overall_res["score_imu"]],
                    }
                )

                # Add CI columns if bootstrap is enabled and CI data exists
                if (
                    bootstrap_toggle.value
                    and "ci_lo" in overall_res
                    and "ci_hi" in overall_res
                ):
                    delta_chart_data["ci_lo"] = overall_res["ci_lo"]
                    delta_chart_data["ci_hi"] = overall_res["ci_hi"]

                overview_chart = (
                    alt.Chart(delta_chart_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("metric:N", title="", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y(
                            "value:Q", title="Competition Metric Delta (All − IMU)"
                        ),
                        color=alt.Color(
                            "value:Q",
                            scale=alt.Scale(scheme=DELTA_COLOR_SCHEME),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("metric:N", title="Metric"),
                            alt.Tooltip("value:Q", format=".4f", title="Delta Value"),
                            alt.Tooltip(
                                "all_sensors_score:Q",
                                title="All-sensors Score",
                                format=".4f",
                            ),
                            alt.Tooltip(
                                "imu_score:Q", title="Only-IMU Score", format=".4f"
                            ),
                        ]
                        + (
                            [
                                alt.Tooltip("ci_lo:Q", title="CI low", format=".4f"),
                                alt.Tooltip("ci_hi:Q", title="CI high", format=".4f"),
                            ]
                            if bootstrap_toggle.value
                            and "ci_lo" in overall_res
                            and "ci_hi" in overall_res
                            else []
                        ),
                    )
                    .properties(
                        width=CHART_WIDTH // 3,
                        height=CHART_HEIGHT,
                        title="Overall Competition Metric Delta (All Submissions)",
                    )
                )

                # Add error bars if bootstrap is enabled
                if (
                    bootstrap_toggle.value
                    and "ci_lo" in overall_res
                    and "ci_hi" in overall_res
                ):
                    error_bars = (
                        alt.Chart(delta_chart_data)
                        .mark_rule(color="#333", strokeWidth=2)
                        .encode(
                            x="metric:N",
                            y="ci_lo:Q",
                            y2="ci_hi:Q",
                            tooltip=[
                                alt.Tooltip("metric:N", title="Metric"),
                                alt.Tooltip("ci_lo:Q", title="CI low", format=".4f"),
                                alt.Tooltip("ci_hi:Q", title="CI high", format=".4f"),
                            ],
                        )
                    )
                    overview_chart = overview_chart + error_bars

                # Create filter summary for chart title
                overview_filter_parts = []
                if public_filter.value and private_filter.value:
                    overview_filter_parts.append("Public + Private")
                elif public_filter.value:
                    overview_filter_parts.append("Public only")
                elif private_filter.value:
                    overview_filter_parts.append("Private only")
                else:
                    overview_filter_parts.append(
                        "No data selected, including the whole dataset"
                    )

                overview_filter_summary = (
                    overview_filter_parts[0] if overview_filter_parts else "No data"
                )

                _bootstrap_status = (
                    " (with bootstrap CIs)"
                    if bootstrap_toggle.value
                    else " (point estimates only)"
                )

                overview_display = mo.vstack(
                    [
                        mo.md("### Overall Competition Metric Delta (All Submissions)"),
                        mo.md(
                            f"This shows the overall performance difference when all submissions are combined{_bootstrap_status}. "
                            f"Positive values indicate all-sensors performs better. This analysis respects Public/Private filters but ignores sensor toggles."
                        ),
                        mo.hstack(
                            [
                                overview_chart,
                                mo.ui.table(_overview_df, page_size=10, selection=None),
                            ],
                            justify="start",
                        ),
                    ]
                )
        else:
            # Original behavior: per-submission deltas
            # Per-submission overall deltas across all NUM_SUBMISSIONS submissions
            overview_sub_rows = []
            for _sid in range(NUM_SUBMISSIONS):
                # Get number of bootstrap samples, default to DEFAULT_BOOTSTRAP_SAMPLES if invalid
                try:
                    _overview_n_bootstrap = (
                        int(bootstrap_samples.value)
                        if bootstrap_samples.value
                        else DEFAULT_BOOTSTRAP_SAMPLES
                    )
                except (ValueError, TypeError):
                    _overview_n_bootstrap = DEFAULT_BOOTSTRAP_SAMPLES

                # Calculate delta for all data
                if bootstrap_toggle.value:
                    all_data_res = bootstrap_delta_ci(
                        overview_filtered_df,
                        np,
                        truth_col,
                        f"gesture{_sid}",
                        "competition",
                        competition_metric,
                        n_boot=_overview_n_bootstrap,
                        seed=100 + _sid,
                    )
                else:
                    all_data_res = compute_simple_delta(
                        overview_filtered_df,
                        truth_col,
                        f"gesture{_sid}",
                        "competition",
                        competition_metric,
                    )
                if all_data_res is not None:
                    overview_sub_rows.append(
                        {
                            "submission_id": _sid,
                            **all_data_res,
                        }
                    )

            if overview_sub_rows:
                overview_subs_df = (
                    pd.DataFrame(overview_sub_rows)
                    .sort_values("submission_id", ascending=True)
                    .reset_index(drop=True)
                )

                # Get selected submission ID from score_chart if available
                _selected_submission_id = None
                if score_chart.value is not None and not score_chart.value.empty:
                    _selected_submission_id = int(
                        score_chart.value.iloc[0]["submission_id"]
                    )

                # Create simple bar chart
                if _selected_submission_id is not None:
                    # When a submission is selected, highlight it and gray out others
                    overview_subs_chart = (
                        alt.Chart(overview_subs_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("submission_id:N", title="Submission ID"),
                            y=alt.Y(
                                "delta:Q", title="Competition metric delta (All − IMU)"
                            ),
                            color=alt.condition(
                                f"datum.submission_id == {_selected_submission_id}",
                                alt.Color(
                                    "delta:Q",
                                    scale=alt.Scale(scheme=DELTA_COLOR_SCHEME),
                                    legend=None,
                                ),
                                alt.value("lightgrey"),
                            ),
                            stroke=alt.condition(
                                f"datum.submission_id == {_selected_submission_id}",
                                alt.value("black"),
                                alt.value(None),
                            ),
                            strokeWidth=alt.condition(
                                f"datum.submission_id == {_selected_submission_id}",
                                alt.value(1),
                                alt.value(0),
                            ),
                            tooltip=[
                                alt.Tooltip("submission_id:N", title="Submission"),
                                alt.Tooltip("delta:Q", format=".3f"),
                                alt.Tooltip(
                                    "score_all:Q",
                                    title="All-sensors score",
                                    format=".3f",
                                ),
                                alt.Tooltip(
                                    "score_imu:Q", title="Only-IMU score", format=".3f"
                                ),
                            ]
                            + (
                                [
                                    alt.Tooltip(
                                        "ci_lo:Q", title="CI low", format=".3f"
                                    ),
                                    alt.Tooltip(
                                        "ci_hi:Q", title="CI high", format=".3f"
                                    ),
                                ]
                                if bootstrap_toggle.value
                                else []
                            ),
                        )
                    )
                else:
                    # When no submission is selected, show all bars in normal colors
                    overview_subs_chart = (
                        alt.Chart(overview_subs_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("submission_id:N", title="Submission ID"),
                            y=alt.Y(
                                "delta:Q", title="Competition metric delta (All − IMU)"
                            ),
                            color=alt.Color(
                                "delta:Q",
                                scale=alt.Scale(scheme=DELTA_COLOR_SCHEME),
                            ),
                            tooltip=[
                                alt.Tooltip("submission_id:N", title="Submission"),
                                alt.Tooltip("delta:Q", format=".3f"),
                                alt.Tooltip(
                                    "score_all:Q",
                                    title="All-sensors score",
                                    format=".3f",
                                ),
                                alt.Tooltip(
                                    "score_imu:Q", title="Only-IMU score", format=".3f"
                                ),
                            ]
                            + (
                                [
                                    alt.Tooltip(
                                        "ci_lo:Q", title="CI low", format=".3f"
                                    ),
                                    alt.Tooltip(
                                        "ci_hi:Q", title="CI high", format=".3f"
                                    ),
                                ]
                                if bootstrap_toggle.value
                                else []
                            ),
                        )
                    )

                if bootstrap_toggle.value:
                    if _selected_submission_id is not None:
                        # When a submission is selected, highlight error bars for selected and gray out others
                        overview_subs_err = (
                            alt.Chart(overview_subs_df)
                            .mark_rule()
                            .encode(
                                x=alt.X("submission_id:N"),
                                y="ci_lo:Q",
                                y2="ci_hi:Q",
                                color=alt.condition(
                                    f"datum.submission_id == {_selected_submission_id}",
                                    alt.value("#333"),
                                    alt.value("lightgrey"),
                                ),
                            )
                        )
                    else:
                        # When no submission is selected, show all error bars in normal color
                        overview_subs_err = (
                            alt.Chart(overview_subs_df)
                            .mark_rule(color="#333")
                            .encode(
                                x="submission_id:N",
                                y="ci_lo:Q",
                                y2="ci_hi:Q",
                            )
                        )
                    overview_chart_final = overview_subs_chart + overview_subs_err
                else:
                    overview_chart_final = overview_subs_chart

                overview_subs_zero = (
                    alt.Chart(pd.DataFrame({"y": [0]}))
                    .mark_rule(color="#888", strokeDash=[4, 4])
                    .encode(y="y:Q")
                )

                # Create filter summary for chart title
                overview_filter_parts = []
                if public_filter.value and private_filter.value:
                    overview_filter_parts.append("Public + Private")
                elif public_filter.value:
                    overview_filter_parts.append("Public only")
                elif private_filter.value:
                    overview_filter_parts.append("Private only")
                else:
                    overview_filter_parts.append(
                        "No data selected, including the whole dataset"
                    )

                overview_filter_summary = (
                    overview_filter_parts[0] if overview_filter_parts else "No data"
                )

                overview_subs_combo = (
                    overview_chart_final + overview_subs_zero
                ).properties(
                    title=f"Competition Metric Delta: All-sensors minus Only-IMU ({overview_filter_summary}){' with 95% CIs' if bootstrap_toggle.value else ''}",
                    width=CHART_WIDTH,
                    height=CHART_HEIGHT,
                )

                _bootstrap_status = (
                    " (with bootstrap CIs)"
                    if bootstrap_toggle.value
                    else " (point estimates only)"
                )
                overview_display = mo.vstack(
                    [
                        mo.md("### Per-Submission Competition Metric Delta Overview"),
                        mo.md(
                            f"This chart shows how much better (or worse) each submission performs with all-sensors versus only-IMU sensors{_bootstrap_status}. "
                            "Positive values indicate all-sensors performs better. When you select a submission from the first chart above, the corresponding bar is highlighted below. "
                            "This analysis respects Public/Private filters but ignores sensor toggles and submission selection."
                        ),
                        overview_subs_combo,
                    ]
                )
            else:
                overview_display = mo.md("No valid data for per-submission overview.")

    overview_display
    return


@app.cell
def _(
    NUM_SUBMISSIONS,
    collapse_submissions_filter,
    mo,
    rank_dict,
    results_df,
    score_chart,
):
    # Show details for selected submission from chart or overall stats when collapsed
    if collapse_submissions_filter.value:
        # When collapsed, statistics are already shown in the first chart
        # submission_details = mo.md(
        #     "*Overall submission statistics are displayed in the chart above*"
        # )
        submission_details = None
    else:
        # Original behavior: show selected submission details
        if score_chart.value is not None and not score_chart.value.empty:
            _selected_submission_id = int(score_chart.value.iloc[0]["submission_id"])
            submission_score = results_df[
                results_df["submission_id"] == _selected_submission_id
            ]["score"].iloc[0]

            submission_details = mo.md(f"""
        ### Selected Submission Details

        **Submission ID:** {_selected_submission_id}  
        **Competition Metric Score:** {submission_score}  
        **Rank:** {rank_dict[_selected_submission_id]} out of {NUM_SUBMISSIONS} submissions
        """)
        else:
            submission_details = mo.md(
                "*Select a submission from the first chart to see details*"
            )

    submission_details
    return


@app.cell
def _(
    all_sensors_filter,
    apply_data_filters,
    calculate_binary_metrics,
    calculate_collapsed_metrics,
    calculate_macro_averaged_metrics,
    collapse_non_target_filter,
    collapse_submissions_filter,
    collapse_target_filter,
    competition_metric,
    create_filter_summary,
    df,
    imu_sensors_filter,
    mo,
    pd,
    private_filter,
    public_filter,
    score_chart,
    stack_submissions,
    submission_cols,
    truth_col,
):
    # Show accuracy breakdown for selected submission from chart or overall when collapsed
    if collapse_submissions_filter.value:
        # When collapsed, compute metrics across all submissions
        # Apply filters to the dataframe
        _filtered_df = apply_data_filters(
            df, public_filter, private_filter, all_sensors_filter, imu_sensors_filter
        )

        # Stack all submissions for collapsed analysis
        _stacked_df = stack_submissions(_filtered_df, truth_col, submission_cols)

        # Create accuracy breakdown data from stacked data
        _y_true = _stacked_df["truth"]
        _y_pred = _stacked_df["prediction"]

        _eval_data = []

        # Handle target gestures based on collapse setting
        if collapse_target_filter.value:
            # Calculate macro-averaged metrics for all target gestures
            _collapsed_target_metrics = calculate_macro_averaged_metrics(
                _y_true, _y_pred, competition_metric.target_gestures
            )
            _eval_data.append(
                {
                    "gesture": "All target gestures",
                    "gesture_type": "Target",
                    **_collapsed_target_metrics,
                }
            )
        else:
            # Calculate metrics for target gestures individually
            for _label in competition_metric.target_gestures:
                _metrics = calculate_binary_metrics(_y_true, _y_pred, _label)
                _eval_data.append(
                    {"gesture": _label, "gesture_type": "Target", **_metrics}
                )

        # Handle non-target gestures based on collapse setting
        if collapse_non_target_filter.value:
            # Calculate collapsed metrics for all non-target gestures (pooling)
            _collapsed_metrics = calculate_collapsed_metrics(
                _y_true, _y_pred, competition_metric.non_target_gestures
            )
            _eval_data.append(
                {
                    "gesture": "All non-target gestures",
                    "gesture_type": "Non-Target",
                    **_collapsed_metrics,
                }
            )
        else:
            # Calculate metrics for non-target gestures individually
            for _label in competition_metric.non_target_gestures:
                _metrics = calculate_binary_metrics(_y_true, _y_pred, _label)
                _eval_data.append(
                    {"gesture": _label, "gesture_type": "Non-Target", **_metrics}
                )

        eval_df = pd.DataFrame(_eval_data)

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
            collapse_target_filter,
            collapse_submissions_filter,
        )
        _sample_count = len(_stacked_df)

        eval_breakdown = mo.vstack(
            [
                mo.md("### Overall Per-Gesture Metrics (All Submissions)"),
                mo.md(
                    f"**Filters applied:** {_filter_summary} | **Total samples:** {_sample_count}"
                ),
                mo.ui.table(eval_df, page_size=20, selection=None),
            ]
        )
    elif score_chart.value is not None and not score_chart.value.empty:
        _selected_submission_id = int(score_chart.value.iloc[0]["submission_id"])
        _submission_col = f"gesture{_selected_submission_id}"

        # Apply filters to the dataframe
        _filtered_df = apply_data_filters(
            df, public_filter, private_filter, all_sensors_filter, imu_sensors_filter
        )

        # Create accuracy breakdown data from filtered data
        _y_true = _filtered_df[truth_col]
        _y_pred = _filtered_df[_submission_col]

        _eval_data = []

        # Handle target gestures based on collapse setting
        if collapse_target_filter.value:
            # Calculate collapsed metrics for all target gestures
            _collapsed_target_metrics = calculate_macro_averaged_metrics(
                _y_true, _y_pred, competition_metric.target_gestures
            )
            _eval_data.append(
                {
                    "gesture": "All target gestures",
                    "gesture_type": "Target",
                    **_collapsed_target_metrics,
                }
            )
        else:
            # Calculate metrics for target gestures individually
            for _label in competition_metric.target_gestures:
                _metrics = calculate_binary_metrics(_y_true, _y_pred, _label)
                _eval_data.append(
                    {"gesture": _label, "gesture_type": "Target", **_metrics}
                )

        # Handle non-target gestures based on collapse setting
        if collapse_non_target_filter.value:
            # Calculate collapsed metrics for all non-target gestures (pooling)
            _collapsed_metrics = calculate_collapsed_metrics(
                _y_true, _y_pred, competition_metric.non_target_gestures
            )
            _eval_data.append(
                {
                    "gesture": "All non-target gestures",
                    "gesture_type": "Non-Target",
                    **_collapsed_metrics,
                }
            )
        else:
            # Calculate metrics for non-target gestures individually
            for _label in competition_metric.non_target_gestures:
                _metrics = calculate_binary_metrics(_y_true, _y_pred, _label)
                _eval_data.append(
                    {"gesture": _label, "gesture_type": "Non-Target", **_metrics}
                )

        eval_df = pd.DataFrame(_eval_data)

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
            collapse_target_filter,
            collapse_submissions_filter,
        )
        _sample_count = len(_filtered_df)

        eval_breakdown = mo.vstack(
            [
                mo.md(
                    f"### Per-Gesture Metrics for Submission {_selected_submission_id}"
                ),
                mo.md(
                    f"**Filters applied:** {_filter_summary} | **Samples:** {_sample_count}"
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
    GESTURE_TYPE_COLORS,
    all_sensors_filter,
    alt,
    collapse_non_target_filter,
    collapse_submissions_filter,
    collapse_target_filter,
    create_filter_summary,
    eval_df,
    imu_sensors_filter,
    mo,
    private_filter,
    public_filter,
    score_chart,
):
    if collapse_submissions_filter.value:
        # When collapsed, show overall chart
        # Create filter summary for chart title
        _filter_summary = create_filter_summary(
            public_filter,
            private_filter,
            all_sensors_filter,
            imu_sensors_filter,
            collapse_non_target_filter,
            collapse_target_filter,
            collapse_submissions_filter,
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
                    scale=alt.Scale(range=GESTURE_TYPE_COLORS),
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
                    alt.Tooltip(
                        "NPV:Q", format=".3f", title="Negative Predictive Value"
                    ),
                ],
            )
            .properties(
                title=f"Overall Per-Gesture F1-Scores (All Submissions) ({_filter_summary})"
            )
            .interactive()
        )
    elif score_chart.value is not None and not score_chart.value.empty:
        _selected_submission_id = int(score_chart.value.iloc[0]["submission_id"])

        # Create filter summary for chart title
        _filter_summary = create_filter_summary(
            public_filter,
            private_filter,
            all_sensors_filter,
            imu_sensors_filter,
            collapse_non_target_filter,
            collapse_target_filter,
            collapse_submissions_filter,
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
                    scale=alt.Scale(range=GESTURE_TYPE_COLORS),
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
                    alt.Tooltip(
                        "NPV:Q", format=".3f", title="Negative Predictive Value"
                    ),
                ],
            )
            .properties(
                title=f"Per-Gesture F1-Scores for Submission {_selected_submission_id} ({_filter_summary})"
            )
            .interactive()
        )
    else:
        f1_score_chart = mo.md(
            "*Select a submission to see per-gesture F1-scores*"
        )

    f1_score_chart
    return


@app.cell
def _(
    BOXPLOT_HEIGHT,
    CHART_WIDTH,
    DEFAULT_BOOTSTRAP_SAMPLES,
    GESTURE_TYPE_COLORS,
    alt,
    bootstrap_delta_ci,
    bootstrap_samples,
    bootstrap_toggle,
    build_filter_masks,
    collapse_non_target_filter,
    collapse_submissions_filter,
    collapse_target_filter,
    competition_metric,
    compute_simple_delta,
    df,
    mo,
    np,
    pd,
    private_filter,
    public_filter,
    score_chart,
    stack_submissions,
    submission_cols,
    truth_col,
):
    # Helper functions for delta analysis
    def _calculate_delta(
        data,
        truth_col,
        pred_col,
        metric_type,
        target_param,
        use_bootstrap,
        n_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES,
    ):
        """Calculate delta with or without bootstrap CI."""
        if use_bootstrap:
            return bootstrap_delta_ci(
                data,
                np,
                truth_col,
                pred_col,
                metric_type,
                target_param,
                n_boot=n_bootstrap,
                seed=42,
            )
        else:
            return compute_simple_delta(
                data, truth_col, pred_col, metric_type, target_param
            )

    def _process_gesture_group(
        data,
        truth_col,
        pred_col,
        gestures,
        group_name,
        group_type,
        collapse_enabled,
        use_bootstrap,
        n_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES,
    ):
        """Process a group of gestures (target or non-target) with optional collapsing."""
        rows = []

        if collapse_enabled:
            # Calculate collapsed delta for all gestures in group
            metric_type = f"collapsed_{group_name.lower().replace('-', '_')}"
            result = _calculate_delta(
                data,
                truth_col,
                pred_col,
                metric_type,
                gestures,
                use_bootstrap,
                n_bootstrap,
            )
            if result is not None:
                rows.append(
                    {
                        "gesture": f"All {group_name.lower()} gestures",
                        "gesture_type": group_type,
                        **result,
                    }
                )
        else:
            # Calculate deltas for gestures individually
            for gesture_label in gestures:
                result = _calculate_delta(
                    data,
                    truth_col,
                    pred_col,
                    "gesture",
                    gesture_label,
                    use_bootstrap,
                    n_bootstrap,
                )
                if result is not None:
                    rows.append(
                        {
                            "gesture": gesture_label,
                            "gesture_type": group_type,
                            **result,
                        }
                    )

        return rows

    def _create_delta_chart(delta_df, use_bootstrap):
        """Create the delta visualization chart."""
        base = alt.Chart(delta_df)

        # Determine the metric type based on column names
        has_score_columns = "score_all" in delta_df.columns

        if has_score_columns:
            metric_label = "Competition Score"
            all_col = "score_all"
            imu_col = "score_imu"
        else:
            metric_label = "F1 Score"
            all_col = "f1_all"
            imu_col = "f1_imu"

        # Base tooltip
        tooltip_list = [
            alt.Tooltip("gesture:N"),
            alt.Tooltip("gesture_type:N", title="Type"),
            alt.Tooltip("delta:Q", format=".3f"),
            alt.Tooltip(
                f"{all_col}:Q", title=f"All-sensors {metric_label}", format=".3f"
            ),
            alt.Tooltip(f"{imu_col}:Q", title=f"IMU-only {metric_label}", format=".3f"),
        ]

        if use_bootstrap:
            tooltip_list.extend(
                [
                    alt.Tooltip("ci_lo:Q", title="CI low", format=".3f"),
                    alt.Tooltip("ci_hi:Q", title="CI high", format=".3f"),
                ]
            )

        # Points
        points = base.mark_point(filled=True, size=60).encode(
            y=alt.Y("gesture:N", sort=list(delta_df["gesture"])),
            x=alt.X("delta:Q", title=f"{metric_label} Delta (All-sensors − IMU-only)"),
            color=alt.Color(
                "gesture_type:N",
                sort=["Target", "Non-Target"],
                scale=alt.Scale(range=GESTURE_TYPE_COLORS),
            ),
            tooltip=tooltip_list,
        )

        # Zero reference line
        zero_rule = (
            alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_rule(color="#888", strokeDash=[4, 4])
            .encode(x="x:Q")
        )

        # Add error bars if bootstrap is enabled
        if use_bootstrap:
            error_bars = base.mark_rule().encode(
                y=alt.Y("gesture:N", sort=list(delta_df["gesture"])),
                x=alt.X(
                    "ci_lo:Q", title=f"{metric_label} Delta (All-sensors − IMU-only)"
                ),
                x2="ci_hi:Q",
                color=alt.Color(
                    "gesture_type:N",
                    sort=["Target", "Non-Target"],
                    scale=alt.Scale(range=GESTURE_TYPE_COLORS),
                ),
                tooltip=tooltip_list,
            )
            return error_bars + points + zero_rule
        else:
            return points + zero_rule

    def _create_filter_summary():
        """Create filter summary string for chart title."""
        parts = []
        if public_filter.value and private_filter.value:
            parts.append("Public + Private")
        elif public_filter.value:
            parts.append("Public only")
        elif private_filter.value:
            parts.append("Private only")
        else:
            parts.append("No data selected, including the whole dataset")

        if collapse_non_target_filter.value:
            parts.append("Non-target gestures collapsed")
        if collapse_target_filter.value:
            parts.append("Target gestures collapsed")
        if collapse_submissions_filter.value:
            parts.append("All submissions collapsed")

        return " | ".join(parts)

    # Main delta analysis logic
    if collapse_submissions_filter.value:
        # Handle collapsed submissions mode
        # Filter by public/private only; always include both sensor types for delta
        delta_filtered_df = build_filter_masks(df, public_filter, private_filter)

        # Check if both sensor subsets exist
        has_all_sensors = delta_filtered_df["all_sensors"].sum() > 0
        has_imu_only = (~delta_filtered_df["all_sensors"]).sum() > 0

        if not (has_all_sensors and has_imu_only):
            view = mo.md(
                "Delta view requires both all-sensors and only-IMU samples in the selected splits. "
                "Adjust filters above."
            )
            delta_df = pd.DataFrame()
        else:
            # Get number of bootstrap samples, default to DEFAULT_BOOTSTRAP_SAMPLES if invalid
            try:
                _delta_n_bootstrap = (
                    int(bootstrap_samples.value)
                    if bootstrap_samples.value
                    else DEFAULT_BOOTSTRAP_SAMPLES
                )
            except (ValueError, TypeError):
                _delta_n_bootstrap = DEFAULT_BOOTSTRAP_SAMPLES

            # Stack all submissions for aggregated analysis
            _delta_stacked_df = stack_submissions(
                delta_filtered_df, truth_col, submission_cols
            )

            # Calculate overall competition metric delta
            comp_boot = _calculate_delta(
                _delta_stacked_df,
                "truth",  # Use the renamed truth column from stacked data
                "prediction",  # Use the stacked prediction column
                "competition",
                competition_metric,
                bootstrap_toggle.value,
                _delta_n_bootstrap,
            )

            # Calculate per-gesture deltas
            rows = []

            # Process target gestures
            rows.extend(
                _process_gesture_group(
                    _delta_stacked_df,
                    "truth",  # Use the renamed truth column from stacked data
                    "prediction",
                    competition_metric.target_gestures,
                    "target",
                    "Target",
                    collapse_target_filter.value,
                    bootstrap_toggle.value,
                    _delta_n_bootstrap,
                )
            )

            # Process non-target gestures
            rows.extend(
                _process_gesture_group(
                    _delta_stacked_df,
                    "truth",  # Use the renamed truth column from stacked data
                    "prediction",
                    competition_metric.non_target_gestures,
                    "non-target",
                    "Non-Target",
                    collapse_non_target_filter.value,
                    bootstrap_toggle.value,
                    _delta_n_bootstrap,
                )
            )

            if not rows:
                view = mo.md("No valid data for delta computation.")
                delta_df = pd.DataFrame()
            else:
                delta_df = (
                    pd.DataFrame(rows)
                    .sort_values(by="delta", ascending=False)
                    .reset_index(drop=True)
                )

                # Create chart
                chart = _create_delta_chart(delta_df, bootstrap_toggle.value)
                filter_summary = _create_filter_summary()

                chart = chart.properties(
                    title=f"Overall Per-Gesture F1 Score Delta (All Submissions) ({filter_summary}){' with 95% CIs' if bootstrap_toggle.value else ''}",
                    width=CHART_WIDTH,
                    height=max(BOXPLOT_HEIGHT, 20 * len(delta_df)),
                ).interactive()

                # Create header with competition metric info
                header_bits = []
                if comp_boot is not None:
                    if bootstrap_toggle.value and "ci_lo" in comp_boot:
                        header_bits.append(
                            f"Overall competition metric delta: {comp_boot['delta']:.3f} "
                            f"(95% CI [{comp_boot['ci_lo']:.3f}, {comp_boot['ci_hi']:.3f}])"
                        )
                    else:
                        header_bits.append(
                            f"Overall competition metric delta: {comp_boot['delta']:.3f}"
                        )
                    header_bits.append(
                        f"All-sensors: {comp_boot['score_all']:.3f} vs Only-IMU: {comp_boot['score_imu']:.3f}"
                    )
                header = " | ".join(header_bits) if header_bits else ""

                # Determine table columns
                table_columns = [
                    "gesture",
                    "gesture_type",
                    "f1_all",
                    "f1_imu",
                    "delta",
                    "n_all",
                    "n_imu",
                ]
                if bootstrap_toggle.value:
                    table_columns.insert(-2, "ci_lo")
                    table_columns.insert(-2, "ci_hi")

                bootstrap_status = (
                    " (with bootstrap CIs)"
                    if bootstrap_toggle.value
                    else " (point estimates only)"
                )

                view = mo.vstack(
                    [
                        mo.md(
                            "### Overall Per-Gesture F1 Score Deltas (All Submissions)"
                        ),
                        mo.md(
                            f"Detailed breakdown showing how each gesture type benefits from additional sensor data across all submissions{bootstrap_status}. "
                            "This analysis respects both collapse gesture settings and Public/Private filters, but ignores sensor toggles (always compares both sensor subsets)."
                        ),
                        mo.md(header) if header else mo.md(""),
                        mo.md(
                            "**Note:** Individual gesture analysis uses binary F1 scores (gesture vs. all others), while the overview chart above uses the full Competition Metric (Hierarchical Macro F1)."
                        ),
                        chart,
                        mo.md("Per-gesture table (with sample counts):"),
                        mo.ui.table(
                            delta_df[table_columns], page_size=20, selection=None
                        ),
                    ]
                )
    elif score_chart.value is None or score_chart.value.empty:
        view = mo.md("*Select a submission to see All − IMU delta analysis*")
        delta_df = pd.DataFrame()
    else:
        _selected_submission_id = int(score_chart.value.iloc[0]["submission_id"])
        pred_col = f"gesture{_selected_submission_id}"

        # Filter by public/private only; always include both sensor types for delta
        delta_filtered_df = build_filter_masks(df, public_filter, private_filter)

        # Check if both sensor subsets exist
        has_all_sensors = delta_filtered_df["all_sensors"].sum() > 0
        has_imu_only = (~delta_filtered_df["all_sensors"]).sum() > 0

        if not (has_all_sensors and has_imu_only):
            view = mo.md(
                "Delta view requires both all-sensors and only-IMU samples in the selected splits. "
                "Adjust filters above."
            )
            delta_df = pd.DataFrame()
        else:
            # Get number of bootstrap samples, default to DEFAULT_BOOTSTRAP_SAMPLES if invalid
            try:
                _delta_n_bootstrap = (
                    int(bootstrap_samples.value)
                    if bootstrap_samples.value
                    else DEFAULT_BOOTSTRAP_SAMPLES
                )
            except (ValueError, TypeError):
                _delta_n_bootstrap = DEFAULT_BOOTSTRAP_SAMPLES

            # Calculate overall competition metric delta
            comp_boot = _calculate_delta(
                delta_filtered_df,
                truth_col,
                pred_col,
                "competition",
                competition_metric,
                bootstrap_toggle.value,
                _delta_n_bootstrap,
            )

            # Calculate per-gesture deltas
            rows = []

            # Process target gestures
            rows.extend(
                _process_gesture_group(
                    delta_filtered_df,
                    truth_col,
                    pred_col,
                    competition_metric.target_gestures,
                    "target",
                    "Target",
                    collapse_target_filter.value,
                    bootstrap_toggle.value,
                    _delta_n_bootstrap,
                )
            )

            # Process non-target gestures
            rows.extend(
                _process_gesture_group(
                    delta_filtered_df,
                    truth_col,
                    pred_col,
                    competition_metric.non_target_gestures,
                    "non-target",
                    "Non-Target",
                    collapse_non_target_filter.value,
                    bootstrap_toggle.value,
                    _delta_n_bootstrap,
                )
            )

            if not rows:
                view = mo.md("No valid data for delta computation.")
                delta_df = pd.DataFrame()
            else:
                delta_df = (
                    pd.DataFrame(rows)
                    .sort_values(by="delta", ascending=False)
                    .reset_index(drop=True)
                )

                # Create chart
                chart = _create_delta_chart(delta_df, bootstrap_toggle.value)
                filter_summary = _create_filter_summary()

                chart = chart.properties(
                    title=f"Per-Gesture F1 Score Delta (All − IMU) for Submission {_selected_submission_id} ({filter_summary}){' with 95% CIs' if bootstrap_toggle.value else ''}",
                    width=CHART_WIDTH,
                    height=max(BOXPLOT_HEIGHT, 20 * len(delta_df)),
                ).interactive()

                # Create header with competition metric info
                header_bits = []
                if comp_boot is not None:
                    if bootstrap_toggle.value and "ci_lo" in comp_boot:
                        header_bits.append(
                            f"Overall competition metric delta: {comp_boot['delta']:.3f} "
                            f"(95% CI [{comp_boot['ci_lo']:.3f}, {comp_boot['ci_hi']:.3f}])"
                        )
                    else:
                        header_bits.append(
                            f"Overall competition metric delta: {comp_boot['delta']:.3f}"
                        )
                    header_bits.append(
                        f"All-sensors: {comp_boot['score_all']:.3f} vs Only-IMU: {comp_boot['score_imu']:.3f}"
                    )
                header = " | ".join(header_bits) if header_bits else ""

                # Determine table columns
                table_columns = [
                    "gesture",
                    "gesture_type",
                    "f1_all",
                    "f1_imu",
                    "delta",
                    "n_all",
                    "n_imu",
                ]
                if bootstrap_toggle.value:
                    table_columns.insert(-2, "ci_lo")
                    table_columns.insert(-2, "ci_hi")

                bootstrap_status = (
                    " (with bootstrap CIs)"
                    if bootstrap_toggle.value
                    else " (point estimates only)"
                )

                view = mo.vstack(
                    [
                        mo.md(
                            f"### Per-Gesture F1 Score Deltas for Submission {_selected_submission_id}"
                        ),
                        mo.md(
                            f"Detailed breakdown showing how each gesture type benefits from additional sensor data{bootstrap_status}. "
                            "This analysis respects both collapse gesture settings and Public/Private filters, but ignores sensor toggles (always compares both sensor subsets)."
                        ),
                        mo.md(header) if header else mo.md(""),
                        mo.md(
                            "**Note:** Individual gesture analysis uses binary F1 scores (gesture vs. all others), while the overview chart above uses the full Competition Metric (Hierarchical Macro F1)."
                        ),
                        chart,
                        mo.md("Per-gesture table (with sample counts):"),
                        mo.ui.table(
                            delta_df[table_columns], page_size=20, selection=None
                        ),
                    ]
                )

    view
    return


@app.cell
def _(
    BOXPLOT_HEIGHT,
    BOXPLOT_WIDTH,
    GESTURE_TYPE_COLORS,
    METRIC_DISPLAY_ORDER,
    alt,
    calculate_binary_metrics,
    collapse_submissions_filter,
    competition_metric,
    df,
    mo,
    pd,
    score_chart,
    stack_submissions,
    submission_cols,
    truth_col,
):
    if collapse_submissions_filter.value:
        # Handle collapsed submissions mode
        # Create comprehensive dataset for all combinations including delta
        _boxplot_data = []

        # Store metrics by dataset and gesture for delta calculation
        _metrics_by_dataset_gesture = {}

        # Define the combinations for 2x3 grid (adding Delta column)
        _data_combinations = [
            ("Public", "All-sensors", True, True),
            ("Public", "Only-IMU-sensors", True, False),
            ("Private", "All-sensors", False, True),
            ("Private", "Only-IMU-sensors", False, False),
        ]

        for (
            _dataset_label,
            _sensor_label,
            _is_public,
            _is_all_sensors,
        ) in _data_combinations:
            # Filter data for this combination
            _subset_df = df[
                (df["public"] == _is_public) & (df["all_sensors"] == _is_all_sensors)
            ]

            if len(_subset_df) == 0:
                continue

            # Stack all submissions for aggregated analysis
            _stacked_df = stack_submissions(_subset_df, truth_col, submission_cols)
            _y_true = _stacked_df[
                "truth"
            ]  # Use the renamed truth column from stacked data
            _y_pred = _stacked_df["prediction"]

            # Calculate metrics for all gestures individually
            _all_gesture_labels = (
                competition_metric.target_gestures
                + competition_metric.non_target_gestures
            )

            for _gesture_label in _all_gesture_labels:
                _gesture_metrics = calculate_binary_metrics(
                    _y_true, _y_pred, _gesture_label
                )
                _gesture_type = (
                    "Target"
                    if _gesture_label in competition_metric.target_gestures
                    else "Non-Target"
                )

                # Extract only the calculated metrics (excluding TP, TN, FP, FN)
                _calculated_metrics = {
                    "Recall": _gesture_metrics["Recall"],
                    "Precision": _gesture_metrics["Precision"],
                    "Specificity": _gesture_metrics["Specificity"],
                    "F1-Score": _gesture_metrics["F1-Score"],
                    "Accuracy": _gesture_metrics["Accuracy"],
                    "NPV": _gesture_metrics["NPV"],
                }

                # Store metrics for delta calculation
                _key = (_dataset_label, _gesture_label, _gesture_type)
                if _key not in _metrics_by_dataset_gesture:
                    _metrics_by_dataset_gesture[_key] = {}
                _metrics_by_dataset_gesture[_key][_sensor_label] = _calculated_metrics

                for _metric_name, _metric_value in _calculated_metrics.items():
                    _boxplot_data.append(
                        {
                            "dataset": _dataset_label,
                            "sensor_type": _sensor_label,
                            "gesture": _gesture_label,
                            "gesture_type": _gesture_type,
                            "metric": _metric_name,
                            "value": _metric_value,
                        }
                    )

        # Calculate and add delta metrics (All sensors - IMU only)
        for (
            _dataset_label,
            _gesture_label,
            _gesture_type,
        ), _sensor_metrics in _metrics_by_dataset_gesture.items():
            if (
                "All-sensors" in _sensor_metrics
                and "Only-IMU-sensors" in _sensor_metrics
            ):
                _all_metrics = _sensor_metrics["All-sensors"]
                _imu_metrics = _sensor_metrics["Only-IMU-sensors"]

                for _metric_name in _all_metrics.keys():
                    _delta_value = (
                        _all_metrics[_metric_name] - _imu_metrics[_metric_name]
                    )
                    _boxplot_data.append(
                        {
                            "dataset": _dataset_label,
                            "sensor_type": "Delta (All - IMU)",
                            "gesture": _gesture_label,
                            "gesture_type": _gesture_type,
                            "metric": _metric_name,
                            "value": _delta_value,
                        }
                    )

        boxplot_df = pd.DataFrame(_boxplot_data)

        if len(boxplot_df) > 0:
            # Create the 2x3 grid of boxplots (now including Delta column)
            boxplot_chart = (
                alt.Chart(boxplot_df)
                .mark_boxplot(size=15, outliers={"size": 10, "opacity": 0.6})
                .encode(
                    x=alt.X(
                        "metric:N",
                        title="Evaluation Metric",
                        axis=alt.Axis(labelAngle=45),
                        sort=METRIC_DISPLAY_ORDER,
                    ),
                    xOffset=alt.XOffset(
                        "gesture_type:N",
                        title="Gesture Type",
                        sort=["Target", "Non-Target"],
                    ),
                    y=alt.Y(
                        "value:Q",
                        title="Metric Value",
                        scale=alt.Scale(zero=False),
                    ),
                    color=alt.Color(
                        "gesture_type:N",
                        title="Gesture Type",
                        scale=alt.Scale(range=GESTURE_TYPE_COLORS),
                        sort=["Target", "Non-Target"],
                    ),
                    column=alt.Column(
                        "sensor_type:N",
                        title="Sensor Configuration",
                        header=alt.Header(titleFontSize=14, labelFontSize=12),
                        sort=["All-sensors", "Only-IMU-sensors", "Delta (All - IMU)"],
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
                    width=BOXPLOT_WIDTH,  # Reduced width since we now have 3 columns
                    height=BOXPLOT_HEIGHT,
                )
                .resolve_scale(
                    y="independent"
                )  # Independent scales for different value ranges
            )

            boxplot_display = mo.vstack(
                [
                    mo.md("### Overall Metric Distributions (All Submissions)"),
                    mo.md(
                        "Boxplots showing the distribution of evaluation metrics across different data subsets and gesture types. The third column shows the delta (All-sensors - Only-IMU-sensors) for each dataset split:"
                    ),
                    boxplot_chart,
                ]
            )
        else:
            boxplot_display = mo.md("No data available for boxplot analysis")
    elif score_chart.value is not None and not score_chart.value.empty:
        _selected_submission_id = int(score_chart.value.iloc[0]["submission_id"])
        _submission_col = f"gesture{_selected_submission_id}"

        # Create comprehensive dataset for all combinations including delta
        _boxplot_data = []

        # Store metrics by dataset and gesture for delta calculation
        _metrics_by_dataset_gesture = {}

        # Define the combinations for 2x3 grid (adding Delta column)
        _data_combinations = [
            ("Public", "All-sensors", True, True),
            ("Public", "Only-IMU-sensors", True, False),
            ("Private", "All-sensors", False, True),
            ("Private", "Only-IMU-sensors", False, False),
        ]

        for (
            _dataset_label,
            _sensor_label,
            _is_public,
            _is_all_sensors,
        ) in _data_combinations:
            # Filter data for this combination
            _subset_df = df[
                (df["public"] == _is_public) & (df["all_sensors"] == _is_all_sensors)
            ]

            if len(_subset_df) == 0:
                continue

            _y_true = _subset_df[truth_col]
            _y_pred = _subset_df[_submission_col]

            # Calculate metrics for all gestures individually
            _all_gesture_labels = (
                competition_metric.target_gestures
                + competition_metric.non_target_gestures
            )

            for _gesture_label in _all_gesture_labels:
                _gesture_metrics = calculate_binary_metrics(
                    _y_true, _y_pred, _gesture_label
                )
                _gesture_type = (
                    "Target"
                    if _gesture_label in competition_metric.target_gestures
                    else "Non-Target"
                )

                # Extract only the calculated metrics (excluding TP, TN, FP, FN)
                _calculated_metrics = {
                    "Recall": _gesture_metrics["Recall"],
                    "Precision": _gesture_metrics["Precision"],
                    "Specificity": _gesture_metrics["Specificity"],
                    "F1-Score": _gesture_metrics["F1-Score"],
                    "Accuracy": _gesture_metrics["Accuracy"],
                    "NPV": _gesture_metrics["NPV"],
                }

                # Store metrics for delta calculation
                _key = (_dataset_label, _gesture_label, _gesture_type)
                if _key not in _metrics_by_dataset_gesture:
                    _metrics_by_dataset_gesture[_key] = {}
                _metrics_by_dataset_gesture[_key][_sensor_label] = _calculated_metrics

                for _metric_name, _metric_value in _calculated_metrics.items():
                    _boxplot_data.append(
                        {
                            "dataset": _dataset_label,
                            "sensor_type": _sensor_label,
                            "gesture": _gesture_label,
                            "gesture_type": _gesture_type,
                            "metric": _metric_name,
                            "value": _metric_value,
                        }
                    )

        # Calculate and add delta metrics (All sensors - IMU only)
        for (
            _dataset_label,
            _gesture_label,
            _gesture_type,
        ), _sensor_metrics in _metrics_by_dataset_gesture.items():
            if (
                "All-sensors" in _sensor_metrics
                and "Only-IMU-sensors" in _sensor_metrics
            ):
                _all_metrics = _sensor_metrics["All-sensors"]
                _imu_metrics = _sensor_metrics["Only-IMU-sensors"]

                for _metric_name in _all_metrics.keys():
                    _delta_value = (
                        _all_metrics[_metric_name] - _imu_metrics[_metric_name]
                    )
                    _boxplot_data.append(
                        {
                            "dataset": _dataset_label,
                            "sensor_type": "Delta (All - IMU)",
                            "gesture": _gesture_label,
                            "gesture_type": _gesture_type,
                            "metric": _metric_name,
                            "value": _delta_value,
                        }
                    )

        boxplot_df = pd.DataFrame(_boxplot_data)

        if len(boxplot_df) > 0:
            # Create the 2x3 grid of boxplots (now including Delta column)
            boxplot_chart = (
                alt.Chart(boxplot_df)
                .mark_boxplot(size=15, outliers={"size": 10, "opacity": 0.6})
                .encode(
                    x=alt.X(
                        "metric:N",
                        title="Evaluation Metric",
                        axis=alt.Axis(labelAngle=45),
                        sort=METRIC_DISPLAY_ORDER,
                    ),
                    xOffset=alt.XOffset(
                        "gesture_type:N",
                        title="Gesture Type",
                        sort=["Target", "Non-Target"],
                    ),
                    y=alt.Y(
                        "value:Q",
                        title="Metric Value",
                        scale=alt.Scale(zero=False),
                    ),
                    color=alt.Color(
                        "gesture_type:N",
                        title="Gesture Type",
                        scale=alt.Scale(range=GESTURE_TYPE_COLORS),
                        sort=["Target", "Non-Target"],
                    ),
                    column=alt.Column(
                        "sensor_type:N",
                        title="Sensor Configuration",
                        header=alt.Header(titleFontSize=14, labelFontSize=12),
                        sort=["All-sensors", "Only-IMU-sensors", "Delta (All - IMU)"],
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
                    width=BOXPLOT_WIDTH,  # Reduced width since we now have 3 columns
                    height=BOXPLOT_HEIGHT,
                )
                .resolve_scale(
                    y="independent"
                )  # Independent scales for different value ranges
            )

            boxplot_display = mo.vstack(
                [
                    mo.md(
                        f"### Distributions of Metrics for Submission {_selected_submission_id}"
                    ),
                    mo.md(
                        "Boxplots showing the distribution of evaluation metrics across different data subsets and gesture types. The third column shows the delta (All-sensors - Only-IMU-sensors) for each dataset split:"
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
