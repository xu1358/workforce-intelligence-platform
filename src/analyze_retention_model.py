from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------
# Project settings
# ---------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "selected_retention_model.joblib"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "selected_retention_predictions.csv"
)

FEATURE_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retention_feature_importance.csv"
)

FEATURE_GROUP_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retention_feature_group_importance.csv"
)

THRESHOLD_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retention_threshold_analysis.csv"
)

RISK_SEGMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retention_risk_segments.csv"
)


# ---------------------------------------------------------
# Original feature definitions
# ---------------------------------------------------------

NUMERICAL_FEATURES = [
    "approx_age",
    "tenure_years",
    "years_experience_at_hire",
    "initial_base_salary",
    "base_salary",
    "bonus_target",
    "equity_value",
    "salary_growth_percent",
    "days_since_compensation_change",
    "compensation_record_count",
    "promotion_compensation_count",
    "performance_rating",
    "goal_completion",
    "days_since_review",
    "review_count",
    "average_performance_rating",
    "completed_training_programs",
    "failed_training_programs",
    "in_progress_training_programs",
    "completed_training_hours",
    "average_training_score",
    "prior_promotion_events",
    "prior_transfer_events",
    "prior_manager_change_events",
    "prior_leave_events",
    "prior_change_events",
    "days_since_last_change_event",
]


CATEGORICAL_FEATURES = [
    "employment_type",
    "education_level",
    "application_source",
    "hire_department_name",
    "hire_region",
    "hire_job_family",
    "hire_job_level",
    "promotion_recommended",
]


# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------

def load_model():
    """Load the selected retention model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing selected model. "
            "Run compare_retention_models.py first."
        )

    return joblib.load(
        MODEL_PATH
    )


def load_predictions() -> pd.DataFrame:
    """Load selected-model test predictions."""

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            "Missing selected retention predictions."
        )

    predictions = pd.read_csv(
        PREDICTION_PATH
    )

    required_columns = {
        "employee_id",
        "actual_attrition",
        "predicted_attrition",
        "attrition_probability",
        "model",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing prediction columns: "
            f"{sorted(missing_columns)}"
        )

    return predictions


# ---------------------------------------------------------
# Feature source helper
# ---------------------------------------------------------

def identify_source_feature(
    transformed_feature: str,
) -> str:
    """
    Map a transformed feature back to
    its original source feature.
    """

    clean_name = (
        transformed_feature
        .replace(
            "numerical__",
            "",
        )
        .replace(
            "categorical__",
            "",
        )
    )

    for feature in (
        NUMERICAL_FEATURES
    ):
        if clean_name == feature:
            return feature

    for feature in (
        CATEGORICAL_FEATURES
    ):
        prefix = (
            f"{feature}_"
        )

        if clean_name.startswith(
            prefix
        ):
            return feature

    return clean_name


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

def create_feature_importance(
    model,
) -> pd.DataFrame:
    """Extract feature importance from the selected model."""

    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    if hasattr(
        classifier,
        "coef_",
    ):
        raw_values = (
            classifier.coef_[0]
        )

        importance_values = (
            np.abs(
                raw_values
            )
        )

        directions = np.where(
            raw_values > 0,
            "Higher attrition risk",
            np.where(
                raw_values < 0,
                "Lower attrition risk",
                "Neutral",
            ),
        )

    elif hasattr(
        classifier,
        "feature_importances_",
    ):
        raw_values = (
            classifier
            .feature_importances_
        )

        importance_values = (
            raw_values
        )

        directions = np.repeat(
            "Non-directional",
            len(raw_values),
        )

    else:
        raise ValueError(
            "Selected model does not expose "
            "supported feature importance."
        )

    if (
        len(feature_names)
        != len(importance_values)
    ):
        raise ValueError(
            "Feature-name and importance "
            "length mismatch."
        )

    importance = pd.DataFrame(
        {
            "transformed_feature": (
                feature_names
            ),
            "raw_effect": (
                raw_values
            ),
            "importance": (
                importance_values
            ),
            "direction": (
                directions
            ),
        }
    )

    importance[
        "source_feature"
    ] = (
        importance[
            "transformed_feature"
        ]
        .apply(
            identify_source_feature
        )
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return importance


# ---------------------------------------------------------
# Group feature importance
# ---------------------------------------------------------

def create_feature_group_importance(
    feature_importance: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate transformed features to original feature groups."""

    grouped = (
        feature_importance
        .groupby(
            "source_feature",
            as_index=False,
        )
        .agg(
            total_importance=(
                "importance",
                "sum",
            ),
            transformed_feature_count=(
                "transformed_feature",
                "count",
            ),
        )
        .sort_values(
            "total_importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return grouped


# ---------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------

def create_threshold_analysis(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate classification performance at multiple thresholds."""

    y_true = (
        predictions[
            "actual_attrition"
        ]
        .astype(int)
        .to_numpy()
    )

    probabilities = (
        predictions[
            "attrition_probability"
        ]
        .to_numpy()
    )

    thresholds = np.arange(
        0.10,
        0.91,
        0.05,
    )

    records = []

    for threshold in thresholds:

        predicted = (
            probabilities
            >= threshold
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                predicted,
                labels=[
                    0,
                    1,
                ],
            )
            .ravel()
        )

        records.append(
            {
                "threshold": round(
                    float(
                        threshold
                    ),
                    2,
                ),

                "accuracy": (
                    accuracy_score(
                        y_true,
                        predicted,
                    )
                ),

                "precision": (
                    precision_score(
                        y_true,
                        predicted,
                        zero_division=0,
                    )
                ),

                "recall": (
                    recall_score(
                        y_true,
                        predicted,
                        zero_division=0,
                    )
                ),

                "f1": (
                    f1_score(
                        y_true,
                        predicted,
                        zero_division=0,
                    )
                ),

                "predicted_attrition_count": (
                    int(
                        predicted.sum()
                    )
                ),

                "true_negative": int(
                    tn
                ),

                "false_positive": int(
                    fp
                ),

                "false_negative": int(
                    fn
                ),

                "true_positive": int(
                    tp
                ),
            }
        )

    analysis = pd.DataFrame(
        records
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    analysis[
        metric_columns
    ] = (
        analysis[
            metric_columns
        ]
        .round(4)
    )

    return analysis


# ---------------------------------------------------------
# Recommended threshold
# ---------------------------------------------------------

def select_recommended_threshold(
    threshold_analysis: pd.DataFrame,
) -> float:
    """
    Select a practical threshold.

    Project rule:
    Keep recall at or above 60%,
    then choose the threshold with
    the highest precision.
    """

    eligible = (
        threshold_analysis[
            threshold_analysis[
                "recall"
            ]
            >= 0.60
        ]
        .copy()
    )

    if not eligible.empty:

        selected = (
            eligible
            .sort_values(
                [
                    "precision",
                    "f1",
                    "threshold",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

    else:

        selected = (
            threshold_analysis
            .sort_values(
                "f1",
                ascending=False,
            )
            .iloc[0]
        )

    return float(
        selected[
            "threshold"
        ]
    )


# ---------------------------------------------------------
# Risk segmentation
# ---------------------------------------------------------

def create_risk_segments(
    predictions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    float,
    float,
]:
    """
    Divide test employees into risk bands.

    High risk = top 10% of scores.
    Medium risk = next 20%.
    Low risk = remaining 70%.
    """

    risk_data = (
        predictions
        .copy()
    )

    high_threshold = float(
        risk_data[
            "attrition_probability"
        ]
        .quantile(
            0.90
        )
    )

    medium_threshold = float(
        risk_data[
            "attrition_probability"
        ]
        .quantile(
            0.70
        )
    )

    risk_data[
        "risk_segment"
    ] = np.select(
        [
            risk_data[
                "attrition_probability"
            ]
            >= high_threshold,

            risk_data[
                "attrition_probability"
            ]
            >= medium_threshold,
        ],
        [
            "High",
            "Medium",
        ],
        default="Low",
    )

    return (
        risk_data,
        medium_threshold,
        high_threshold,
    )


# ---------------------------------------------------------
# Print summaries
# ---------------------------------------------------------

def print_feature_summary(
    importance: pd.DataFrame,
    grouped_importance: pd.DataFrame,
) -> None:
    """Print important model features."""

    print(
        "\nTop 15 transformed features:"
    )

    print(
        importance[
            [
                "transformed_feature",
                "importance",
                "direction",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print(
        "\nTop 15 original feature groups:"
    )

    print(
        grouped_importance
        .head(15)
        .to_string(
            index=False
        )
    )


def print_threshold_summary(
    threshold_analysis: pd.DataFrame,
    recommended_threshold: float,
) -> None:
    """Print threshold results."""

    print(
        "\nThreshold analysis:"
    )

    print(
        threshold_analysis
        .to_string(
            index=False
        )
    )

    print(
        "\nRecommended classification threshold:"
    )

    print(
        recommended_threshold
    )


def print_risk_summary(
    risk_segments: pd.DataFrame,
    medium_threshold: float,
    high_threshold: float,
) -> None:
    """Print risk-segment statistics."""

    summary = (
        risk_segments
        .groupby(
            "risk_segment"
        )
        .agg(
            employee_count=(
                "employee_id",
                "count",
            ),

            average_predicted_probability=(
                "attrition_probability",
                "mean",
            ),

            actual_attrition_rate=(
                "actual_attrition",
                "mean",
            ),
        )
    )

    order = [
        "Low",
        "Medium",
        "High",
    ]

    summary = (
        summary
        .reindex(
            order
        )
    )

    print(
        "\nRisk segment thresholds:"
    )

    print(
        f"Medium risk starts at: "
        f"{medium_threshold:.4f}"
    )

    print(
        f"High risk starts at: "
        f"{high_threshold:.4f}"
    )

    print(
        "\nRisk segment summary:"
    )

    print(
        summary.to_string()
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    """Run model interpretation and threshold analysis."""

    model = (
        load_model()
    )

    predictions = (
        load_predictions()
    )

    selected_model_names = (
        predictions[
            "model"
        ]
        .dropna()
        .unique()
    )

    if (
        len(
            selected_model_names
        )
        != 1
    ):
        raise ValueError(
            "Expected exactly one "
            "selected model."
        )

    selected_model_name = (
        selected_model_names[
            0
        ]
    )

    print(
        "Selected model:"
    )

    print(
        selected_model_name
    )

    # -----------------------------------------------------
    # Feature interpretation
    # -----------------------------------------------------

    feature_importance = (
        create_feature_importance(
            model
        )
    )

    grouped_importance = (
        create_feature_group_importance(
            feature_importance
        )
    )

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    grouped_importance.to_csv(
        FEATURE_GROUP_IMPORTANCE_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Threshold analysis
    # -----------------------------------------------------

    threshold_analysis = (
        create_threshold_analysis(
            predictions
        )
    )

    recommended_threshold = (
        select_recommended_threshold(
            threshold_analysis
        )
    )

    threshold_analysis.to_csv(
        THRESHOLD_ANALYSIS_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Risk segmentation
    # -----------------------------------------------------

    (
        risk_segments,
        medium_threshold,
        high_threshold,
    ) = create_risk_segments(
        predictions
    )

    risk_segments.to_csv(
        RISK_SEGMENT_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print_feature_summary(
        feature_importance,
        grouped_importance,
    )

    print_threshold_summary(
        threshold_analysis,
        recommended_threshold,
    )

    print_risk_summary(
        risk_segments,
        medium_threshold,
        high_threshold,
    )

    print(
        "\nSaved feature importance:"
    )

    print(
        FEATURE_IMPORTANCE_PATH
    )

    print(
        "\nSaved grouped feature importance:"
    )

    print(
        FEATURE_GROUP_IMPORTANCE_PATH
    )

    print(
        "\nSaved threshold analysis:"
    )

    print(
        THRESHOLD_ANALYSIS_PATH
    )

    print(
        "\nSaved risk segments:"
    )

    print(
        RISK_SEGMENT_PATH
    )

    print(
        "\nRetention model interpretation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()