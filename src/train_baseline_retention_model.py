from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ---------------------------------------------------------
# Project settings
# ---------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retention_modeling_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "baseline_retention_model.joblib"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_retention_predictions.csv"
)

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ---------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------

TARGET_COLUMN = (
    "attrition_next_12m"
)


EXCLUDED_COLUMNS = [
    "employee_id",
    "snapshot_date",
    "prediction_end_date",
    TARGET_COLUMN,
]


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
# Load data
# ---------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Load the retention modeling dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing retention modeling dataset. "
            "Run build_retention_dataset.py first."
        )

    dataset = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Loaded dataset: "
        f"{len(dataset):,} rows, "
        f"{len(dataset.columns)} columns"
    )

    return dataset


# ---------------------------------------------------------
# Validate feature definitions
# ---------------------------------------------------------

def validate_features(
    dataset: pd.DataFrame,
) -> None:
    """Validate required modeling columns."""

    required_columns = set(
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
        + EXCLUDED_COLUMNS
    )

    missing_columns = (
        required_columns
        - set(dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing modeling columns: "
            f"{sorted(missing_columns)}"
        )

    if dataset[
        TARGET_COLUMN
    ].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    if not set(
        dataset[
            TARGET_COLUMN
        ].astype(int)
    ).issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            "Target must contain only 0 and 1."
        )


# ---------------------------------------------------------
# Split features and target
# ---------------------------------------------------------

def prepare_features_and_target(
    dataset: pd.DataFrame,
):
    """Create X and y."""

    feature_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
    )

    X = dataset[
        feature_columns
    ].copy()

    y = (
        dataset[
            TARGET_COLUMN
        ]
        .astype(int)
        .copy()
    )

    return X, y


# ---------------------------------------------------------
# Build preprocessing pipeline
# ---------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Create numerical and categorical preprocessing."""

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


# ---------------------------------------------------------
# Build logistic-regression pipeline
# ---------------------------------------------------------

def build_model() -> Pipeline:
    """Build the baseline logistic-regression pipeline."""

    preprocessor = (
        build_preprocessor()
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return model


# ---------------------------------------------------------
# Evaluate one model
# ---------------------------------------------------------

def evaluate_model(
    model_name: str,
    y_true: pd.Series,
    y_pred,
    y_probability,
) -> dict:
    """Calculate classification metrics."""

    metrics = {
        "model": model_name,

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_probability,
        ),
    }

    return metrics


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------

def main() -> None:
    """Train and evaluate baseline retention models."""

    dataset = (
        load_dataset()
    )

    validate_features(
        dataset
    )

    X, y = (
        prepare_features_and_target(
            dataset
        )
    )

    employee_ids = (
        dataset[
            "employee_id"
        ]
        .copy()
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        employee_train,
        employee_test,
    ) = train_test_split(
        X,
        y,
        employee_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        "\nTrain/test split:"
    )

    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing rows: "
        f"{len(X_test):,}"
    )

    print(
        "\nTraining target distribution:"
    )

    print(
        y_train
        .value_counts()
        .sort_index()
    )

    print(
        "\nTesting target distribution:"
    )

    print(
        y_test
        .value_counts()
        .sort_index()
    )

    # -----------------------------------------------------
    # Dummy baseline
    # -----------------------------------------------------

    dummy_model = DummyClassifier(
        strategy="prior",
        random_state=RANDOM_STATE,
    )

    dummy_model.fit(
        X_train,
        y_train,
    )

    dummy_predictions = (
        dummy_model.predict(
            X_test
        )
    )

    dummy_probabilities = (
        dummy_model.predict_proba(
            X_test
        )[:, 1]
    )

    dummy_metrics = (
        evaluate_model(
            model_name=(
                "Dummy Classifier"
            ),
            y_true=y_test,
            y_pred=(
                dummy_predictions
            ),
            y_probability=(
                dummy_probabilities
            ),
        )
    )

    # -----------------------------------------------------
    # Logistic regression
    # -----------------------------------------------------

    model = (
        build_model()
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = (
        model.predict(
            X_test
        )
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    logistic_metrics = (
        evaluate_model(
            model_name=(
                "Logistic Regression"
            ),
            y_true=y_test,
            y_pred=predictions,
            y_probability=(
                probabilities
            ),
        )
    )

    # -----------------------------------------------------
    # Display metrics
    # -----------------------------------------------------

    metrics_table = pd.DataFrame(
        [
            dummy_metrics,
            logistic_metrics,
        ]
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    metrics_table[
        metric_columns
    ] = (
        metrics_table[
            metric_columns
        ]
        .round(4)
    )

    print(
        "\nModel comparison:"
    )

    print(
        metrics_table.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print(
        "\nLogistic Regression "
        "confusion matrix:"
    )

    print(
        matrix
    )

    # -----------------------------------------------------
    # Save test predictions
    # -----------------------------------------------------

    predictions_output = (
        pd.DataFrame(
            {
                "employee_id": (
                    employee_test
                    .to_numpy()
                ),
                "actual_attrition": (
                    y_test
                    .to_numpy()
                ),
                "predicted_attrition": (
                    predictions
                ),
                "attrition_probability": (
                    probabilities
                ),
            }
        )
        .sort_values(
            "employee_id"
        )
        .reset_index(
            drop=True
        )
    )

    predictions_output.to_csv(
        PREDICTION_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Save trained pipeline
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"\nSaved model: "
        f"{MODEL_PATH}"
    )

    print(
        f"Saved predictions: "
        f"{PREDICTION_PATH}"
    )

    print(
        "\nBaseline retention model "
        "training completed successfully."
    )


if __name__ == "__main__":
    main()