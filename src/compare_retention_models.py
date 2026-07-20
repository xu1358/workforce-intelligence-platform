from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
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
from sklearn.utils.class_weight import (
    compute_sample_weight,
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

SELECTED_MODEL_PATH = (
    MODEL_DIR
    / "selected_retention_model.joblib"
)

COMPARISON_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retention_model_comparison.csv"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "selected_retention_predictions.csv"
)

RANDOM_STATE = 42

TEST_SIZE = 0.20

TARGET_COLUMN = (
    "attrition_next_12m"
)


# ---------------------------------------------------------
# Feature definitions
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


FEATURE_COLUMNS = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Load retention modeling data."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing retention modeling dataset."
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
# Validate data
# ---------------------------------------------------------

def validate_dataset(
    dataset: pd.DataFrame,
) -> None:
    """Validate required model columns."""

    required_columns = set(
        FEATURE_COLUMNS
        + [
            "employee_id",
            TARGET_COLUMN,
        ]
    )

    missing_columns = (
        required_columns
        - set(dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataset[
        TARGET_COLUMN
    ].nunique() != 2:
        raise ValueError(
            "Target must contain both classes."
        )


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing for model features."""

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
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
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


# ---------------------------------------------------------
# Build candidate models
# ---------------------------------------------------------

def build_models() -> dict:
    """Create the models to compare."""

    logistic_regression = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight=(
                        "balanced_subsample"
                    ),
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    gradient_boosting = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": (
            logistic_regression
        ),
        "Random Forest": (
            random_forest
        ),
        "Gradient Boosting": (
            gradient_boosting
        ),
    }


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_model(
    model_name: str,
    y_true,
    predictions,
    probabilities,
) -> dict:
    """Calculate model performance metrics."""

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
        ).ravel()
    )

    return {
        "model": model_name,

        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities,
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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    """Train, compare, and select attrition models."""

    dataset = (
        load_dataset()
    )

    validate_dataset(
        dataset
    )

    X = dataset[
        FEATURE_COLUMNS
    ].copy()

    y = (
        dataset[
            TARGET_COLUMN
        ]
        .astype(int)
        .copy()
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
        "\nTraining rows:",
        len(X_train),
    )

    print(
        "Testing rows:",
        len(X_test),
    )

    models = (
        build_models()
    )

    results = []

    prediction_results = {}

    for (
        model_name,
        model,
    ) in models.items():

        print(
            f"\nTraining: "
            f"{model_name}"
        )

        if (
            model_name
            == "Gradient Boosting"
        ):
            sample_weights = (
                compute_sample_weight(
                    class_weight="balanced",
                    y=y_train,
                )
            )

            model.fit(
                X_train,
                y_train,
                classifier__sample_weight=(
                    sample_weights
                ),
            )

        else:

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

        metrics = evaluate_model(
            model_name=model_name,
            y_true=y_test,
            predictions=predictions,
            probabilities=probabilities,
        )

        results.append(
            metrics
        )

        prediction_results[
            model_name
        ] = {
            "predictions": predictions,
            "probabilities": (
                probabilities
            ),
        }

    # -----------------------------------------------------
    # Compare models
    # -----------------------------------------------------

    comparison = pd.DataFrame(
        results
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    comparison[
        metric_columns
    ] = (
        comparison[
            metric_columns
        ]
        .round(4)
    )

    comparison = (
        comparison
        .sort_values(
            [
                "pr_auc",
                "roc_auc",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nModel comparison:"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Select best model
    # -----------------------------------------------------

    selected_model_name = (
        comparison.loc[
            0,
            "model",
        ]
    )

    selected_model = (
        models[
            selected_model_name
        ]
    )

    print(
        "\nSelected model "
        "(highest PR-AUC):"
    )

    print(
        selected_model_name
    )

    # -----------------------------------------------------
    # Save comparison
    # -----------------------------------------------------

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Save selected predictions
    # -----------------------------------------------------

    selected_predictions = (
        prediction_results[
            selected_model_name
        ][
            "predictions"
        ]
    )

    selected_probabilities = (
        prediction_results[
            selected_model_name
        ][
            "probabilities"
        ]
    )

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
                    selected_predictions
                ),

                "attrition_probability": (
                    selected_probabilities
                ),

                "model": (
                    selected_model_name
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
    # Save selected model
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        selected_model,
        SELECTED_MODEL_PATH,
    )

    # -----------------------------------------------------
    # Validate outputs
    # -----------------------------------------------------

    if len(comparison) != 3:
        raise ValueError(
            "Expected three models."
        )

    if (
        len(predictions_output)
        != len(y_test)
    ):
        raise ValueError(
            "Prediction row count mismatch."
        )

    print(
        f"\nSaved comparison: "
        f"{COMPARISON_PATH}"
    )

    print(
        f"Saved selected model: "
        f"{SELECTED_MODEL_PATH}"
    )

    print(
        f"Saved predictions: "
        f"{PREDICTION_PATH}"
    )

    print(
        "\nRetention model comparison "
        "completed successfully."
    )


if __name__ == "__main__":
    main()