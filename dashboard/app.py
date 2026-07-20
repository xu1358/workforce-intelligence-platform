from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DASHBOARD_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dashboard"
)


# ---------------------------------------------------------
# Required dashboard files
# ---------------------------------------------------------

DASHBOARD_FILES = {
    "overview": (
        "overview_kpis.csv"
    ),
    "department": (
        "headcount_by_department.csv"
    ),
    "location": (
        "headcount_by_location.csv"
    ),
    "recruiting": (
        "recruiting_funnel.csv"
    ),
    "requisitions": (
        "requisition_metrics.csv"
    ),
    "risk_summary": (
        "retention_risk_summary.csv"
    ),
    "risk_employees": (
        "retention_risk_employees.csv"
    ),
    "model_performance": (
        "model_performance.csv"
    ),
    "model_summary": (
        "model_summary.csv"
    ),
}


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title=(
        "Workforce Intelligence Platform"
    ),
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

@st.cache_data
def load_dashboard_data():
    """Load all dashboard-ready CSV files."""

    datasets = {}

    for (
        dataset_name,
        filename,
    ) in DASHBOARD_FILES.items():

        path = (
            DASHBOARD_DATA_DIR
            / filename
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing dashboard file: "
                f"{path}"
            )

        datasets[
            dataset_name
        ] = pd.read_csv(
            path
        )

    return datasets


# ---------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------

def format_integer(
    value,
) -> str:
    """Format a value as an integer."""

    return (
        f"{int(value):,}"
    )


def format_percent(
    value,
) -> str:
    """Format an already-percent value."""

    return (
        f"{float(value):.2f}%"
    )


def format_decimal_percent(
    value,
) -> str:
    """Convert a decimal ratio to a percentage."""

    return (
        f"{100 * float(value):.2f}%"
    )


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

try:

    data = (
        load_dashboard_data()
    )

except FileNotFoundError as error:

    st.error(
        "Dashboard data is missing."
    )

    st.write(
        str(
            error
        )
    )

    st.write(
        "Build the dashboard data layer "
        "before starting the dashboard:"
    )

    st.code(
        "python src\\build_dashboard_data.py"
    )

    st.stop()


overview = (
    data[
        "overview"
    ]
)

department = (
    data[
        "department"
    ]
)

location = (
    data[
        "location"
    ]
)

recruiting = (
    data[
        "recruiting"
    ]
)

requisitions = (
    data[
        "requisitions"
    ]
)

risk_summary = (
    data[
        "risk_summary"
    ]
)

risk_employees = (
    data[
        "risk_employees"
    ]
)

model_performance = (
    data[
        "model_performance"
    ]
)

model_summary = (
    data[
        "model_summary"
    ]
)


# ---------------------------------------------------------
# Dashboard title
# ---------------------------------------------------------

st.title(
    "Workforce Intelligence and "
    "Retention Decision Platform"
)

st.caption(
    "Synthetic workforce analytics, "
    "recruiting intelligence, and "
    "employee retention risk analysis."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title(
    "Dashboard Controls"
)

st.sidebar.caption(
    "The filters below apply to the "
    "Retention Risk employee analysis."
)


# Department filter
department_options = sorted(
    risk_employees[
        "hire_department_name"
    ]
    .dropna()
    .unique()
    .tolist()
)

selected_departments = (
    st.sidebar.multiselect(
        "Department",
        options=(
            department_options
        ),
        default=(
            department_options
        ),
    )
)


# Region filter
region_options = sorted(
    risk_employees[
        "hire_region"
    ]
    .dropna()
    .unique()
    .tolist()
)

selected_regions = (
    st.sidebar.multiselect(
        "Region",
        options=(
            region_options
        ),
        default=(
            region_options
        ),
    )
)


# Risk segment filter
risk_options = [
    "High",
    "Medium",
    "Low",
]

selected_risk_segments = (
    st.sidebar.multiselect(
        "Risk Segment",
        options=(
            risk_options
        ),
        default=(
            risk_options
        ),
    )
)


# ---------------------------------------------------------
# Apply retention filters
# ---------------------------------------------------------

filtered_risk = (
    risk_employees[
        risk_employees[
            "hire_department_name"
        ].isin(
            selected_departments
        )
        &
        risk_employees[
            "hire_region"
        ].isin(
            selected_regions
        )
        &
        risk_employees[
            "risk_segment"
        ].isin(
            selected_risk_segments
        )
    ]
    .copy()
)


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

(
    overview_tab,
    workforce_tab,
    recruiting_tab,
    retention_tab,
    model_tab,
) = st.tabs(
    [
        "Overview",
        "Workforce",
        "Recruiting",
        "Retention Risk",
        "Model Performance",
    ]
)


# =========================================================
# OVERVIEW TAB
# =========================================================

with overview_tab:

    st.header(
        "Executive Workforce Overview"
    )

    overview_row = (
        overview.iloc[0]
    )

    # -----------------------------------------------------
    # KPI row 1
    # -----------------------------------------------------

    (
        kpi_1,
        kpi_2,
        kpi_3,
        kpi_4,
    ) = st.columns(
        4
    )

    with kpi_1:

        st.metric(
            "Active Headcount",
            format_integer(
                overview_row[
                    "active_headcount"
                ]
            ),
        )

    with kpi_2:

        st.metric(
            "2025 Turnover Rate",
            format_percent(
                overview_row[
                    "turnover_rate_2025_percent"
                ]
            ),
        )

    with kpi_3:

        st.metric(
            "Open Requisitions",
            format_integer(
                overview_row[
                    "open_requisitions"
                ]
            ),
        )

    with kpi_4:

        st.metric(
            "High-Risk Employees",
            format_integer(
                overview_row[
                    "high_risk_employees"
                ]
            ),
        )


    # -----------------------------------------------------
    # KPI row 2
    # -----------------------------------------------------

    (
        kpi_5,
        kpi_6,
        kpi_7,
        kpi_8,
    ) = st.columns(
        4
    )

    with kpi_5:

        st.metric(
            "Total Applications",
            format_integer(
                overview_row[
                    "total_applications"
                ]
            ),
        )

    with kpi_6:

        st.metric(
            "Hired Applications",
            format_integer(
                overview_row[
                    "hired_applications"
                ]
            ),
        )

    with kpi_7:

        st.metric(
            "Retention Snapshot Population",
            format_integer(
                overview_row[
                    "retention_snapshot_population"
                ]
            ),
        )

    with kpi_8:

        st.metric(
            "Active Locations",
            format_integer(
                overview_row[
                    "active_location_count"
                ]
            ),
        )


    st.divider()


    # -----------------------------------------------------
    # Overview charts
    # -----------------------------------------------------

    (
        overview_left,
        overview_right,
    ) = st.columns(
        2
    )


    with overview_left:

        st.subheader(
            "Active Headcount by Department"
        )

        department_chart = px.bar(
            department,
            x="department_name",
            y="active_headcount",
            labels={
                "department_name": (
                    "Department"
                ),
                "active_headcount": (
                    "Active Headcount"
                ),
            },
        )

        department_chart.update_layout(
            xaxis_tickangle=-35
        )

        st.plotly_chart(
            department_chart,
            width="stretch",
        )


    with overview_right:

        st.subheader(
            "Retention Risk Population"
        )

        risk_chart = px.bar(
            risk_summary,
            x="risk_segment",
            y="employee_count",
            category_orders={
                "risk_segment": [
                    "Low",
                    "Medium",
                    "High",
                ]
            },
            labels={
                "risk_segment": (
                    "Risk Segment"
                ),
                "employee_count": (
                    "Employees"
                ),
            },
        )

        st.plotly_chart(
            risk_chart,
            width="stretch",
        )


    st.info(
        "Retention risk scores are model-based "
        "prioritization scores generated from "
        "synthetic workforce data. They are not "
        "causal estimates or guaranteed departure "
        "probabilities."
    )


# =========================================================
# WORKFORCE TAB
# =========================================================

with workforce_tab:

    st.header(
        "Workforce Distribution"
    )


    # -----------------------------------------------------
    # Department
    # -----------------------------------------------------

    st.subheader(
        "Headcount by Department"
    )

    department_chart = px.bar(
        department,
        x="department_name",
        y="active_headcount",
        hover_data=[
            "headcount_percent",
        ],
        labels={
            "department_name": (
                "Department"
            ),
            "active_headcount": (
                "Active Headcount"
            ),
            "headcount_percent": (
                "Workforce %"
            ),
        },
    )

    department_chart.update_layout(
        xaxis_tickangle=-35
    )

    st.plotly_chart(
        department_chart,
        width="stretch",
    )

    st.dataframe(
        department
    )


    st.divider()


    # -----------------------------------------------------
    # Location
    # -----------------------------------------------------

    st.subheader(
        "Headcount by Location"
    )

    location_chart = px.bar(
        location,
        x="location_name",
        y="active_headcount",
        hover_data=[
            "region",
            "headcount_percent",
        ],
        labels={
            "location_name": (
                "Location"
            ),
            "active_headcount": (
                "Active Headcount"
            ),
            "headcount_percent": (
                "Workforce %"
            ),
            "region": (
                "Region"
            ),
        },
    )

    location_chart.update_layout(
        xaxis_tickangle=-35
    )

    st.plotly_chart(
        location_chart,
        width="stretch",
    )

    st.dataframe(
        location
    )


# =========================================================
# RECRUITING TAB
# =========================================================

with recruiting_tab:

    st.header(
        "Recruiting Intelligence"
    )


    # -----------------------------------------------------
    # Recruiting outcomes
    # -----------------------------------------------------

    st.subheader(
        "Recruiting Funnel by Application Source"
    )

    recruiting_long = (
        recruiting
        .melt(
            id_vars=[
                "application_source",
            ],
            value_vars=[
                "hired",
                "rejected",
                "withdrawn",
                "offer_declined",
                "in_process",
                "position_cancelled",
            ],
            var_name=(
                "application_status"
            ),
            value_name=(
                "application_count"
            ),
        )
    )

    recruiting_chart = px.bar(
        recruiting_long,
        x="application_source",
        y="application_count",
        color="application_status",
        barmode="group",
        labels={
            "application_source": (
                "Application Source"
            ),
            "application_count": (
                "Applications"
            ),
            "application_status": (
                "Status"
            ),
        },
    )

    recruiting_chart.update_layout(
        xaxis_tickangle=-35
    )

    st.plotly_chart(
        recruiting_chart,
        width="stretch",
    )


    # -----------------------------------------------------
    # Hire rate
    # -----------------------------------------------------

    st.subheader(
        "Hire Rate by Application Source"
    )

    hire_rate_chart = px.bar(
        recruiting.sort_values(
            "hire_rate_percent",
            ascending=False,
        ),
        x="application_source",
        y="hire_rate_percent",
        labels={
            "application_source": (
                "Application Source"
            ),
            "hire_rate_percent": (
                "Hire Rate (%)"
            ),
        },
    )

    hire_rate_chart.update_layout(
        xaxis_tickangle=-35
    )

    st.plotly_chart(
        hire_rate_chart,
        width="stretch",
    )

    st.dataframe(
        recruiting
    )


    st.divider()


    # -----------------------------------------------------
    # Requisitions
    # -----------------------------------------------------

    st.subheader(
        "Job Requisition Metrics"
    )

    requisition_chart = px.bar(
        requisitions,
        x="requisition_status",
        y="requisition_count",
        hover_data=[
            "target_headcount",
            "average_days_open",
        ],
        labels={
            "requisition_status": (
                "Requisition Status"
            ),
            "requisition_count": (
                "Requisitions"
            ),
            "target_headcount": (
                "Target Headcount"
            ),
            "average_days_open": (
                "Average Days Open"
            ),
        },
    )

    st.plotly_chart(
        requisition_chart,
        width="stretch",
    )

    st.dataframe(
        requisitions
    )


# =========================================================
# RETENTION RISK TAB
# =========================================================

with retention_tab:

    st.header(
        "Employee Retention Risk"
    )

    st.caption(
        "Filters in the sidebar apply to "
        "this section."
    )


    # -----------------------------------------------------
    # Filtered KPIs
    # -----------------------------------------------------

    filtered_employee_count = (
        len(
            filtered_risk
        )
    )

    if (
        filtered_employee_count
        > 0
    ):

        average_probability = (
            filtered_risk[
                "attrition_probability"
            ]
            .mean()
        )

        high_risk_count = int(
            (
                filtered_risk[
                    "risk_segment"
                ]
                == "High"
            )
            .sum()
        )

        average_tenure = (
            filtered_risk[
                "tenure_years"
            ]
            .mean()
        )

    else:

        average_probability = 0

        high_risk_count = 0

        average_tenure = 0


    (
        retention_kpi_1,
        retention_kpi_2,
        retention_kpi_3,
        retention_kpi_4,
    ) = st.columns(
        4
    )


    with retention_kpi_1:

        st.metric(
            "Employees in Current View",
            format_integer(
                filtered_employee_count
            ),
        )


    with retention_kpi_2:

        st.metric(
            "Average Risk Score",
            format_decimal_percent(
                average_probability
            ),
        )


    with retention_kpi_3:

        st.metric(
            "High-Risk Employees",
            format_integer(
                high_risk_count
            ),
        )


    with retention_kpi_4:

        st.metric(
            "Average Tenure",
            f"{average_tenure:.2f} years",
        )


    st.divider()


    # -----------------------------------------------------
    # Risk distribution
    # -----------------------------------------------------

    st.subheader(
        "Risk Segment Distribution"
    )

    if (
        filtered_employee_count
        > 0
    ):

        filtered_distribution = (
            filtered_risk[
                "risk_segment"
            ]
            .value_counts()
            .rename_axis(
                "risk_segment"
            )
            .reset_index(
                name=(
                    "employee_count"
                )
            )
        )

        risk_distribution_chart = (
            px.bar(
                filtered_distribution,
                x="risk_segment",
                y="employee_count",
                category_orders={
                    "risk_segment": [
                        "Low",
                        "Medium",
                        "High",
                    ]
                },
                labels={
                    "risk_segment": (
                        "Risk Segment"
                    ),
                    "employee_count": (
                        "Employees"
                    ),
                },
            )
        )

        st.plotly_chart(
            risk_distribution_chart,
            width="stretch",
        )

    else:

        st.warning(
            "No employees match the "
            "selected filters."
        )


    # -----------------------------------------------------
    # Risk score distribution
    # -----------------------------------------------------

    st.subheader(
        "Attrition Risk Score Distribution"
    )

    if (
        filtered_employee_count
        > 0
    ):

        probability_chart = (
            px.histogram(
                filtered_risk,
                x=(
                    "attrition_probability"
                ),
                nbins=30,
                labels={
                    "attrition_probability": (
                        "Attrition Risk Score"
                    ),
                },
            )
        )

        st.plotly_chart(
            probability_chart,
            width="stretch",
        )


    # -----------------------------------------------------
    # Employee table
    # -----------------------------------------------------

    st.subheader(
        "Highest-Ranked Retention Risks"
    )

    employee_columns = [
        "employee_id",
        "hire_department_name",
        "hire_region",
        "hire_job_family",
        "hire_job_level",
        "employment_type",
        "tenure_years",
        "base_salary",
        "performance_rating",
        "attrition_probability",
        "risk_segment",
    ]


    available_employee_columns = [
        column
        for column
        in employee_columns
        if column
        in filtered_risk.columns
    ]


    filtered_employee_table = (
        filtered_risk[
            available_employee_columns
        ]
        .sort_values(
            "attrition_probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    st.dataframe(
        filtered_employee_table
    )


    st.warning(
        "This project uses synthetic employees. "
        "In a real HR system, individual risk "
        "predictions would require appropriate "
        "privacy, governance, fairness, and "
        "human-review controls."
    )


# =========================================================
# MODEL PERFORMANCE TAB
# =========================================================

with model_tab:

    st.header(
        "Retention Model Performance"
    )

    summary_row = (
        model_summary.iloc[0]
    )


    # -----------------------------------------------------
    # Selected model
    # -----------------------------------------------------

    st.subheader(
        "Selected Model"
    )

    st.success(
        summary_row[
            "selected_model"
        ]
    )


    (
        model_kpi_1,
        model_kpi_2,
        model_kpi_3,
        model_kpi_4,
    ) = st.columns(
        4
    )


    with model_kpi_1:

        st.metric(
            "Test Recall",
            format_decimal_percent(
                summary_row[
                    "test_recall"
                ]
            ),
        )


    with model_kpi_2:

        st.metric(
            "Test Precision",
            format_decimal_percent(
                summary_row[
                    "test_precision"
                ]
            ),
        )


    with model_kpi_3:

        st.metric(
            "ROC-AUC",
            f"{float(summary_row['test_roc_auc']):.4f}",
        )


    with model_kpi_4:

        st.metric(
            "PR-AUC",
            f"{float(summary_row['test_pr_auc']):.4f}",
        )


    # -----------------------------------------------------
    # Thresholds
    # -----------------------------------------------------

    st.subheader(
        "Decision Thresholds"
    )

    (
        threshold_1,
        threshold_2,
        threshold_3,
    ) = st.columns(
        3
    )


    with threshold_1:

        st.metric(
            "Classification Threshold",
            f"{float(summary_row['recommended_classification_threshold']):.2f}",
        )


    with threshold_2:

        st.metric(
            "Medium Risk Starts",
            f"{float(summary_row['medium_risk_threshold']):.4f}",
        )


    with threshold_3:

        st.metric(
            "High Risk Starts",
            f"{float(summary_row['high_risk_threshold']):.4f}",
        )


    st.divider()


    # -----------------------------------------------------
    # Model comparison
    # -----------------------------------------------------

    st.subheader(
        "Model Comparison"
    )

    model_metric_long = (
        model_performance
        .melt(
            id_vars=[
                "model",
            ],
            value_vars=[
                "roc_auc",
                "pr_auc",
            ],
            var_name=(
                "metric"
            ),
            value_name=(
                "score"
            ),
        )
    )


    comparison_chart = px.bar(
        model_metric_long,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        labels={
            "model": (
                "Model"
            ),
            "score": (
                "Score"
            ),
            "metric": (
                "Metric"
            ),
        },
    )


    st.plotly_chart(
        comparison_chart,
        width="stretch",
    )


    st.dataframe(
        model_performance
    )


    st.info(
        "The model was selected primarily "
        "using PR-AUC because employee "
        "attrition is an imbalanced "
        "classification problem. Model "
        "performance is measured on the "
        "held-out test set."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Workforce Intelligence and Retention "
    "Decision Platform | Synthetic portfolio project"
)