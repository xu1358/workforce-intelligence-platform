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
    "policy_summary": (
        "current_policy_summary.csv"
    ),
    "policy_group": (
        "current_policy_by_department.csv"
    ),
    "metadata": (
        "dashboard_metadata.csv"
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

policy_summary = (
    data[
        "policy_summary"
    ]
)

policy_group = (
    data[
        "policy_group"
    ]
)

metadata = (
    data[
        "metadata"
    ]
)

metadata_row = (
    metadata.iloc[0]
)

policy_summary_row = (
    policy_summary.iloc[0]
)


# ---------------------------------------------------------
# Dashboard title
# ---------------------------------------------------------

st.title(
    "Workforce Intelligence and "
    "Retention Decision Platform"
)

st.caption(
    f"{metadata_row['workforce_status_label']} — "
    f"data as of {metadata_row['as_of_date']}."
)

st.info(
    f"**{metadata_row['model_status_label']}**  \n"
    f"{metadata_row['probability_label']} for the "
    f"{int(metadata_row['score_horizon_months'])} months "
    "after the as-of date."
)

with st.expander(
    "How to interpret this current workforce view",
):

    st.write(
        metadata_row[
            "current_score_caveat"
        ]
    )

    st.write(
        metadata_row[
            "legacy_score_caveat"
        ]
    )

    st.warning(
        metadata_row[
            "use_notice"
        ]
    )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title(
    "Dashboard Controls"
)

st.sidebar.caption(
    "The filters below apply to the "
    "current active retention plan."
)


# Department filter
department_options = sorted(
    risk_employees[
        "department_name"
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
        "region"
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


# Probability-band filter
band_options = (
    risk_employees[
        "probability_band"
    ]
    .dropna()
    .drop_duplicates()
    .tolist()
)

selected_probability_bands = (
    st.sidebar.multiselect(
        "Probability Band",
        options=(
            band_options
        ),
        default=(
            band_options
        ),
    )
)

# Review-status filter
review_options = (
    risk_employees[
        "review_status"
    ]
    .dropna()
    .drop_duplicates()
    .tolist()
)

selected_review_statuses = (
    st.sidebar.multiselect(
        "Review Status",
        options=(
            review_options
        ),
        default=(
            review_options
        ),
    )
)


# ---------------------------------------------------------
# Apply retention filters
# ---------------------------------------------------------

filtered_risk = (
    risk_employees[
        risk_employees["department_name"].isin(
            selected_departments
        )
        &
        risk_employees["region"].isin(
            selected_regions
        )
        &
        risk_employees["probability_band"].isin(
            selected_probability_bands
        )
        &
        risk_employees["review_status"].isin(
            selected_review_statuses
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
            "Selected for Human Review",
            format_integer(
                policy_summary_row[
                    "selected_for_human_review"
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
            "Current Eligible Population",
            format_integer(
                policy_summary_row[
                    "eligible_employees"
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
            "Current Review Plan by Department"
        )

        risk_chart = px.bar(
            policy_group.sort_values(
                "selected_for_human_review",
                ascending=False,
            ),
            x="department_name",
            y="selected_for_human_review",
            labels={
                "department_name": (
                    "Department"
                ),
                "selected_for_human_review": (
                    "Selected for Human Review"
                ),
            },
        )

        risk_chart.update_layout(
            xaxis_tickangle=-35
        )

        st.plotly_chart(
            risk_chart,
            width="stretch",
        )


    st.info(
        metadata_row[
            "use_notice"
        ]
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
        "Current Active Retention Plan"
    )

    st.caption(
        f"Only model-eligible employees active as of "
        f"{metadata_row['as_of_date']} are shown. "
        "Filters in the sidebar apply to this section."
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

        selected_review_count = int(
            filtered_risk[
                "selected_for_human_review"
            ].sum()
        )

        average_net_value = (
            filtered_risk[
                "predicted_net_value_usd"
            ]
            .mean()
        )

    else:

        average_probability = 0

        selected_review_count = 0

        average_net_value = 0


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
            "Average Estimated Probability",
            format_decimal_percent(
                average_probability
            ),
        )


    with retention_kpi_3:

        st.metric(
            "Selected for Human Review",
            format_integer(
                selected_review_count
            ),
        )


    with retention_kpi_4:

        st.metric(
            "Average Predicted Net Value",
            f"${average_net_value:,.0f}",
        )


    st.divider()


    # -----------------------------------------------------
    # Review-status distribution
    # -----------------------------------------------------

    st.subheader(
        "Probability Bands and Review Status"
    )

    if (
        filtered_employee_count
        > 0
    ):

        filtered_distribution = (
            filtered_risk.groupby(
                [
                    "probability_band",
                    "review_status",
                ],
                as_index=False,
            )
            .agg(
                employee_count=(
                    "employee_id",
                    "count",
                )
            )
        )

        risk_distribution_chart = (
            px.bar(
                filtered_distribution,
                x="probability_band",
                y="employee_count",
                color="review_status",
                barmode="stack",
                labels={
                    "probability_band": (
                        "Probability Band"
                    ),
                    "employee_count": (
                        "Employees"
                    ),
                    "review_status": (
                        "Review Status"
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
        metadata_row[
            "probability_label"
        ]
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
                        "Estimated Probability"
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
        "Human-Review Prioritization"
    )

    employee_columns = [
        "employee_id",
        "as_of_date",
        "employment_status",
        "department_name",
        "city",
        "region",
        "organizational_level",
        "employment_type",
        "job_level",
        "base_salary",
        "attrition_probability",
        "probability_band",
        "predicted_net_value_usd",
        "expected_value_rank",
        "selected_for_human_review",
        "review_status",
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
            [
                "selected_for_human_review",
                "expected_value_rank",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    st.dataframe(
        filtered_employee_table
    )


    st.warning(
        f"{metadata_row['synthetic_data_notice']} "
        f"{metadata_row['use_notice']}"
    )


# =========================================================
# MODEL PERFORMANCE TAB
# =========================================================

with model_tab:

    st.header(
        "Version 2 Model Evidence"
    )

    summary_row = (
        model_summary.iloc[0]
    )


    # -----------------------------------------------------
    # Selected model
    # -----------------------------------------------------

    st.subheader(
        "Selected Model, Calibration, and Policy"
    )

    st.success(
        f"{summary_row['selected_model']} with "
        f"{summary_row['selected_calibration_method']} calibration"
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
            "Final Test PR-AUC",
            f"{float(summary_row['final_test_pr_auc']):.4f}",
        )


    with model_kpi_2:

        st.metric(
            "Final Test ROC-AUC",
            f"{float(summary_row['final_test_roc_auc']):.4f}",
        )


    with model_kpi_3:

        st.metric(
            "Final Test Brier Score",
            f"{float(summary_row['final_test_brier_score']):.4f}",
        )


    with model_kpi_4:

        st.metric(
            "Final Test Snapshot",
            str(
                summary_row[
                    "final_test_snapshot"
                ]
            ),
        )


    # -----------------------------------------------------
    # Frozen policy
    # -----------------------------------------------------

    st.subheader(
        "Frozen Current Planning Policy"
    )

    (
        policy_1,
        policy_2,
        policy_3,
    ) = st.columns(
        3
    )


    with policy_1:

        st.metric(
            "Policy",
            str(
                summary_row[
                    "selected_policy"
                ]
            ),
        )


    with policy_2:

        st.metric(
            "Maximum Reviews",
            format_integer(
                summary_row[
                    "maximum_employees"
                ]
            ),
        )


    with policy_3:

        st.metric(
            "Budget",
            f"${float(summary_row['budget_usd']):,.0f}",
        )


    st.divider()


    # -----------------------------------------------------
    # Model comparison
    # -----------------------------------------------------

    st.subheader(
        "Once-Only Out-of-Time Final Test"
    )

    st.dataframe(
        model_performance
    )


    st.info(
        "The displayed performance metrics come from the "
        "reserved 2025 final test. The current 2026 workforce "
        "has no known future outcome, so current scores are "
        "planning projections and are not used to claim new "
        "model performance."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Workforce Intelligence and Retention "
    "Decision Platform | Synthetic portfolio project"
)
