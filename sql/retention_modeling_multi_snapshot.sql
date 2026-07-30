/*
Executable Version 2 temporal-equivalence query
================================================

Checkpoint 63 executes this read-only query and compares every resulting
employee-snapshot value with the Python implementation in
src/build_multi_snapshot_retention_dataset.py:

1. Construct three historical snapshots with twelve-month outcomes.
2. Construct one current active-workforce snapshot with no known target.
3. Use only feature records dated on or before each snapshot.
4. Reconstruct historical location from hire and prior transfer events.

The query uses the same least-privilege analytical views as the Python
PostgreSQL backend. Snapshot literals are checked against
config/temporal_snapshots.yaml before execution. The equivalence validator
checks keys, columns, null patterns, exact values, numeric tolerances, and
aggregate fingerprints; neither implementation is trusted by assertion alone.
*/

WITH snapshot_definitions (
    snapshot_sequence,
    dataset_type,
    snapshot_date,
    prediction_end_date
) AS (
    VALUES
        (
            1,
            'historical',
            DATE '2023-06-30',
            DATE '2024-06-30'
        ),
        (
            2,
            'historical',
            DATE '2024-06-30',
            DATE '2025-06-30'
        ),
        (
            3,
            'historical',
            DATE '2025-06-30',
            DATE '2026-06-30'
        ),
        (
            4,
            'current_scoring',
            DATE '2026-06-30',
            NULL::DATE
        )
),

eligible_employee_snapshots AS (
    SELECT
        e.employee_id,
        s.dataset_type,
        s.snapshot_sequence,
        s.snapshot_date,
        s.prediction_end_date,
        e.hire_date,
        e.birth_year,
        e.employment_type,
        e.education_level,
        e.organizational_level,
        e.department_id,
        e.location_id AS final_location_id,
        e.job_role_id,

        CASE
            WHEN s.dataset_type = 'historical'
            THEN COALESCE(
                e.termination_date > s.snapshot_date
                AND e.termination_date <= s.prediction_end_date,
                FALSE
            )::INTEGER
            ELSE NULL::INTEGER
        END AS attrition_next_12m

    FROM analytics_v2_employees AS e

    CROSS JOIN snapshot_definitions AS s

    WHERE
        e.hire_date <= s.snapshot_date
        AND (
            e.termination_date IS NULL
            OR e.termination_date > s.snapshot_date
        )
),

point_in_time_employee AS (
    SELECT
        es.*,

        COALESCE(
            (
                SELECT ee.new_value::INTEGER
                FROM analytics_v2_employee_events AS ee
                WHERE
                    ee.employee_id = es.employee_id
                    AND ee.event_type = 'Transfer'
                    AND ee.event_date <= es.snapshot_date
                ORDER BY
                    ee.event_date DESC,
                    ee.event_id DESC
                LIMIT 1
            ),
            (
                SELECT
                    SUBSTRING(
                        ee.notes
                        FROM 'location_id=([0-9]+)'
                    )::INTEGER
                FROM analytics_v2_employee_events AS ee
                WHERE
                    ee.employee_id = es.employee_id
                    AND ee.event_type = 'Hire'
                ORDER BY
                    ee.event_date,
                    ee.event_id
                LIMIT 1
            ),
            es.final_location_id
        ) AS snapshot_location_id

    FROM eligible_employee_snapshots AS es
),

base_features AS (
    SELECT
        pe.employee_id,
        pe.dataset_type,
        pe.snapshot_sequence,
        pe.snapshot_date,
        pe.prediction_end_date,
        pe.birth_year,
        pe.snapshot_date::DATE - pe.hire_date::DATE
            AS tenure_days,
        pe.employment_type,
        pe.education_level,
        pe.organizational_level,
        d.department_name,
        d.department_group,
        l.city,
        l.region,
        l.location_type,
        jr.job_title,
        jr.job_family,
        jr.job_level,

        initial_compensation.base_salary
            AS initial_base_salary,
        current_compensation.base_salary,
        current_compensation.bonus_target,
        current_compensation.equity_value,

        100.0 * (
            current_compensation.base_salary
            / NULLIF(
                initial_compensation.base_salary,
                0
            )
            - 1.0
        ) AS salary_growth_percent,

        100.0 * (
            current_compensation.base_salary
            / NULLIF(
                COALESCE(
                    prior_year_compensation.base_salary,
                    initial_compensation.base_salary
                ),
                0
            )
            - 1.0
        ) AS salary_growth_12m_percent,

        pe.snapshot_date::DATE
            - current_compensation.effective_date::DATE
            AS days_since_compensation_change,

        compensation_summary.compensation_record_count,
        compensation_summary.promotion_compensation_count,

        performance_summary.performance_rating,
        performance_summary.goal_completion,
        performance_summary.promotion_recommended,
        pe.snapshot_date::DATE
            - performance_summary.latest_review_date::DATE
            AS days_since_review,
        COALESCE(
            performance_summary.review_count,
            0
        ) AS review_count,
        performance_summary.average_performance_rating,
        COALESCE(
            performance_summary.performance_rating
            - performance_summary.first_performance_rating,
            0.0
        ) AS performance_trend,
        (
            performance_summary.latest_review_date IS NULL
        )::INTEGER AS no_prior_review,

        COALESCE(
            training_summary.completed_training_programs_12m,
            0
        ) AS completed_training_programs_12m,
        COALESCE(
            training_summary.failed_training_programs_12m,
            0
        ) AS failed_training_programs_12m,
        COALESCE(
            training_summary.completed_training_hours_12m,
            0.0
        ) AS completed_training_hours_12m,
        training_summary.average_training_score_12m,

        COALESCE(
            event_summary.prior_promotion_events,
            0
        ) AS prior_promotion_events,
        COALESCE(
            event_summary.prior_transfer_events,
            0
        ) AS prior_transfer_events,
        COALESCE(
            event_summary.prior_manager_change_events,
            0
        ) AS prior_manager_change_events,
        COALESCE(
            event_summary.prior_leave_events,
            0
        ) AS prior_leave_events,
        COALESCE(
            event_summary.prior_change_events,
            0
        ) AS prior_change_events,
        COALESCE(
            event_summary.promotions_12m,
            0
        ) AS promotions_12m,
        COALESCE(
            event_summary.manager_changes_12m,
            0
        ) AS manager_changes_12m,
        COALESCE(
            event_summary.leaves_12m,
            0
        ) AS leaves_12m,
        pe.snapshot_date::DATE
            - event_summary.latest_change_date::DATE
            AS days_since_last_change_event,

        COALESCE(
            (
                EXTRACT(YEAR FROM pe.snapshot_date) * 12
                + EXTRACT(MONTH FROM pe.snapshot_date)
                - EXTRACT(
                    YEAR FROM event_summary.latest_promotion_date
                ) * 12
                - EXTRACT(
                    MONTH FROM event_summary.latest_promotion_date
                )
            )::DOUBLE PRECISION,
            (
                pe.snapshot_date::DATE
                - pe.hire_date::DATE
            ) / 365.25 * 12.0
        ) AS months_since_promotion,
        (
            event_summary.latest_promotion_date IS NULL
        )::INTEGER AS no_prior_promotion,

        pe.attrition_next_12m

    FROM point_in_time_employee AS pe

    JOIN analytics_v2_departments AS d
        ON d.department_id = pe.department_id

    JOIN analytics_v2_locations AS l
        ON l.location_id = pe.snapshot_location_id

    JOIN analytics_v2_job_roles AS jr
        ON jr.job_role_id = pe.job_role_id

    LEFT JOIN LATERAL (
        SELECT ch.base_salary
        FROM analytics_v2_compensation_history AS ch
        WHERE
            ch.employee_id = pe.employee_id
            AND ch.effective_date <= pe.snapshot_date
        ORDER BY
            ch.effective_date,
            ch.compensation_id
        LIMIT 1
    ) AS initial_compensation
        ON TRUE

    LEFT JOIN LATERAL (
        SELECT
            ch.effective_date,
            ch.base_salary,
            ch.bonus_target,
            ch.equity_value
        FROM analytics_v2_compensation_history AS ch
        WHERE
            ch.employee_id = pe.employee_id
            AND ch.effective_date <= pe.snapshot_date
        ORDER BY
            ch.effective_date DESC,
            ch.compensation_id DESC
        LIMIT 1
    ) AS current_compensation
        ON TRUE

    LEFT JOIN LATERAL (
        SELECT ch.base_salary
        FROM analytics_v2_compensation_history AS ch
        WHERE
            ch.employee_id = pe.employee_id
            AND ch.effective_date
                <= pe.snapshot_date - INTERVAL '1 year'
        ORDER BY
            ch.effective_date DESC,
            ch.compensation_id DESC
        LIMIT 1
    ) AS prior_year_compensation
        ON TRUE

    LEFT JOIN LATERAL (
        SELECT
            COUNT(*)::INTEGER
                AS compensation_record_count,
            COUNT(*) FILTER (
                WHERE ch.change_reason = 'Promotion'
            )::INTEGER
                AS promotion_compensation_count
        FROM analytics_v2_compensation_history AS ch
        WHERE
            ch.employee_id = pe.employee_id
            AND ch.effective_date <= pe.snapshot_date
    ) AS compensation_summary
        ON TRUE

    LEFT JOIN LATERAL (
        SELECT
            (
                ARRAY_AGG(
                    pr.performance_rating
                    ORDER BY
                        pr.review_date DESC,
                        pr.review_id DESC
                )
            )[1] AS performance_rating,
            (
                ARRAY_AGG(
                    pr.goal_completion
                    ORDER BY
                        pr.review_date DESC,
                        pr.review_id DESC
                )
            )[1] AS goal_completion,
            (
                ARRAY_AGG(
                    pr.promotion_recommended
                    ORDER BY
                        pr.review_date DESC,
                        pr.review_id DESC
                )
            )[1] AS promotion_recommended,
            MAX(pr.review_date) AS latest_review_date,
            COUNT(*)::INTEGER AS review_count,
            AVG(pr.performance_rating)
                AS average_performance_rating,
            (
                ARRAY_AGG(
                    pr.performance_rating
                    ORDER BY
                        pr.review_date,
                        pr.review_id
                )
            )[1] AS first_performance_rating
        FROM analytics_v2_performance_reviews AS pr
        WHERE
            pr.employee_id = pe.employee_id
            AND pr.review_date <= pe.snapshot_date
    ) AS performance_summary
        ON TRUE

    LEFT JOIN LATERAL (
        SELECT
            COUNT(*) FILTER (
                WHERE tr.completion_status = 'Completed'
            )::INTEGER
                AS completed_training_programs_12m,
            COUNT(*) FILTER (
                WHERE tr.completion_status = 'Failed'
            )::INTEGER
                AS failed_training_programs_12m,
            SUM(
                CASE
                    WHEN tr.completion_status = 'Completed'
                    THEN tr.training_hours
                    ELSE 0.0
                END
            ) AS completed_training_hours_12m,
            AVG(tr.score) FILTER (
                WHERE tr.completion_status = 'Completed'
            ) AS average_training_score_12m
        FROM analytics_v2_training_records AS tr
        WHERE
            tr.employee_id = pe.employee_id
            AND tr.completion_date IS NOT NULL
            AND tr.completion_date
                > pe.snapshot_date - INTERVAL '1 year'
            AND tr.completion_date <= pe.snapshot_date
    ) AS training_summary
        ON TRUE

    LEFT JOIN LATERAL (
        SELECT
            COUNT(*) FILTER (
                WHERE ee.event_type = 'Promotion'
            )::INTEGER AS prior_promotion_events,
            COUNT(*) FILTER (
                WHERE ee.event_type = 'Transfer'
            )::INTEGER AS prior_transfer_events,
            COUNT(*) FILTER (
                WHERE ee.event_type = 'Manager Change'
            )::INTEGER AS prior_manager_change_events,
            COUNT(*) FILTER (
                WHERE ee.event_type = 'Leave'
            )::INTEGER AS prior_leave_events,
            COUNT(*) FILTER (
                WHERE ee.event_type IN (
                    'Promotion',
                    'Transfer',
                    'Manager Change',
                    'Leave'
                )
            )::INTEGER AS prior_change_events,
            COUNT(*) FILTER (
                WHERE
                    ee.event_type = 'Promotion'
                    AND ee.event_date
                        > pe.snapshot_date - INTERVAL '1 year'
            )::INTEGER AS promotions_12m,
            COUNT(*) FILTER (
                WHERE
                    ee.event_type = 'Manager Change'
                    AND ee.event_date
                        > pe.snapshot_date - INTERVAL '1 year'
            )::INTEGER AS manager_changes_12m,
            COUNT(*) FILTER (
                WHERE
                    ee.event_type = 'Leave'
                    AND ee.event_date
                        > pe.snapshot_date - INTERVAL '1 year'
            )::INTEGER AS leaves_12m,
            MAX(ee.event_date) FILTER (
                WHERE ee.event_type IN (
                    'Promotion',
                    'Transfer',
                    'Manager Change',
                    'Leave'
                )
            ) AS latest_change_date,
            MAX(ee.event_date) FILTER (
                WHERE ee.event_type = 'Promotion'
            ) AS latest_promotion_date
        FROM analytics_v2_employee_events AS ee
        WHERE
            ee.employee_id = pe.employee_id
            AND ee.event_date <= pe.snapshot_date
    ) AS event_summary
        ON TRUE
),

peer_medians AS (
    SELECT
        snapshot_date,
        job_family,
        job_level,
        PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY base_salary
        ) AS peer_median_salary
    FROM base_features
    GROUP BY
        snapshot_date,
        job_family,
        job_level
)

SELECT
    bf.employee_id,
    bf.dataset_type,
    bf.snapshot_sequence,
    bf.snapshot_date,
    bf.prediction_end_date,
    EXTRACT(YEAR FROM bf.snapshot_date)::INTEGER
        - bf.birth_year AS approx_age,
    bf.tenure_days / 365.25 AS tenure_years,
    bf.employment_type,
    bf.education_level,
    bf.organizational_level,
    bf.department_name,
    bf.department_group,
    bf.city,
    bf.region,
    bf.location_type,
    bf.job_title,
    bf.job_family,
    bf.job_level,
    bf.initial_base_salary,
    bf.base_salary,
    bf.bonus_target,
    bf.equity_value,
    bf.salary_growth_percent,
    bf.salary_growth_12m_percent,
    100.0 * (
        bf.base_salary
        / NULLIF(pm.peer_median_salary, 0)
        - 1.0
    ) AS salary_position_percent,
    bf.days_since_compensation_change,
    bf.compensation_record_count,
    bf.promotion_compensation_count,
    bf.performance_rating,
    bf.goal_completion,
    bf.promotion_recommended,
    bf.days_since_review,
    bf.review_count,
    bf.average_performance_rating,
    bf.performance_trend,
    bf.no_prior_review,
    bf.completed_training_programs_12m,
    bf.failed_training_programs_12m,
    bf.completed_training_hours_12m,
    bf.average_training_score_12m,
    bf.prior_promotion_events,
    bf.prior_transfer_events,
    bf.prior_manager_change_events,
    bf.prior_leave_events,
    bf.prior_change_events,
    bf.promotions_12m,
    bf.manager_changes_12m,
    bf.leaves_12m,
    bf.days_since_last_change_event,
    bf.months_since_promotion,
    bf.no_prior_promotion,
    bf.attrition_next_12m

FROM base_features AS bf

LEFT JOIN peer_medians AS pm
    ON pm.snapshot_date = bf.snapshot_date
    AND pm.job_family = bf.job_family
    AND pm.job_level = bf.job_level

ORDER BY
    bf.snapshot_sequence,
    bf.employee_id;
