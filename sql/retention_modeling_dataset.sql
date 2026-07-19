-- =========================================================
-- Workforce Intelligence Platform
-- Retention Modeling Dataset
-- =========================================================
--
-- Snapshot date:
-- 2025-06-30
--
-- Prediction window:
-- 2025-07-01 through 2026-06-30
--
-- Target:
-- attrition_next_12m
-- =========================================================


WITH


-- =========================================================
-- 1. Analysis dates
-- =========================================================

params AS (
    SELECT
        DATE '2025-06-30'
            AS snapshot_date,

        DATE '2026-06-30'
            AS prediction_end_date
),


-- =========================================================
-- 2. Modeling population
-- =========================================================

population AS (
    SELECT
        e.*

    FROM employees AS e

    CROSS JOIN params AS p

    WHERE
        e.hire_date
            <= p.snapshot_date

        AND (
            e.termination_date IS NULL
            OR e.termination_date
                > p.snapshot_date
        )
),


-- =========================================================
-- 3. Original hiring context
-- =========================================================

hire_context AS (
    SELECT DISTINCT ON (
        a.employee_id
    )
        a.employee_id,

        c.application_source,

        c.years_experience
            AS years_experience_at_hire,

        d.department_name
            AS hire_department_name,

        l.region
            AS hire_region,

        jr.job_family
            AS hire_job_family,

        jr.job_level
            AS hire_job_level

    FROM applications AS a

    JOIN candidates AS c
        ON a.candidate_id
           = c.candidate_id

    JOIN job_requisitions AS req
        ON a.requisition_id
           = req.requisition_id

    JOIN departments AS d
        ON req.department_id
           = d.department_id

    JOIN locations AS l
        ON req.location_id
           = l.location_id

    JOIN job_roles AS jr
        ON req.job_role_id
           = jr.job_role_id

    CROSS JOIN params AS p

    WHERE
        a.application_status = 'Hired'

        AND a.employee_id
            IS NOT NULL

        AND a.decision_date
            <= p.snapshot_date

    ORDER BY
        a.employee_id,
        a.decision_date,
        a.application_id
),


-- =========================================================
-- 4. Initial compensation
-- =========================================================

initial_compensation AS (
    SELECT DISTINCT ON (
        ch.employee_id
    )
        ch.employee_id,
        ch.base_salary
            AS initial_base_salary

    FROM compensation_history AS ch

    CROSS JOIN params AS p

    WHERE
        ch.effective_date
            <= p.snapshot_date

    ORDER BY
        ch.employee_id,
        ch.effective_date ASC,
        ch.compensation_id ASC
),


-- =========================================================
-- 5. Latest compensation at snapshot
-- =========================================================

latest_compensation AS (
    SELECT DISTINCT ON (
        ch.employee_id
    )
        ch.employee_id,
        ch.effective_date,
        ch.base_salary,
        ch.bonus_target,
        ch.equity_value

    FROM compensation_history AS ch

    CROSS JOIN params AS p

    WHERE
        ch.effective_date
            <= p.snapshot_date

    ORDER BY
        ch.employee_id,
        ch.effective_date DESC,
        ch.compensation_id DESC
),


-- =========================================================
-- 6. Compensation history summary
-- =========================================================

compensation_summary AS (
    SELECT
        ch.employee_id,

        COUNT(*)
            AS compensation_record_count,

        COUNT(*) FILTER (
            WHERE
                ch.change_reason = 'Promotion'
        ) AS promotion_compensation_count

    FROM compensation_history AS ch

    CROSS JOIN params AS p

    WHERE
        ch.effective_date
            <= p.snapshot_date

    GROUP BY
        ch.employee_id
),


-- =========================================================
-- 7. Latest performance review
-- =========================================================

latest_review AS (
    SELECT DISTINCT ON (
        pr.employee_id
    )
        pr.employee_id,
        pr.review_date,
        pr.performance_rating,
        pr.goal_completion,
        pr.promotion_recommended

    FROM performance_reviews AS pr

    CROSS JOIN params AS p

    WHERE
        pr.review_date
            <= p.snapshot_date

    ORDER BY
        pr.employee_id,
        pr.review_date DESC,
        pr.review_id DESC
),


-- =========================================================
-- 8. Performance history summary
-- =========================================================

review_summary AS (
    SELECT
        pr.employee_id,

        COUNT(*)
            AS review_count,

        AVG(
            pr.performance_rating
        ) AS average_performance_rating

    FROM performance_reviews AS pr

    CROSS JOIN params AS p

    WHERE
        pr.review_date
            <= p.snapshot_date

    GROUP BY
        pr.employee_id
),


-- =========================================================
-- 9. Training history available at snapshot
-- =========================================================

training_summary AS (
    SELECT
        tr.employee_id,

        COUNT(*) FILTER (
            WHERE
                tr.completion_status = 'Completed'
                AND tr.completion_date
                    <= p.snapshot_date
        ) AS completed_training_programs,

        COUNT(*) FILTER (
            WHERE
                tr.completion_status = 'Failed'
                AND tr.completion_date
                    <= p.snapshot_date
        ) AS failed_training_programs,

        COUNT(*) FILTER (
            WHERE
                tr.start_date
                    <= p.snapshot_date

                AND (
                    tr.completion_date IS NULL
                    OR tr.completion_date
                        > p.snapshot_date
                )
        ) AS in_progress_training_programs,

        COALESCE(
            SUM(
                tr.training_hours
            ) FILTER (
                WHERE
                    tr.completion_status = 'Completed'
                    AND tr.completion_date
                        <= p.snapshot_date
            ),
            0
        ) AS completed_training_hours,

        AVG(
            tr.score
        ) FILTER (
            WHERE
                tr.completion_date
                    <= p.snapshot_date

                AND tr.score
                    IS NOT NULL
        ) AS average_training_score

    FROM training_records AS tr

    CROSS JOIN params AS p

    WHERE
        tr.start_date
            <= p.snapshot_date

    GROUP BY
        tr.employee_id
),


-- =========================================================
-- 10. Employee events available at snapshot
-- =========================================================

event_summary AS (
    SELECT
        ee.employee_id,

        COUNT(*) FILTER (
            WHERE
                ee.event_type = 'Promotion'
        ) AS prior_promotion_events,

        COUNT(*) FILTER (
            WHERE
                ee.event_type = 'Transfer'
        ) AS prior_transfer_events,

        COUNT(*) FILTER (
            WHERE
                ee.event_type = 'Manager Change'
        ) AS prior_manager_change_events,

        COUNT(*) FILTER (
            WHERE
                ee.event_type = 'Leave'
        ) AS prior_leave_events,

        COUNT(*) FILTER (
            WHERE
                ee.event_type <> 'Hire'
        ) AS prior_change_events,

        MAX(
            ee.event_date
        ) FILTER (
            WHERE
                ee.event_type <> 'Hire'
        ) AS last_change_event_date

    FROM employee_events AS ee

    CROSS JOIN params AS p

    WHERE
        ee.event_date
            <= p.snapshot_date

    GROUP BY
        ee.employee_id
)


-- =========================================================
-- 11. Final modeling dataset
-- =========================================================

SELECT
    pop.employee_id,

    prm.snapshot_date,

    prm.prediction_end_date,

    EXTRACT(
        YEAR
        FROM prm.snapshot_date
    )::INTEGER
        - pop.birth_year
        AS approx_age,

    ROUND(
        (
            prm.snapshot_date
            - pop.hire_date
        )::NUMERIC
        / 365.25,
        2
    ) AS tenure_years,

    pop.employment_type,

    pop.education_level,

    hc.application_source,

    hc.years_experience_at_hire,

    hc.hire_department_name,

    hc.hire_region,

    hc.hire_job_family,

    hc.hire_job_level,

    ic.initial_base_salary,

    lc.base_salary,

    lc.bonus_target,

    lc.equity_value,

    ROUND(
        100.0
        * (
            lc.base_salary
            - ic.initial_base_salary
        )
        / NULLIF(
            ic.initial_base_salary,
            0
        ),
        2
    ) AS salary_growth_percent,

    (
        prm.snapshot_date
        - lc.effective_date
    ) AS days_since_compensation_change,

    COALESCE(
        cs.compensation_record_count,
        0
    ) AS compensation_record_count,

    COALESCE(
        cs.promotion_compensation_count,
        0
    ) AS promotion_compensation_count,

    lr.performance_rating,

    lr.goal_completion,

    lr.promotion_recommended,

    (
        prm.snapshot_date
        - lr.review_date
    ) AS days_since_review,

    COALESCE(
        rs.review_count,
        0
    ) AS review_count,

    ROUND(
        rs.average_performance_rating,
        2
    ) AS average_performance_rating,

    COALESCE(
        ts.completed_training_programs,
        0
    ) AS completed_training_programs,

    COALESCE(
        ts.failed_training_programs,
        0
    ) AS failed_training_programs,

    COALESCE(
        ts.in_progress_training_programs,
        0
    ) AS in_progress_training_programs,

    COALESCE(
        ts.completed_training_hours,
        0
    ) AS completed_training_hours,

    ROUND(
        ts.average_training_score,
        2
    ) AS average_training_score,

    COALESCE(
        es.prior_promotion_events,
        0
    ) AS prior_promotion_events,

    COALESCE(
        es.prior_transfer_events,
        0
    ) AS prior_transfer_events,

    COALESCE(
        es.prior_manager_change_events,
        0
    ) AS prior_manager_change_events,

    COALESCE(
        es.prior_leave_events,
        0
    ) AS prior_leave_events,

    COALESCE(
        es.prior_change_events,
        0
    ) AS prior_change_events,

    (
        prm.snapshot_date
        - es.last_change_event_date
    ) AS days_since_last_change_event,

    CASE
        WHEN
            pop.termination_date
                > prm.snapshot_date

            AND pop.termination_date
                <= prm.prediction_end_date

        THEN 1

        ELSE 0

    END AS attrition_next_12m


FROM population AS pop

CROSS JOIN params AS prm


LEFT JOIN hire_context AS hc
    ON pop.employee_id
       = hc.employee_id


LEFT JOIN initial_compensation AS ic
    ON pop.employee_id
       = ic.employee_id


LEFT JOIN latest_compensation AS lc
    ON pop.employee_id
       = lc.employee_id


LEFT JOIN compensation_summary AS cs
    ON pop.employee_id
       = cs.employee_id


LEFT JOIN latest_review AS lr
    ON pop.employee_id
       = lr.employee_id


LEFT JOIN review_summary AS rs
    ON pop.employee_id
       = rs.employee_id


LEFT JOIN training_summary AS ts
    ON pop.employee_id
       = ts.employee_id


LEFT JOIN event_summary AS es
    ON pop.employee_id
       = es.employee_id


ORDER BY
    pop.employee_id;