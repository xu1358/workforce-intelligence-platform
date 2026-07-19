-- =========================================================
-- Workforce Intelligence Platform
-- Workforce Analytics
-- =========================================================


-- =========================================================
-- 1. Active headcount by department
-- =========================================================

SELECT
    d.department_name,
    COUNT(*) FILTER (
        WHERE e.employment_status = 'Active'
    ) AS active_headcount,
    COUNT(*) FILTER (
        WHERE e.employment_status = 'Terminated'
    ) AS terminated_employee_count,
    COUNT(*) AS total_employee_records
FROM employees AS e
JOIN departments AS d
    ON e.department_id = d.department_id
GROUP BY
    d.department_name
ORDER BY
    active_headcount DESC;

-- =========================================================
-- 2. Active headcount by location
-- =========================================================

SELECT
    l.city,
    l.state,
    l.region,
    l.location_type,
    COUNT(*) AS active_headcount
FROM employees AS e
JOIN locations AS l
    ON e.location_id = l.location_id
WHERE
    e.employment_status = 'Active'
GROUP BY
    l.city,
    l.state,
    l.region,
    l.location_type
ORDER BY
    active_headcount DESC;

-- =========================================================
-- 3. Active workforce by job family
-- =========================================================

SELECT
    jr.job_family,
    COUNT(*) AS active_headcount
FROM employees AS e
JOIN job_roles AS jr
    ON e.job_role_id = jr.job_role_id
WHERE
    e.employment_status = 'Active'
GROUP BY
    jr.job_family
ORDER BY
    active_headcount DESC;

-- =========================================================
-- 4. 2025 annual turnover rate by department
-- =========================================================

WITH department_headcount AS (
    SELECT
        d.department_name,

        COUNT(*) FILTER (
            WHERE
                e.hire_date <= DATE '2025-01-01'
                AND (
                    e.termination_date IS NULL
                    OR e.termination_date > DATE '2025-01-01'
                )
        ) AS starting_headcount,

        COUNT(*) FILTER (
            WHERE
                e.hire_date <= DATE '2025-12-31'
                AND (
                    e.termination_date IS NULL
                    OR e.termination_date > DATE '2025-12-31'
                )
        ) AS ending_headcount,

        COUNT(*) FILTER (
            WHERE
                e.termination_date
                BETWEEN DATE '2025-01-01'
                AND DATE '2025-12-31'
        ) AS terminations_2025

    FROM employees AS e
    JOIN departments AS d
        ON e.department_id = d.department_id
    GROUP BY
        d.department_name
)

SELECT
    department_name,
    starting_headcount,
    ending_headcount,
    terminations_2025,

    ROUND(
        (
            starting_headcount
            + ending_headcount
        ) / 2.0,
        2
    ) AS average_headcount,

    ROUND(
        100.0
        * terminations_2025
        / NULLIF(
            (
                starting_headcount
                + ending_headcount
            ) / 2.0,
            0
        ),
        2
    ) AS turnover_rate_percent

FROM department_headcount
ORDER BY
    turnover_rate_percent DESC;

-- =========================================================
-- 5. Recruiting funnel by application source
-- =========================================================

SELECT
    c.application_source,

    COUNT(
        DISTINCT c.candidate_id
    ) AS candidate_count,

    COUNT(
        a.application_id
    ) AS application_count,

    COUNT(
        a.application_id
    ) FILTER (
        WHERE a.interview_score IS NOT NULL
    ) AS interviewed_applications,

    COUNT(
        a.application_id
    ) FILTER (
        WHERE a.offer_date IS NOT NULL
    ) AS offers,

    COUNT(
        a.application_id
    ) FILTER (
        WHERE a.application_status = 'Hired'
    ) AS hires,

    ROUND(
        100.0
        * COUNT(
            a.application_id
        ) FILTER (
            WHERE a.application_status = 'Hired'
        )
        / NULLIF(
            COUNT(a.application_id),
            0
        ),
        2
    ) AS application_to_hire_rate_percent

FROM candidates AS c
JOIN applications AS a
    ON c.candidate_id = a.candidate_id

GROUP BY
    c.application_source

ORDER BY
    hires DESC;

-- =========================================================
-- 6. Requisition duration by status
-- =========================================================

SELECT
    requisition_status,

    COUNT(*) AS requisition_count,

    ROUND(
        AVG(
            CASE
                WHEN close_date IS NOT NULL
                    THEN close_date - open_date
                ELSE
                    DATE '2026-06-30'
                    - open_date
            END
        ),
        2
    ) AS average_days_open,

    MIN(
        CASE
            WHEN close_date IS NOT NULL
                THEN close_date - open_date
            ELSE
                DATE '2026-06-30'
                - open_date
        END
    ) AS minimum_days_open,

    MAX(
        CASE
            WHEN close_date IS NOT NULL
                THEN close_date - open_date
            ELSE
                DATE '2026-06-30'
                - open_date
        END
    ) AS maximum_days_open

FROM job_requisitions

GROUP BY
    requisition_status

ORDER BY
    requisition_status;

-- =========================================================
-- 7. Current compensation by department
-- =========================================================

WITH latest_compensation AS (
    SELECT DISTINCT ON (
        employee_id
    )
        employee_id,
        effective_date,
        base_salary,
        bonus_target,
        equity_value

    FROM compensation_history

    WHERE
        effective_date
        <= DATE '2026-06-30'

    ORDER BY
        employee_id,
        effective_date DESC,
        compensation_id DESC
)

SELECT
    d.department_name,

    COUNT(*) AS active_employee_count,

    ROUND(
        AVG(
            lc.base_salary
        ),
        2
    ) AS average_base_salary,

    ROUND(
        MIN(
            lc.base_salary
        ),
        2
    ) AS minimum_base_salary,

    ROUND(
        MAX(
            lc.base_salary
        ),
        2
    ) AS maximum_base_salary,

    ROUND(
        AVG(
            lc.bonus_target
        ),
        2
    ) AS average_bonus_target,

    ROUND(
        AVG(
            lc.equity_value
        ),
        2
    ) AS average_equity_value

FROM employees AS e

JOIN departments AS d
    ON e.department_id
       = d.department_id

JOIN latest_compensation AS lc
    ON e.employee_id
       = lc.employee_id

WHERE
    e.employment_status = 'Active'

GROUP BY
    d.department_name

ORDER BY
    average_base_salary DESC;

-- =========================================================
-- 8. Latest employee performance by department
-- =========================================================

WITH latest_review AS (
    SELECT DISTINCT ON (
        employee_id
    )
        employee_id,
        review_date,
        performance_rating,
        goal_completion,
        promotion_recommended

    FROM performance_reviews

    ORDER BY
        employee_id,
        review_date DESC,
        review_id DESC
)

SELECT
    d.department_name,

    COUNT(
        lr.employee_id
    ) AS employees_with_reviews,

    ROUND(
        AVG(
            lr.performance_rating
        ),
        2
    ) AS average_performance_rating,

    ROUND(
        AVG(
            lr.goal_completion
        ),
        2
    ) AS average_goal_completion,

    ROUND(
        100.0
        * COUNT(*) FILTER (
            WHERE
                lr.promotion_recommended = TRUE
        )
        / NULLIF(
            COUNT(lr.employee_id),
            0
        ),
        2
    ) AS promotion_recommendation_rate_percent

FROM employees AS e

JOIN departments AS d
    ON e.department_id
       = d.department_id

JOIN latest_review AS lr
    ON e.employee_id
       = lr.employee_id

WHERE
    e.employment_status = 'Active'

GROUP BY
    d.department_name

ORDER BY
    average_performance_rating DESC;

-- =========================================================
-- 9. Training activity by department
-- =========================================================

WITH training_by_employee AS (
    SELECT
        employee_id,

        COUNT(*) FILTER (
            WHERE
                completion_status = 'Completed'
        ) AS completed_programs,

        COALESCE(
            SUM(
                training_hours
            ) FILTER (
                WHERE
                    completion_status = 'Completed'
            ),
            0
        ) AS completed_training_hours

    FROM training_records

    GROUP BY
        employee_id
)

SELECT
    d.department_name,

    COUNT(
        e.employee_id
    ) AS active_employee_count,

    ROUND(
        AVG(
            COALESCE(
                t.completed_programs,
                0
            )
        ),
        2
    ) AS average_completed_programs,

    ROUND(
        AVG(
            COALESCE(
                t.completed_training_hours,
                0
            )
        ),
        2
    ) AS average_completed_training_hours

FROM employees AS e

JOIN departments AS d
    ON e.department_id
       = d.department_id

LEFT JOIN training_by_employee AS t
    ON e.employee_id
       = t.employee_id

WHERE
    e.employment_status = 'Active'

GROUP BY
    d.department_name

ORDER BY
    average_completed_training_hours DESC;

-- =========================================================
-- 10. Combined active-employee analytics dataset
-- =========================================================

WITH latest_compensation AS (
    SELECT DISTINCT ON (
        employee_id
    )
        employee_id,
        base_salary,
        bonus_target,
        equity_value

    FROM compensation_history

    ORDER BY
        employee_id,
        effective_date DESC,
        compensation_id DESC
),

latest_review AS (
    SELECT DISTINCT ON (
        employee_id
    )
        employee_id,
        performance_rating,
        goal_completion,
        promotion_recommended

    FROM performance_reviews

    ORDER BY
        employee_id,
        review_date DESC,
        review_id DESC
),

training_summary AS (
    SELECT
        employee_id,

        COUNT(*) FILTER (
            WHERE
                completion_status = 'Completed'
        ) AS completed_training_programs,

        COALESCE(
            SUM(
                training_hours
            ) FILTER (
                WHERE
                    completion_status = 'Completed'
            ),
            0
        ) AS completed_training_hours

    FROM training_records

    GROUP BY
        employee_id
)

SELECT
    e.employee_id,
    e.hire_date,
    e.employment_type,
    e.education_level,
    e.organizational_level,

    d.department_name,

    l.city,
    l.region,

    jr.job_title,
    jr.job_family,
    jr.job_level,

    lc.base_salary,
    lc.bonus_target,
    lc.equity_value,

    lr.performance_rating,
    lr.goal_completion,
    lr.promotion_recommended,

    COALESCE(
        ts.completed_training_programs,
        0
    ) AS completed_training_programs,

    COALESCE(
        ts.completed_training_hours,
        0
    ) AS completed_training_hours

FROM employees AS e

JOIN departments AS d
    ON e.department_id
       = d.department_id

JOIN locations AS l
    ON e.location_id
       = l.location_id

JOIN job_roles AS jr
    ON e.job_role_id
       = jr.job_role_id

LEFT JOIN latest_compensation AS lc
    ON e.employee_id
       = lc.employee_id

LEFT JOIN latest_review AS lr
    ON e.employee_id
       = lr.employee_id

LEFT JOIN training_summary AS ts
    ON e.employee_id
       = ts.employee_id

WHERE
    e.employment_status = 'Active'

ORDER BY
    e.employee_id;