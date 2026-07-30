-- Version 2 PostgreSQL analytical source contract
-- =========================================================
-- These views expose only fields required by the leakage-safe
-- multi-snapshot retention builder. Direct names, reviewer IDs,
-- manager IDs, and unused operational fields remain outside the
-- Version 2 analytical query boundary.

CREATE OR REPLACE VIEW analytics_v2_employees AS
SELECT
    employee_id,
    hire_date,
    termination_date,
    department_id,
    location_id,
    job_role_id,
    employment_type,
    birth_year,
    education_level,
    organizational_level
FROM employees;

CREATE OR REPLACE VIEW analytics_v2_departments AS
SELECT
    department_id,
    department_name,
    department_group
FROM departments;

CREATE OR REPLACE VIEW analytics_v2_locations AS
SELECT
    location_id,
    city,
    region,
    location_type
FROM locations;

CREATE OR REPLACE VIEW analytics_v2_job_roles AS
SELECT
    job_role_id,
    job_title,
    job_family,
    job_level
FROM job_roles;

CREATE OR REPLACE VIEW analytics_v2_compensation_history AS
SELECT
    compensation_id,
    employee_id,
    effective_date,
    base_salary,
    bonus_target,
    equity_value,
    change_reason
FROM compensation_history;

CREATE OR REPLACE VIEW analytics_v2_performance_reviews AS
SELECT
    review_id,
    employee_id,
    review_date,
    performance_rating,
    goal_completion,
    promotion_recommended
FROM performance_reviews;

CREATE OR REPLACE VIEW analytics_v2_training_records AS
SELECT
    training_record_id,
    employee_id,
    completion_date,
    completion_status,
    training_hours,
    score
FROM training_records;

CREATE OR REPLACE VIEW analytics_v2_employee_events AS
SELECT
    event_id,
    employee_id,
    event_date,
    event_type,
    new_value,
    notes
FROM employee_events;
