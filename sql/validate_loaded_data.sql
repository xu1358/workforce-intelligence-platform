-- =========================================================
-- Workforce Intelligence Platform
-- Loaded Data Validation Queries
-- =========================================================


-- 1. Employee count
SELECT
    COUNT(*) AS employee_count
FROM employees;


-- 2. Employee status counts
SELECT
    employment_status,
    COUNT(*) AS employee_count
FROM employees
GROUP BY employment_status
ORDER BY employment_status;


-- 3. Employees by department
SELECT
    d.department_name,
    COUNT(*) AS employee_count
FROM employees AS e
JOIN departments AS d
    ON e.department_id
       = d.department_id
GROUP BY
    d.department_name
ORDER BY
    employee_count DESC;


-- 4. Application status counts
SELECT
    application_status,
    COUNT(*) AS application_count
FROM applications
GROUP BY
    application_status
ORDER BY
    application_count DESC;


-- 5. Hired applications
SELECT
    COUNT(*) AS hired_applications
FROM applications
WHERE
    application_status = 'Hired';


-- 6. Employees connected to hired applications
SELECT
    COUNT(
        DISTINCT employee_id
    ) AS hired_employees
FROM applications
WHERE
    application_status = 'Hired';


-- 7. Filled requisition headcount
SELECT
    SUM(
        target_headcount
    ) AS filled_target_headcount
FROM job_requisitions
WHERE
    requisition_status = 'Filled';


-- 8. Example employee-department join
SELECT
    e.employee_id,
    e.first_name,
    e.last_name,
    d.department_name,
    l.city,
    jr.job_title
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
ORDER BY
    e.employee_id
LIMIT 20;