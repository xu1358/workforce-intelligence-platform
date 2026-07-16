# Workforce Intelligence Platform Data Model

## 1. Purpose

This document defines the first version of the data model for the Workforce Intelligence and Retention Decision Platform.

The project uses synthetic information from six fictional source systems:

1. Human Resources Information System
2. Recruiting System
3. Compensation System
4. Performance Review System
5. Learning and Training System
6. Employee Event System

## 2. Basic database concepts

A table stores information about one type of object or event.

A row represents one record.

A column represents one attribute.

A primary key uniquely identifies a row.

A foreign key connects a row to another table.

## 3. Planned tables

### Human Resources Information System

- employees
- departments
- locations
- job_roles

### Recruiting System

- candidates
- job_requisitions
- applications

### Compensation System

- compensation_history

### Performance Review System

- performance_reviews

### Learning and Training System

- training_programs
- training_records

### Employee Event System

- employee_events

## 4. Table definitions

### departments

Primary key: `department_id`

| Column | Description |
|---|---|
| department_id | Unique department identifier |
| department_name | Department name |
| department_group | Broader organizational group |
| cost_center | Financial tracking code |

### locations

Primary key: `location_id`

| Column | Description |
|---|---|
| location_id | Unique location identifier |
| city | City |
| state | State |
| region | Geographic region |
| location_type | Office, factory, or distribution center |

### job_roles

Primary key: `job_role_id`

| Column | Description |
|---|---|
| job_role_id | Unique job-role identifier |
| job_title | Position title |
| job_family | Broader job category |
| job_level | Seniority level |
| salary_band_min | Minimum expected salary |
| salary_band_max | Maximum expected salary |

### employees

Primary key: `employee_id`

Foreign keys:

- `department_id` references `departments.department_id`
- `location_id` references `locations.location_id`
- `job_role_id` references `job_roles.job_role_id`
- `manager_id` references `employees.employee_id`

| Column | Description |
|---|---|
| employee_id | Unique employee identifier |
| first_name | Synthetic first name |
| last_name | Synthetic last name |
| hire_date | Employment start date |
| termination_date | Employment end date |
| employment_status | Active or terminated |
| termination_type | Voluntary, involuntary, or not applicable |
| department_id | Employee department |
| location_id | Employee work location |
| job_role_id | Employee job role |
| manager_id | Employee identifier of the manager |
| employment_type | Salaried or hourly |
| birth_year | Synthetic birth year |
| education_level | Highest education level |
| organizational_level | Department head, senior manager, team manager, or individual contributor |

### Employee hierarchy rules

- Every department must have exactly one department head.
- Department heads do not have a manager in the current simplified model.
- Senior managers must report to department heads.
- Team managers must report to senior managers or department heads.
- Individual contributors must report to team managers or department heads.
- Employees and their managers must belong to the same department.
- All referenced managers must be active.
- Department heads may have no more than 20 direct reports.
- Senior managers and team managers may have no more than 12 direct reports.

### candidates

Primary key: `candidate_id`

| Column | Description |
|---|---|
| candidate_id | Unique candidate identifier |
| application_source | Recruiting source |
| education_level | Highest education level |
| years_experience | Previous experience |
| candidate_location | Candidate location |

### job_requisitions

Primary key: `requisition_id`

Foreign keys:

- `job_role_id` references `job_roles.job_role_id`
- `department_id` references `departments.department_id`
- `location_id` references `locations.location_id`
- `recruiter_id` references `employees.employee_id`

| Column | Description |
|---|---|
| requisition_id | Unique job opening |
| job_role_id | Role being recruited |
| department_id | Hiring department |
| location_id | Hiring location |
| open_date | Date the requisition opened |
| close_date | Date the requisition closed |
| target_headcount | Number of employees requested |
| recruiter_id | Employee responsible for recruiting |
| requisition_status | Open, filled, or cancelled |

### applications

Primary key: `application_id`

Foreign keys:

- `candidate_id` references `candidates.candidate_id`
- `requisition_id` references `job_requisitions.requisition_id`
- `hire_employee_id` references `employees.employee_id`

| Column | Description |
|---|---|
| application_id | Unique application identifier |
| candidate_id | Candidate |
| requisition_id | Job requisition |
| application_date | Application date |
| screen_date | Resume-screen date |
| interview_date | Interview date |
| offer_date | Offer date |
| offer_status | Accepted, rejected, declined, or no offer |
| hire_employee_id | Employee identifier created after hiring |

### compensation_history

Primary key: `compensation_id`

Foreign key:

- `employee_id` references `employees.employee_id`

| Column | Description |
|---|---|
| compensation_id | Unique compensation record |
| employee_id | Employee |
| effective_date | Date the compensation became effective |
| base_salary | Annual base salary |
| bonus_target | Target bonus percentage |
| equity_value | Estimated annual equity value |
| change_reason | Hire, promotion, annual review, or adjustment |

### performance_reviews

Primary key: `review_id`

Foreign keys:

- `employee_id` references `employees.employee_id`
- `reviewer_id` references `employees.employee_id`

| Column | Description |
|---|---|
| review_id | Unique review identifier |
| employee_id | Employee being reviewed |
| review_date | Review date |
| review_period | Period evaluated |
| performance_rating | Numeric rating |
| goal_completion | Percentage of goals completed |
| promotion_recommended | Whether promotion was recommended |
| reviewer_id | Employee who completed the review |

### training_programs

Primary key: `program_id`

| Column | Description |
|---|---|
| program_id | Unique training-program identifier |
| program_name | Program name |
| program_category | Technical, leadership, safety, or onboarding |
| required_hours | Expected training hours |
| mandatory | Whether the program is required |

### training_records

Primary key: `training_record_id`

Foreign keys:

- `employee_id` references `employees.employee_id`
- `program_id` references `training_programs.program_id`

| Column | Description |
|---|---|
| training_record_id | Unique training record |
| employee_id | Employee |
| program_id | Training program |
| start_date | Training start date |
| completion_date | Completion date |
| completion_status | Completed, incomplete, or failed |
| training_hours | Completed hours |
| score | Optional assessment score |

### employee_events

Primary key: `event_id`

Foreign key:

- `employee_id` references `employees.employee_id`

| Column | Description |
|---|---|
| event_id | Unique event identifier |
| employee_id | Employee affected |
| event_date | Event date |
| event_type | Promotion, transfer, manager change, leave, or termination |
| old_value | Previous value |
| new_value | New value |
| notes | Optional event explanation |

## 5. Main relationships

- One department has many employees.
- One location has many employees.
- One job role has many employees.
- One employee may manage many employees.
- One employee can have many compensation records.
- One employee can have many performance reviews.
- One employee can complete many training programs.
- One employee can have many employee events.
- One candidate can submit many applications.
- One requisition can receive many applications.
- One training program can have many employee completions.

## 6. Entity relationship diagram

```mermaid
erDiagram
    DEPARTMENTS ||--o{ EMPLOYEES : contains
    LOCATIONS ||--o{ EMPLOYEES : hosts
    JOB_ROLES ||--o{ EMPLOYEES : assigns

    EMPLOYEES ||--o{ EMPLOYEES : manages
    EMPLOYEES ||--o{ COMPENSATION_HISTORY : receives
    EMPLOYEES ||--o{ PERFORMANCE_REVIEWS : receives
    EMPLOYEES ||--o{ TRAINING_RECORDS : completes
    EMPLOYEES ||--o{ EMPLOYEE_EVENTS : experiences

    JOB_ROLES ||--o{ JOB_REQUISITIONS : requested_for
    DEPARTMENTS ||--o{ JOB_REQUISITIONS : opens
    LOCATIONS ||--o{ JOB_REQUISITIONS : located_at

    CANDIDATES ||--o{ APPLICATIONS : submits
    JOB_REQUISITIONS ||--o{ APPLICATIONS : receives

    TRAINING_PROGRAMS ||--o{ TRAINING_RECORDS : includes