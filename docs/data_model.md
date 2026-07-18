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
| candidate_id | Unique recruiting-candidate identifier |
| application_source | Channel through which the candidate entered recruiting |
| education_level | Candidate's highest education level |
| years_experience | Candidate's previous professional experience |
| candidate_location | Candidate's city and state |

### Candidate rules

- Candidate IDs must be complete, unique, and sequential.
- The first candidate ID is 200001.
- The candidate table contains 40,000 records.
- Application sources must use approved recruiting channels.
- Education levels must agree with the workforce education categories.
- Years of experience must be whole numbers between 0 and 30.
- Candidate locations must use a City, State format.
- The first 10,000 candidates are reserved for future accepted applications linked to employees.
- Candidate IDs and employee IDs remain separate identifiers.
- Candidates can later submit one or more applications.

### job_requisitions

Primary key: `requisition_id`

Foreign keys:

- `job_role_id` references `job_roles.job_role_id`
- `department_id` references `departments.department_id`
- `location_id` references `locations.location_id`
- `recruiter_id` references `employees.employee_id`

| Column | Description |
|---|---|
| requisition_id | Unique job-opening identifier |
| job_role_id | Role being recruited |
| department_id | Hiring department |
| location_id | Hiring location |
| open_date | Date the requisition opened |
| close_date | Date the requisition was filled or cancelled; blank when open |
| target_headcount | Number of employees requested |
| recruiter_id | Employee responsible for recruiting |
| requisition_status | Open, filled, or cancelled |

### Job requisition rules

- Requisition IDs must be complete, unique, and sequential.
- Job-role, department, location, and recruiter IDs must reference valid records.
- Target headcount must be a positive whole number.
- Requisition status must be Open, Filled, or Cancelled.
- Open requisitions must have blank close dates.
- Filled and cancelled requisitions must have close dates.
- Close dates cannot occur before open dates.
- Requisitions cannot open or close after the analysis date.
- Recruiters must be employed throughout their assigned requisition period.
- Role, department, and location combinations must exist in the employee population.
- Historical filled requisitions are grouped by role, department, location, and hiring quarter.
- Filled requisition target headcount must agree with employee hiring counts.
- Filled requisition target headcount must total 10,000.
- The current dataset contains 120 open requisitions and 250 cancelled requisitions.

### applications

Primary key: `application_id`

Foreign keys:

- `candidate_id` references `candidates.candidate_id`
- `requisition_id` references `job_requisitions.requisition_id`
- `employee_id` references `employees.employee_id` when the application is hired

| Column | Description |
|---|---|
| application_id | Unique job-application identifier |
| candidate_id | Candidate who submitted the application |
| requisition_id | Job requisition receiving the application |
| application_date | Date the application was submitted |
| application_status | Hired, rejected, withdrawn, offer declined, in process, or position cancelled |
| interview_score | Optional interview score from 0 to 100 |
| offer_date | Date an offer was made; blank when no offer occurred |
| decision_date | Date the application reached its current or final outcome |
| employee_id | Employee created from a hired application; blank for non-hired applications |

### Application rules

- Application IDs must be complete, unique, and sequential.
- Candidate, requisition, and populated employee IDs must reference valid records.
- Every candidate must have at least one application.
- A candidate can apply to a requisition only once.
- Applications cannot occur before the requisition opens or after it closes.
- Final applications must have decision dates.
- In-process applications must have blank decision dates and belong to open requisitions.
- Hired and offer-declined applications must have offer dates.
- Other application statuses must have blank offer dates.
- Hired applications must belong to filled requisitions.
- Position-cancelled applications must belong to cancelled requisitions.
- Interview scores must remain between 0 and 100.
- Hired applications must have interview scores of at least 75.
- Offer-declined applications must have interview scores of at least 70.
- Hired applications must contain employee IDs.
- Non-hired applications must have blank employee IDs.
- Every employee must be connected to exactly one hired application.
- Hired application counts must match filled requisition target headcounts.

### compensation_history

Primary key: `compensation_id`

Foreign key:

- `employee_id` references `employees.employee_id`

| Column | Description |
|---|---|
| compensation_id | Unique compensation record |
| employee_id | Employee |
| effective_date | Date the compensation became effective |
| base_salary | Annualized base pay in USD |
| bonus_target | Target bonus percentage; 10.0 means 10% |
| equity_value | Estimated annual equity value in USD |
| change_reason | Hire, promotion, annual review, or adjustment |

### Compensation history rules

- Every employee must have exactly one hire compensation record.
- The hire compensation date must match the employee's hire date.
- Compensation records cannot occur before hire or after employment ends.
- Compensation records for one employee must use unique effective dates.
- Base salary cannot decrease in the current synthetic model.
- Base salary must remain within the allowed job-role salary range.
- Bonus target is stored as a percentage, where 10.0 means 10%.
- Base salary is annualized for both salaried and hourly employees.

### performance_reviews

Primary key: `review_id`

Foreign keys:

- `employee_id` references `employees.employee_id`
- `reviewer_id` references `employees.employee_id`

| Column | Description |
|---|---|
| review_id | Unique performance-review identifier |
| employee_id | Employee being reviewed |
| review_date | Date the review was completed |
| review_period | Annual review label such as 2025 Annual |
| performance_rating | Rating from 1.0 to 5.0 |
| goal_completion | Goal-completion percentage from 0 to 120 |
| promotion_recommended | Whether promotion was recommended |
| reviewer_id | Current manager who completed the review |

### Performance review rules

- Review IDs must be complete and unique.
- Employees and reviewers must reference valid employee records.
- Employees cannot review themselves.
- Reviewers must be active department heads, senior managers, or team managers.
- Employees and reviewers must belong to the same department.
- The reviewer must match the employee's current manager in the present synthetic model.
- Reviews cannot occur before hire or after employment ends.
- Reviews occur only after at least one complete year of employment.
- An employee can have only one review in each annual review period.
- Performance ratings must remain between 1.0 and 5.0.
- Goal completion must remain between 0% and 120%.
- Department heads do not receive reviews until an executive level is added.

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
| training_record_id | Unique employee-training record identifier |
| employee_id | Employee assigned to the training program |
| program_id | Training program assigned to the employee |
| start_date | Date the employee began the program |
| completion_date | Date the program was completed or failed; blank when incomplete |
| completion_status | Completed, incomplete, or failed |
| training_hours | Number of training hours recorded |
| score | Optional assessment score from 0 to 100 |

### Training record rules

- Training-record IDs must be complete and unique.
- Employee and program IDs must reference valid records.
- An employee can have only one record for each training program in the current model.
- Training cannot begin before hire or after employment ends.
- Completed and failed programs must have completion dates.
- Incomplete programs must not have completion dates.
- Completion dates cannot occur before start dates or after employment ends.
- Completed programs must meet or exceed the program's required hours.
- Failed and incomplete programs must remain below the required hours.
- Completed assessment scores must remain between 70 and 100.
- Failed assessment scores must remain below 70.
- Incomplete programs do not receive assessment scores.
- Every employee must have exactly one onboarding record.
- Managers must receive leadership training.
- Manufacturing, supply-chain, and hourly employees must receive safety training.

### employee_events

Primary key: `event_id`

Foreign key:

- `employee_id` references `employees.employee_id`

| Column | Description |
|---|---|
| event_id | Unique employee-event identifier |
| employee_id | Employee affected by the event |
| event_date | Date the event occurred |
| event_type | Hire, promotion, transfer, manager change, leave, or termination |
| old_value | Value before the event; blank for hire events |
| new_value | Value after the event |
| notes | Explanation of the event and value type |

### Employee event rules

- Event IDs must be complete and unique.
- Employee IDs must reference valid employee records.
- Events cannot occur before hire or after employment ends.
- Employee, event-date, and event-type combinations must be unique.
- Every employee must have exactly one hire event.
- Hire-event dates must match employee hire dates.
- Terminated employees must have exactly one termination event.
- Active employees cannot have termination events.
- Termination-event dates must match employee termination dates.
- Promotion events must agree with promotion records in compensation history.
- Transfer events use old and new location IDs.
- Manager-change events use old and new manager IDs.
- Managers referenced in manager-change events must be valid active employees in the appropriate department and management level.
- Each leave period contains one leave-start event and one return event.

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