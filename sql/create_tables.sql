-- =========================================================
-- Workforce Intelligence and Retention Decision Platform
-- PostgreSQL Table Schema
-- =========================================================


-- =========================================================
-- 1. Reference tables
-- =========================================================

CREATE TABLE IF NOT EXISTS departments (
    department_id INTEGER PRIMARY KEY,
    department_name TEXT NOT NULL UNIQUE,
    department_group TEXT NOT NULL,
    cost_center TEXT NOT NULL UNIQUE
);


CREATE TABLE IF NOT EXISTS locations (
    location_id INTEGER PRIMARY KEY,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    region TEXT NOT NULL,
    location_type TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS job_roles (
    job_role_id INTEGER PRIMARY KEY,
    job_title TEXT NOT NULL,
    job_family TEXT NOT NULL,
    job_level TEXT NOT NULL,
    salary_band_min NUMERIC(12, 2) NOT NULL,
    salary_band_max NUMERIC(12, 2) NOT NULL,

    CONSTRAINT chk_job_roles_salary_min
        CHECK (
            salary_band_min >= 0
        ),

    CONSTRAINT chk_job_roles_salary_range
        CHECK (
            salary_band_max
            >= salary_band_min
        )
);


CREATE TABLE IF NOT EXISTS training_programs (
    program_id INTEGER PRIMARY KEY,
    program_name TEXT NOT NULL UNIQUE,
    program_category TEXT NOT NULL,
    required_hours NUMERIC(8, 2) NOT NULL,
    mandatory BOOLEAN NOT NULL,

    CONSTRAINT chk_training_programs_hours
        CHECK (
            required_hours > 0
        )
);


-- =========================================================
-- 2. Employee table
-- =========================================================

CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    hire_date DATE NOT NULL,
    termination_date DATE,
    employment_status TEXT NOT NULL,
    termination_type TEXT,
    department_id INTEGER NOT NULL,
    location_id INTEGER NOT NULL,
    job_role_id INTEGER NOT NULL,
    manager_id INTEGER,
    employment_type TEXT NOT NULL,
    birth_year INTEGER NOT NULL,
    education_level TEXT NOT NULL,
    hierarchy_level TEXT NOT NULL,

    CONSTRAINT fk_employees_department
        FOREIGN KEY (
            department_id
        )
        REFERENCES departments (
            department_id
        ),

    CONSTRAINT fk_employees_location
        FOREIGN KEY (
            location_id
        )
        REFERENCES locations (
            location_id
        ),

    CONSTRAINT fk_employees_job_role
        FOREIGN KEY (
            job_role_id
        )
        REFERENCES job_roles (
            job_role_id
        ),

    CONSTRAINT fk_employees_manager
        FOREIGN KEY (
            manager_id
        )
        REFERENCES employees (
            employee_id
        )
        DEFERRABLE
        INITIALLY DEFERRED,

    CONSTRAINT chk_employees_status
        CHECK (
            employment_status
            IN (
                'Active',
                'Terminated'
            )
        ),

    CONSTRAINT chk_employees_termination_date
        CHECK (
            termination_date IS NULL
            OR termination_date >= hire_date
        ),

    CONSTRAINT chk_employees_status_date
        CHECK (
            (
                employment_status = 'Active'
                AND termination_date IS NULL
            )
            OR
            (
                employment_status = 'Terminated'
                AND termination_date IS NOT NULL
            )
        ),

    CONSTRAINT chk_employees_manager
        CHECK (
            manager_id IS NULL
            OR manager_id <> employee_id
        ),

    CONSTRAINT chk_employees_hierarchy
        CHECK (
            hierarchy_level
            IN (
                'Department Head',
                'Senior Manager',
                'Team Manager',
                'Individual Contributor'
            )
        )
);


-- =========================================================
-- 3. Recruiting tables
-- =========================================================

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id INTEGER PRIMARY KEY,
    application_source TEXT NOT NULL,
    education_level TEXT NOT NULL,
    years_experience NUMERIC(5, 1) NOT NULL,
    candidate_location TEXT NOT NULL,

    CONSTRAINT chk_candidates_experience
        CHECK (
            years_experience >= 0
        )
);


CREATE TABLE IF NOT EXISTS job_requisitions (
    requisition_id INTEGER PRIMARY KEY,
    job_role_id INTEGER NOT NULL,
    department_id INTEGER NOT NULL,
    location_id INTEGER NOT NULL,
    open_date DATE NOT NULL,
    close_date DATE,
    target_headcount INTEGER NOT NULL,
    recruiter_id INTEGER NOT NULL,
    requisition_status TEXT NOT NULL,

    CONSTRAINT fk_requisitions_job_role
        FOREIGN KEY (
            job_role_id
        )
        REFERENCES job_roles (
            job_role_id
        ),

    CONSTRAINT fk_requisitions_department
        FOREIGN KEY (
            department_id
        )
        REFERENCES departments (
            department_id
        ),

    CONSTRAINT fk_requisitions_location
        FOREIGN KEY (
            location_id
        )
        REFERENCES locations (
            location_id
        ),

    CONSTRAINT fk_requisitions_recruiter
        FOREIGN KEY (
            recruiter_id
        )
        REFERENCES employees (
            employee_id
        ),

    CONSTRAINT chk_requisitions_headcount
        CHECK (
            target_headcount > 0
        ),

    CONSTRAINT chk_requisitions_status
        CHECK (
            requisition_status
            IN (
                'Open',
                'Filled',
                'Cancelled'
            )
        ),

    CONSTRAINT chk_requisitions_dates
        CHECK (
            close_date IS NULL
            OR close_date >= open_date
        ),

    CONSTRAINT chk_requisitions_status_dates
        CHECK (
            (
                requisition_status = 'Open'
                AND close_date IS NULL
            )
            OR
            (
                requisition_status
                IN (
                    'Filled',
                    'Cancelled'
                )
                AND close_date IS NOT NULL
            )
        )
);


CREATE TABLE IF NOT EXISTS applications (
    application_id INTEGER PRIMARY KEY,
    candidate_id INTEGER NOT NULL,
    requisition_id INTEGER NOT NULL,
    application_date DATE NOT NULL,
    application_status TEXT NOT NULL,
    interview_score NUMERIC(5, 2),
    offer_date DATE,
    decision_date DATE,
    employee_id INTEGER,

    CONSTRAINT fk_applications_candidate
        FOREIGN KEY (
            candidate_id
        )
        REFERENCES candidates (
            candidate_id
        ),

    CONSTRAINT fk_applications_requisition
        FOREIGN KEY (
            requisition_id
        )
        REFERENCES job_requisitions (
            requisition_id
        ),

    CONSTRAINT fk_applications_employee
        FOREIGN KEY (
            employee_id
        )
        REFERENCES employees (
            employee_id
        ),

    CONSTRAINT uq_applications_candidate_requisition
        UNIQUE (
            candidate_id,
            requisition_id
        ),

    CONSTRAINT uq_applications_employee
        UNIQUE (
            employee_id
        ),

    CONSTRAINT chk_applications_status
        CHECK (
            application_status
            IN (
                'Hired',
                'Rejected',
                'Withdrawn',
                'Offer Declined',
                'In Process',
                'Position Cancelled'
            )
        ),

    CONSTRAINT chk_applications_score
        CHECK (
            interview_score IS NULL
            OR (
                interview_score >= 0
                AND interview_score <= 100
            )
        ),

    CONSTRAINT chk_applications_offer_date
        CHECK (
            offer_date IS NULL
            OR offer_date >= application_date
        ),

    CONSTRAINT chk_applications_decision_date
        CHECK (
            decision_date IS NULL
            OR decision_date >= application_date
        ),

    CONSTRAINT chk_applications_final_decision
        CHECK (
            (
                application_status = 'In Process'
                AND decision_date IS NULL
            )
            OR
            (
                application_status <> 'In Process'
                AND decision_date IS NOT NULL
            )
        ),

    CONSTRAINT chk_applications_offer
        CHECK (
            (
                application_status
                IN (
                    'Hired',
                    'Offer Declined'
                )
                AND offer_date IS NOT NULL
            )
            OR
            (
                application_status
                NOT IN (
                    'Hired',
                    'Offer Declined'
                )
                AND offer_date IS NULL
            )
        ),

    CONSTRAINT chk_applications_employee
        CHECK (
            (
                application_status = 'Hired'
                AND employee_id IS NOT NULL
            )
            OR
            (
                application_status <> 'Hired'
                AND employee_id IS NULL
            )
        )
);


-- =========================================================
-- 4. Compensation table
-- =========================================================

CREATE TABLE IF NOT EXISTS compensation_history (
    compensation_id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL,
    effective_date DATE NOT NULL,
    base_salary NUMERIC(12, 2) NOT NULL,
    bonus_target NUMERIC(6, 2) NOT NULL,
    equity_value NUMERIC(12, 2) NOT NULL,
    change_reason TEXT NOT NULL,

    CONSTRAINT fk_compensation_employee
        FOREIGN KEY (
            employee_id
        )
        REFERENCES employees (
            employee_id
        ),

    CONSTRAINT chk_compensation_salary
        CHECK (
            base_salary > 0
        ),

    CONSTRAINT chk_compensation_bonus
        CHECK (
            bonus_target >= 0
        ),

    CONSTRAINT chk_compensation_equity
        CHECK (
            equity_value >= 0
        )
);


-- =========================================================
-- 5. Performance-review table
-- =========================================================

CREATE TABLE IF NOT EXISTS performance_reviews (
    review_id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL,
    review_date DATE NOT NULL,
    review_period TEXT NOT NULL,
    performance_rating NUMERIC(3, 2) NOT NULL,
    goal_completion NUMERIC(6, 2) NOT NULL,
    promotion_recommended BOOLEAN NOT NULL,
    reviewer_id INTEGER NOT NULL,

    CONSTRAINT fk_reviews_employee
        FOREIGN KEY (
            employee_id
        )
        REFERENCES employees (
            employee_id
        ),

    CONSTRAINT fk_reviews_reviewer
        FOREIGN KEY (
            reviewer_id
        )
        REFERENCES employees (
            employee_id
        ),

    CONSTRAINT chk_reviews_rating
        CHECK (
            performance_rating >= 1
            AND performance_rating <= 5
        ),

    CONSTRAINT chk_reviews_goal_completion
        CHECK (
            goal_completion >= 0
            AND goal_completion <= 200
        ),

    CONSTRAINT chk_reviews_reviewer
        CHECK (
            reviewer_id <> employee_id
        )
);


-- =========================================================
-- 6. Training-record table
-- =========================================================

CREATE TABLE IF NOT EXISTS training_records (
    training_record_id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL,
    program_id INTEGER NOT NULL,
    start_date DATE NOT NULL,
    completion_date DATE,
    completion_status TEXT NOT NULL,
    training_hours NUMERIC(8, 2) NOT NULL,
    score NUMERIC(5, 2),

    CONSTRAINT fk_training_records_employee
        FOREIGN KEY (
            employee_id
        )
        REFERENCES employees (
            employee_id
        ),

    CONSTRAINT fk_training_records_program
        FOREIGN KEY (
            program_id
        )
        REFERENCES training_programs (
            program_id
        ),

    CONSTRAINT chk_training_records_status
        CHECK (
            completion_status
            IN (
                'Completed',
                'Incomplete',
                'Failed'
            )
        ),

    CONSTRAINT chk_training_records_dates
        CHECK (
            completion_date IS NULL
            OR completion_date >= start_date
        ),

    CONSTRAINT chk_training_records_hours
        CHECK (
            training_hours >= 0
        ),

    CONSTRAINT chk_training_records_score
        CHECK (
            score IS NULL
            OR (
                score >= 0
                AND score <= 100
            )
        )
);


-- =========================================================
-- 7. Employee-event table
-- =========================================================

CREATE TABLE IF NOT EXISTS employee_events (
    event_id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL,
    event_date DATE NOT NULL,
    event_type TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    notes TEXT,

    CONSTRAINT fk_employee_events_employee
        FOREIGN KEY (
            employee_id
        )
        REFERENCES employees (
            employee_id
        ),

    CONSTRAINT chk_employee_events_type
        CHECK (
            event_type
            IN (
                'Hire',
                'Promotion',
                'Transfer',
                'Manager Change',
                'Leave',
                'Termination'
            )
        )
);


-- =========================================================
-- 8. Indexes
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_employees_department_id
    ON employees (
        department_id
    );


CREATE INDEX IF NOT EXISTS idx_employees_location_id
    ON employees (
        location_id
    );


CREATE INDEX IF NOT EXISTS idx_employees_job_role_id
    ON employees (
        job_role_id
    );


CREATE INDEX IF NOT EXISTS idx_employees_manager_id
    ON employees (
        manager_id
    );


CREATE INDEX IF NOT EXISTS idx_requisitions_job_role_id
    ON job_requisitions (
        job_role_id
    );


CREATE INDEX IF NOT EXISTS idx_requisitions_department_id
    ON job_requisitions (
        department_id
    );


CREATE INDEX IF NOT EXISTS idx_requisitions_location_id
    ON job_requisitions (
        location_id
    );


CREATE INDEX IF NOT EXISTS idx_requisitions_recruiter_id
    ON job_requisitions (
        recruiter_id
    );


CREATE INDEX IF NOT EXISTS idx_applications_candidate_id
    ON applications (
        candidate_id
    );


CREATE INDEX IF NOT EXISTS idx_applications_requisition_id
    ON applications (
        requisition_id
    );


CREATE INDEX IF NOT EXISTS idx_compensation_employee_id
    ON compensation_history (
        employee_id
    );


CREATE INDEX IF NOT EXISTS idx_reviews_employee_id
    ON performance_reviews (
        employee_id
    );


CREATE INDEX IF NOT EXISTS idx_reviews_reviewer_id
    ON performance_reviews (
        reviewer_id
    );


CREATE INDEX IF NOT EXISTS idx_training_records_employee_id
    ON training_records (
        employee_id
    );


CREATE INDEX IF NOT EXISTS idx_training_records_program_id
    ON training_records (
        program_id
    );


CREATE INDEX IF NOT EXISTS idx_employee_events_employee_id
    ON employee_events (
        employee_id
    );