# Workforce Analytics

## Purpose

The SQL analytics layer transforms the synthetic workforce database into business-focused workforce metrics.

## Main analysis areas

The analytics currently cover:

- Active employee headcount
- Department distribution
- Geographic workforce distribution
- Job-family distribution
- Annual department turnover
- Recruiting funnel performance
- Requisition duration
- Current compensation
- Latest performance reviews
- Employee training activity

## SQL queries

The main SQL analysis queries are stored in:

`sql/workforce_analytics.sql`

## Analysis notebook

The PostgreSQL analytics notebook is:

`notebooks/11_sql_workforce_analytics.ipynb`

The notebook connects directly to PostgreSQL and validates key analytical results.

## Turnover methodology

The simplified annual turnover calculation is:

`Terminations during the year / Average annual headcount`

Average annual headcount is estimated as:

`(Starting headcount + Ending headcount) / 2`

Version 1 does not generate department transfers: all 10,000 employees retain their hire department, so current-department and event-time department attribution are identical by construction. Transfer events represent location changes, not department changes. Department-level turnover remains a synthetic analytical estimate because its underlying termination rates are generated. Historical location analysis should reconstruct event-time location when relevant. See [Historical Assignment Attribution Analysis](department_history_bias.md).

## Current-record methodology

For employee-level analysis:

- Current compensation uses the latest available compensation record.
- Current performance uses the latest available performance review.
- Training records are aggregated by employee.
- Only active employees are included in current workforce summaries.

## Future use

The SQL analytics layer will support:

- Retention feature engineering
- Attrition modeling
- Workforce dashboards
- Management decision-support metrics