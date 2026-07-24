# Historical Assignment Attribution Analysis

## Purpose

This analysis tests whether historical workforce activity is incorrectly assigned to employees' final departments.

The proposed method assumed that `Transfer` events recorded department changes. Inspection showed that this assumption was incorrect: Version 1 transfers represent location changes.

The analysis therefore has two parts:

1. Validate whether department-history bias exists.
2. Quantify the analogous location-history attribution bias.

## Correction to the Original Assumption

Every `Transfer` event contains:

```text
old_value = previous location_id
new_value = final location_id