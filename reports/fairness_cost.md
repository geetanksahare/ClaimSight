# Fairness Audit Report: Cost Module

> This report evaluates demographic subgroup performance parity. It assesses statistical disparities in prediction errors across protected attributes; it neither proves nor disproves systemic discrimination.

**Task Type:** regression  
**Disparity Threshold:** 20% relative gap  

## Subgroup Performance Summary

### Attribute: `sex` (Passed)
- Overall mae: 1514.3828  
- Max-Min Disparity Gap: 285.3843 (Relative Gap: 18.8%)

| Group | Count | Low N (<30) | Primary Metric |
|---|---|---|---|
| female | 104.0 | False | 1376.66 |
| male | 97.0 | False | 1662.0443 |

### Attribute: `region` (**FLAGGED**)
- Overall mae: 1514.3828  
- Max-Min Disparity Gap: 389.2052 (Relative Gap: 25.7%)

| Group | Count | Low N (<30) | Primary Metric |
|---|---|---|---|
| northeast | 47.0 | False | 1300.5717 |
| northwest | 54.0 | False | 1689.7769 |
| southeast | 47.0 | False | 1379.0975 |
| southwest | 53.0 | False | 1645.2554 |

### Attribute: `age_band` (**FLAGGED**)
- Overall mae: 1514.3828  
- Max-Min Disparity Gap: 1371.3164 (Relative Gap: 90.5%)

| Group | Count | Low N (<30) | Primary Metric |
|---|---|---|---|
| 18-29 | 59.0 | False | 1253.0717 |
| 30-39 | 40.0 | False | 1136.8824 |
| 40-49 | 44.0 | False | 2508.1988 |
| 50-64 | 58.0 | False | 1286.6149 |
