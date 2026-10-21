# Fairness Audit Report: Member Module

> This report evaluates demographic subgroup performance parity. It assesses statistical disparities in prediction errors across protected attributes; it neither proves nor disproves systemic discrimination.

**Task Type:** classification  
**Disparity Threshold:** 20% relative gap  

## Subgroup Performance Summary

### Attribute: `sex` (**FLAGGED**)
- Overall selection_rate: 0.0038  
- Max-Min Disparity Gap: 0.0081 (Relative Gap: 210.6%)

| Group | Count | Low N (<30) | Primary Metric |
|---|---|---|---|
| female | 410.0 | False | 0.0 |
| male | 370.0 | False | 0.0081 |

### Attribute: `race` (**FLAGGED**)
- Overall selection_rate: 0.0038  
- Max-Min Disparity Gap: 0.0047 (Relative Gap: 122.2%)

| Group | Count | Low N (<30) | Primary Metric |
|---|---|---|---|
| 1 | 634.0 | False | 0.0047 |
| 2 | 97.0 | False | 0.0 |
| 3 | 26.0 | True | 0.0 |
| 5 | 23.0 | True | 0.0 |

### Attribute: `age_band` (**FLAGGED**)
- Overall selection_rate: 0.0038  
- Max-Min Disparity Gap: 0.0067 (Relative Gap: 174.2%)

| Group | Count | Low N (<30) | Primary Metric |
|---|---|---|---|
| 65-74 | 326.0 | False | 0.0031 |
| 75-84 | 300.0 | False | 0.0067 |
| <65 | 154.0 | False | 0.0 |
