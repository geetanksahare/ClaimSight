# Fairness Audit Report: Fraud Module

> This report evaluates demographic subgroup performance parity. It assesses statistical disparities in prediction errors across protected attributes; it neither proves nor disproves systemic discrimination.

**Task Type:** classification  
**Disparity Threshold:** 20% relative gap  

## Subgroup Performance Summary

### Attribute: `inpatient_share_tercile` (**FLAGGED**)
- Overall selection_rate: 0.0958  
- Max-Min Disparity Gap: 0.0758 (Relative Gap: 79.1%)

| Group | Count | Low N (<30) | Primary Metric |
|---|---|---|---|
| high | 79.0 | False | 0.0886 |
| low | 81.0 | False | 0.0617 |
| medium | 80.0 | False | 0.1375 |

### Attribute: `provider_size_tercile` (**FLAGGED**)
- Overall selection_rate: 0.0958  
- Max-Min Disparity Gap: 0.2875 (Relative Gap: 300.0%)

| Group | Count | Low N (<30) | Primary Metric |
|---|---|---|---|
| large | 80.0 | False | 0.2875 |
| medium | 80.0 | False | 0.0 |
| small | 80.0 | False | 0.0 |
