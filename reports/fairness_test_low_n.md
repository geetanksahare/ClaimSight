# Fairness Audit Report: Test_Low_N Module

> This report evaluates demographic subgroup performance parity. It assesses statistical disparities in prediction errors across protected attributes; it neither proves nor disproves systemic discrimination.

**Task Type:** regression  
**Disparity Threshold:** 20% relative gap  

## Subgroup Performance Summary

### Attribute: `group` (Passed)
- Overall mae: 60.0  
- Max-Min Disparity Gap: 0.0 (Relative Gap: 0.0%)

| Group | Count | Low N (<30) | Primary Metric |
|---|---|---|---|
| A | 40.0 | False | 0.0 |
| B | 10.0 | True | 300.0 |
