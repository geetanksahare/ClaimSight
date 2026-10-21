# Fairness Audit Report: Test_Mod Module

> This report evaluates demographic subgroup performance parity. It assesses statistical disparities in prediction errors across protected attributes; it neither proves nor disproves systemic discrimination.

**Task Type:** regression  
**Disparity Threshold:** 20% relative gap  

## Subgroup Performance Summary

### Attribute: `sex` (**FLAGGED**)
- Overall mae: 30.0  
- Max-Min Disparity Gap: 40.0 (Relative Gap: 133.3%)

| Group | Count | Low N (<30) | Primary Metric |
|---|---|---|---|
| female | 35.0 | False | 50.0 |
| male | 35.0 | False | 10.0 |
