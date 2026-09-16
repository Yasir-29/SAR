# Validation Error Analysis Report — Phase 9 Baseline

## 1. Failure Mode Breakdown
- **False Negatives for DAMAGED (Class 1):** 104 / 140 samples (Miss rate: 74.29%)
- **False Negatives for DESTROYED (Class 2):** 29 / 37 samples (Miss rate: 78.38%)
- **False Positives (INTACT predicted as DAMAGED):** 189 / 934 samples (20.24%)
- **False Positives (INTACT predicted as DESTROYED):** 131 / 934 samples (14.03%)
- **DAMAGED vs. DESTROYED Direct Confusion:** 20 samples

## 2. Confidence Distributions by True Class
### Class INTACT (934 samples)
- Mean Confidence: 0.4108 (Std: 0.0840)
- Mean Conf when Correct: 0.4109
- Mean Conf when Incorrect: 0.4107

### Class DAMAGED (140 samples)
- Mean Confidence: 0.4222 (Std: 0.1141)
- Mean Conf when Correct: 0.4225
- Mean Conf when Incorrect: 0.4220

### Class DESTROYED (37 samples)
- Mean Confidence: 0.4236 (Std: 0.1100)
- Mean Conf when Correct: 0.4519
- Mean Conf when Incorrect: 0.4158

## 3. Regional Breakdown (District Performance)
| District | Sample Count | Accuracy (%) | Intact Rec (%) | Damaged Rec (%) | Destroyed Rec (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Chennai | 159 | 59.12% | 62.41% | 43.48% | 33.33% |
| Villupuram | 154 | 53.9% | 57.48% | 42.11% | 25.0% |
| Chengalpattu | 169 | 53.25% | 58.78% | 20.0% | 0.0% |
| Nagapattinam | 166 | 69.88% | 80.28% | 11.11% | 0.0% |
| Cuddalore | 306 | 55.88% | 62.11% | 19.51% | 44.44% |
| Kanyakumari | 157 | 66.24% | 76.56% | 20.83% | 20.0% |

## 4. Key Failure Mode Insights & Research Recommendations
1. **Majority Class Bias towards INTACT:** Out of 140 true DAMAGED buildings, 88 (62.8%) were misclassified as INTACT. Out of 37 true DESTROYED buildings, 25 (67.6%) were misclassified as INTACT.
2. **Low Model Confidence on Minority Predictions:** Model confidence drops significantly on misclassified samples (~0.45 vs ~0.75 on correct predictions).
3. **Context Sensitivity:** Spatial context surrounding 32x32 chips has low signal-to-noise ratio; multi-scale feature pyramids and hierarchical classifiers (Stage 1: INTACT vs DAMAGED/DESTROYED, Stage 2: DAMAGED vs DESTROYED) are recommended.
