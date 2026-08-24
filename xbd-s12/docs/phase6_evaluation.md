# Phase 6 Model Evaluation Metrics

This document describes the validation strategy, test metrics, and performance checks used to assess the building-damage classification model.

## 1. Multi-Class Evaluation Metrics
We evaluate the model on the Validation and Test sets using:
- **Per-Class Precision, Recall, and F1-Score:**
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Macro F1-Score:** The unweighted average of the F1-scores of Class 0, Class 1, and Class 2. This is the primary metric to assess damage classification performance under remaining class imbalance.
- **Overall Accuracy:** The fraction of correctly classified buildings.
- **Confusion Matrix:** Tracks misclassifications (e.g. destroyed buildings predicted as intact).

## 2. Validation Checks & Safety Controls
Before deploying the model weights, the following validation checks must pass:
1. **Class-Wise Check:** The model must achieve a macro F1-score $> 0.60$ on the test set.
2. **False Clear Prevention:** A high penalty is applied to False Clears (i.e. destroyed/damaged buildings predicted as intact). The recall for Class 2 (Destroyed) must be monitored and optimized using class-weighted loss.
3. **Scientific Safety Override:** If prediction confidence for a building is low (e.g. maximum class probability $< 0.50$), the pipeline flags it as **"uncertain"** rather than forcing a damage classification label.
