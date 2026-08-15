# Hallucination Probe Training & Evaluation Report

- **Best Representation**: `mean_pooled`
- **Best Layers (Cohen's d Rankings)**: `[32, 15, 17, 16, 14]`
- **Selected Probe Layers (Top 3)**: `[32, 15, 17]`

## In-Domain Evaluation (80/20 Stratified Split)
- **Accuracy**: `0.6074`
- **Precision**: `0.6512`
- **Recall**: `0.6222`
- **F1 Score**: `0.6364`
- **MLP Validation F1**: `0.6813`

### Confusion Matrix (Linear):
```
[[43 30]
 [34 56]]
```

### Classification Report (Linear):
```
              precision    recall  f1-score   support

           0       0.56      0.59      0.57        73
           1       0.65      0.62      0.64        90

    accuracy                           0.61       163
   macro avg       0.60      0.61      0.60       163
weighted avg       0.61      0.61      0.61       163

```

## Held-Out Category Generalization Tests

### Hold-Out Category: `Misconceptions`
- **Generalization F1**: `0.4359`
- **Generalization Accuracy**: `0.5600`
- **Generalization Precision**: `0.5312`
- **Generalization Recall**: `0.3696`
- **Positive Predictions Made**: `32`

#### Confusion Matrix:
```
[[39 15]
 [29 17]]
```
```
              precision    recall  f1-score   support

           0       0.57      0.72      0.64        54
           1       0.53      0.37      0.44        46

    accuracy                           0.56       100
   macro avg       0.55      0.55      0.54       100
weighted avg       0.55      0.56      0.55       100

```

### Hold-Out Category: `Law`
- **Generalization F1**: `0.5974`
- **Generalization Accuracy**: `0.5156`
- **Generalization Precision**: `0.6571`
- **Generalization Recall**: `0.5476`
- **Positive Predictions Made**: `35`

#### Confusion Matrix:
```
[[10 12]
 [19 23]]
```
```
              precision    recall  f1-score   support

           0       0.34      0.45      0.39        22
           1       0.66      0.55      0.60        42

    accuracy                           0.52        64
   macro avg       0.50      0.50      0.49        64
weighted avg       0.55      0.52      0.53        64

```
