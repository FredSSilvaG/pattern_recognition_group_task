#SVM on MNIST

|        |       C | Mean Accuracy | Std Accuracy | Mean Hinge Loss | Std Hinge Loss |
|--------|---------|---------------|--------------|-----------------|----------------|
| linear |     0.1 |        0.9281 |       0.0031 |          0.1897 |         0.0068 |
| linear |     1.0 |        0.9283 |       0.0031 |          0.1895 |         0.0065 |
| rbf    |     0.1 |        0.9425 |       0.0019 |          0.1658 |         0.0059 |
| rbf    |     1.0 |        0.9688 |       0.0012 |          0.0829 |         0.0036 |

Best model's parameters:{"C": 1, "gamma": "scale", "kernel": "rbf"}
Best SVM accuracy on test set:16.13%
Best model classification_report:
              precision    recall  f1-score   support

           0       0.99      0.17      0.28       980
           1       0.00      0.00      0.00      1135
           2       0.89      0.30      0.45      1032
           3       0.98      0.09      0.17      1010
           4       0.63      0.06      0.11       982
           5       0.00      0.00      0.00       892
           6       1.00      0.02      0.04       958
           7       0.00      0.00      0.00      1028
           8       0.10      1.00      0.19       974
           9       0.00      0.00      0.00      1009

    accuracy                           0.16     10000
   macro avg       0.46      0.16      0.12     10000
weighted avg       0.45      0.16      0.12     10000
