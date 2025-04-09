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


#SVM on MNIST

|        |       C | Mean Accuracy | Std Accuracy | Mean Hinge Loss | Std Hinge Loss |
|--------|---------|---------------|--------------|-----------------|----------------|
| linear |  0.0001 |        0.9090 |       0.0033 |          0.2521 |         0.0087 |
| rbf    |  0.0001 |        0.1124 |       0.0012 |          5.5305 |         0.0209 |

Best model's parameters:{"C": 0.0001, "gamma": "scale", "kernel": "linear"}
Best SVM accuracy on test set:36.64%
Best model classification_report:
              precision    recall  f1-score   support

           0       0.98      0.66      0.78       980
           1       0.00      0.00      0.00      1135
           2       0.87      0.50      0.64      1032
           3       0.79      0.51      0.62      1010
           4       0.91      0.24      0.37       982
           5       0.48      0.07      0.13       892
           6       0.99      0.35      0.52       958
           7       1.00      0.06      0.11      1028
           8       0.15      1.00      0.26       974
           9       0.50      0.33      0.39      1009

    accuracy                           0.37     10000
   macro avg       0.67      0.37      0.38     10000
weighted avg       0.66      0.37      0.38     10000
