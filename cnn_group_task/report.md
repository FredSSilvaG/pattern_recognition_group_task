Nguyen Chi Manh
1. Model Architecture
Dynamic CNN Structure
The CNN architecture is designed to be configurable with the following hyperparameters:

Number of convolutional layers: Varies between 3, 4, or 5 layers
Kernel size: Configurable as 3×3, 5×5, or 7×7
Learning rate: Varied between 0.1, 0.01, 0.001, and 0.0001

1.1 Layer Configuration
CNN_dynamic(
  (conv_layers): Sequential layers with:
    - Initial Conv2D (1→32 channels, kernel_size×kernel_size)
    - Multiple Conv2D layers (32→32 channels)
    - ReLU activations after each conv layer
    - MaxPool2D (2×2) after all but last convolutional layer
    - Dropout (0.25) for regularization
  
  (fc_layers): Sequential layers with:
    - Flatten operation
    - Linear layer (32*feature_map_size → 128)
    - ReLU activation
    - Linear layer (128 → 10 classes)
)

2. Hyperparameter Search
Grid Search Implementation
The code performs an exhaustive grid search across:

3 kernel sizes [3, 5, 7]
3 layer depths [3, 4, 5]
4 learning rates [0.1, 0.01, 0.001, 0.0001]

3. Report
General statement:
Learning Rate (LR): This is the most critical parameter. LR 0.1 consistently fails (~10% accuracy). LR 0.01 is unstable, sometimes working well (up to 98.26%) but often failing (~11%). LR 0.001 consistently yields the best results (98.7% - 99.2% test accuracy) across all tested layers and kernels. LR 0.0001 also works well (mostly >98%) but converges slightly slower than LR 0.001 within 3 epochs.
Kernel Size (K): When using the best LR (0.001), increasing kernel size from K3 to K5 or K7 generally provides a small improvement in test accuracy (pushing it slightly above 99%). K5 and K7 perform similarly well.
Layer Depth (L): With the optimal LR (0.001), increasing layer depth from L3 to L4 or L5 does not show a significant or consistent impact on final test accuracy within these 3 epochs. All depths (L3, L4, L5) achieve excellent results (~99%).

Numerical report
Epoch 1/3, Train Acc: 10.46%, Test Acc: 9.80%
Config: L3_K3_LR0.1
Epoch 2/3, Train Acc: 10.34%, Test Acc: 10.32%
Config: L3_K3_LR0.1
Epoch 3/3, Train Acc: 10.32%, Test Acc: 9.74%
Config: L3_K3_LR0.1
Epoch 1/3, Train Acc: 93.00%, Test Acc: 97.86%
Config: L3_K3_LR0.01
Epoch 2/3, Train Acc: 96.33%, Test Acc: 97.80%
Config: L3_K3_LR0.01
Epoch 3/3, Train Acc: 96.63%, Test Acc: 98.26%
Config: L3_K3_LR0.01
Epoch 1/3, Train Acc: 95.44%, Test Acc: 98.50%
Config: L3_K3_LR0.001
Epoch 2/3, Train Acc: 98.40%, Test Acc: 98.83%
Config: L3_K3_LR0.001
Epoch 3/3, Train Acc: 98.88%, Test Acc: 98.95%
Config: L3_K3_LR0.001
Epoch 1/3, Train Acc: 88.58%, Test Acc: 96.45%
Config: L3_K3_LR0.0001
Epoch 2/3, Train Acc: 96.48%, Test Acc: 98.00%
Config: L3_K3_LR0.0001
Epoch 3/3, Train Acc: 97.53%, Test Acc: 98.27%
Config: L3_K3_LR0.0001

Epoch 1/3, Train Acc: 10.56%, Test Acc: 11.35%
Config: L3_K5_LR0.1
Epoch 2/3, Train Acc: 10.33%, Test Acc: 10.10%
Config: L3_K5_LR0.1
Epoch 3/3, Train Acc: 10.37%, Test Acc: 11.35%
Config: L3_K5_LR0.1
Epoch 1/3, Train Acc: 91.06%, Test Acc: 96.34%
Config: L3_K5_LR0.01
Epoch 2/3, Train Acc: 93.86%, Test Acc: 96.76%
Config: L3_K5_LR0.01
Epoch 3/3, Train Acc: 94.03%, Test Acc: 96.86%
Config: L3_K5_LR0.01
Epoch 1/3, Train Acc: 96.04%, Test Acc: 99.03%
Config: L3_K5_LR0.001
Epoch 2/3, Train Acc: 98.68%, Test Acc: 98.70%
Config: L3_K5_LR0.001
Epoch 3/3, Train Acc: 99.06%, Test Acc: 99.21%
Config: L3_K5_LR0.001
Epoch 1/3, Train Acc: 91.51%, Test Acc: 97.54%
Config: L3_K5_LR0.0001
Epoch 2/3, Train Acc: 97.60%, Test Acc: 98.53%
Config: L3_K5_LR0.0001
Epoch 3/3, Train Acc: 98.35%, Test Acc: 98.86%
Config: L3_K5_LR0.0001

Epoch 1/3, Train Acc: 10.32%, Test Acc: 10.09%
Config: L3_K7_LR0.1
Epoch 2/3, Train Acc: 10.45%, Test Acc: 10.32%
Config: L3_K7_LR0.1
Epoch 3/3, Train Acc: 10.37%, Test Acc: 11.35%
Config: L3_K7_LR0.1
Epoch 1/3, Train Acc: 10.94%, Test Acc: 11.35%
Config: L3_K7_LR0.01
Epoch 2/3, Train Acc: 11.00%, Test Acc: 11.35%
Config: L3_K7_LR0.01
Epoch 3/3, Train Acc: 11.07%, Test Acc: 11.35%
Config: L3_K7_LR0.01
Epoch 1/3, Train Acc: 96.25%, Test Acc: 98.79%
Config: L3_K7_LR0.001
Epoch 2/3, Train Acc: 98.59%, Test Acc: 98.80%
Config: L3_K7_LR0.001
Epoch 3/3, Train Acc: 98.91%, Test Acc: 99.14%
Config: L3_K7_LR0.001
Epoch 1/3, Train Acc: 92.05%, Test Acc: 98.00%
Config: L3_K7_LR0.0001
Epoch 2/3, Train Acc: 97.86%, Test Acc: 98.58%
Config: L3_K7_LR0.0001
--------------------
Epoch 1/3, Train Acc: 10.33%, Test Acc: 9.80%
Config: L4_K3_LR0.1
Epoch 2/3, Train Acc: 10.48%, Test Acc: 11.35%
Config: L4_K3_LR0.1
Epoch 3/3, Train Acc: 10.46%, Test Acc: 11.35%
Config: L4_K3_LR0.1
Epoch 1/3, Train Acc: 89.38%, Test Acc: 96.99%
Config: L4_K3_LR0.01
Epoch 2/3, Train Acc: 94.21%, Test Acc: 97.08%
Config: L4_K3_LR0.01
Epoch 3/3, Train Acc: 94.72%, Test Acc: 96.75%
Config: L4_K3_LR0.01
Epoch 1/3, Train Acc: 93.57%, Test Acc: 98.55%
Config: L4_K3_LR0.001
Epoch 2/3, Train Acc: 97.94%, Test Acc: 98.83%
Config: L4_K3_LR0.001
Epoch 3/3, Train Acc: 98.40%, Test Acc: 98.75%
Config: L4_K3_LR0.001
Epoch 1/3, Train Acc: 83.23%, Test Acc: 95.40%
Config: L4_K3_LR0.0001
Epoch 2/3, Train Acc: 95.45%, Test Acc: 97.60%
Config: L4_K3_LR0.0001
Epoch 3/3, Train Acc: 96.82%, Test Acc: 98.20%
Config: L4_K3_LR0.0001

Epoch 1/3, Train Acc: 10.46%, Test Acc: 11.35%
Config: L4_K5_LR0.1
Epoch 2/3, Train Acc: 10.37%, Test Acc: 11.35%
Config: L4_K5_LR0.1
Epoch 3/3, Train Acc: 10.41%, Test Acc: 9.80%
Config: L4_K5_LR0.1
Epoch 1/3, Train Acc: 10.98%, Test Acc: 11.35%
Config: L4_K5_LR0.01
Epoch 2/3, Train Acc: 11.02%, Test Acc: 11.35%
Config: L4_K5_LR0.01
Epoch 3/3, Train Acc: 11.13%, Test Acc: 11.35%
Config: L4_K5_LR0.01
Epoch 1/3, Train Acc: 94.72%, Test Acc: 98.50%
Config: L4_K5_LR0.001
Epoch 2/3, Train Acc: 98.33%, Test Acc: 98.73%
Config: L4_K5_LR0.001
Epoch 3/3, Train Acc: 98.69%, Test Acc: 99.13%
Config: L4_K5_LR0.001
Epoch 1/3, Train Acc: 86.01%, Test Acc: 97.01%
Config: L4_K5_LR0.0001
Epoch 2/3, Train Acc: 96.61%, Test Acc: 98.28%
Config: L4_K5_LR0.0001
Epoch 3/3, Train Acc: 97.52%, Test Acc: 98.62%
Config: L4_K5_LR0.0001

Epoch 1/3, Train Acc: 10.37%, Test Acc: 11.35%
Config: L4_K7_LR0.1
Epoch 2/3, Train Acc: 10.58%, Test Acc: 10.28%
Config: L4_K7_LR0.1
Epoch 3/3, Train Acc: 10.31%, Test Acc: 9.80%
Config: L4_K7_LR0.1
Epoch 1/3, Train Acc: 75.32%, Test Acc: 93.13%
Config: L4_K7_LR0.01
Epoch 2/3, Train Acc: 87.64%, Test Acc: 91.44%
Config: L4_K7_LR0.01
Epoch 3/3, Train Acc: 88.74%, Test Acc: 94.57%
Config: L4_K7_LR0.01
Epoch 1/3, Train Acc: 94.64%, Test Acc: 98.45%
Config: L4_K7_LR0.001
Epoch 2/3, Train Acc: 98.31%, Test Acc: 98.99%
Config: L4_K7_LR0.001
Epoch 3/3, Train Acc: 98.62%, Test Acc: 99.16%
Config: L4_K7_LR0.001
Epoch 1/3, Train Acc: 88.32%, Test Acc: 97.28%
Config: L4_K7_LR0.0001
Epoch 2/3, Train Acc: 96.77%, Test Acc: 98.21%
Config: L4_K7_LR0.0001
Epoch 3/3, Train Acc: 97.64%, Test Acc: 98.66%
Config: L4_K7_LR0.0001
-----------
Epoch 1/3, Train Acc: 10.39%, Test Acc: 10.10%
Config: L5_K3_LR0.1
Epoch 2/3, Train Acc: 10.22%, Test Acc: 11.35%
Config: L5_K3_LR0.1
Epoch 3/3, Train Acc: 10.50%, Test Acc: 10.10%
Config: L5_K3_LR0.1
Epoch 1/3, Train Acc: 11.07%, Test Acc: 11.35%
Config: L5_K3_LR0.01
Epoch 2/3, Train Acc: 10.96%, Test Acc: 11.35%
Config: L5_K3_LR0.01
Epoch 3/3, Train Acc: 11.08%, Test Acc: 11.35%
Config: L5_K3_LR0.01
Epoch 1/3, Train Acc: 88.99%, Test Acc: 98.03%
Config: L5_K3_LR0.001
Epoch 2/3, Train Acc: 96.57%, Test Acc: 98.76%
Config: L5_K3_LR0.001
Epoch 3/3, Train Acc: 97.43%, Test Acc: 98.92%
Config: L5_K3_LR0.001
Epoch 1/3, Train Acc: 69.04%, Test Acc: 93.72%
Config: L5_K3_LR0.0001
Epoch 2/3, Train Acc: 90.33%, Test Acc: 95.74%
Config: L5_K3_LR0.0001
Epoch 3/3, Train Acc: 92.94%, Test Acc: 96.81%
Config: L5_K3_LR0.0001

Epoch 1/3, Train Acc: 10.41%, Test Acc: 9.82%
Config: L5_K5_LR0.1
Epoch 2/3, Train Acc: 10.21%, Test Acc: 10.09%
Config: L5_K5_LR0.1
Epoch 3/3, Train Acc: 10.17%, Test Acc: 11.35%
Config: L5_K5_LR0.1
Epoch 1/3, Train Acc: 11.03%, Test Acc: 11.35%
Config: L5_K5_LR0.01
Epoch 2/3, Train Acc: 10.94%, Test Acc: 11.35%
Config: L5_K5_LR0.01
Epoch 3/3, Train Acc: 10.97%, Test Acc: 11.35%
Config: L5_K5_LR0.01
Epoch 1/3, Train Acc: 91.42%, Test Acc: 98.50%
Config: L5_K5_LR0.001
Epoch 2/3, Train Acc: 97.64%, Test Acc: 98.82%
Config: L5_K5_LR0.001
Epoch 3/3, Train Acc: 98.17%, Test Acc: 99.15%
Config: L5_K5_LR0.001
Epoch 1/3, Train Acc: 77.01%, Test Acc: 95.54%
Config: L5_K5_LR0.0001
Epoch 2/3, Train Acc: 94.03%, Test Acc: 97.23%
Config: L5_K5_LR0.0001
Epoch 3/3, Train Acc: 95.84%, Test Acc: 97.86%

Epoch 1/3, Train Acc: 10.32%, Test Acc: 10.28%
Config: L5_K7_LR0.1
Epoch 2/3, Train Acc: 10.20%, Test Acc: 11.35%
Config: L5_K7_LR0.1
Epoch 3/3, Train Acc: 10.54%, Test Acc: 10.32%
Config: L5_K7_LR0.1
Epoch 1/3, Train Acc: 11.12%, Test Acc: 10.28%
Config: L5_K7_LR0.01
Epoch 2/3, Train Acc: 11.10%, Test Acc: 11.35%
Config: L5_K7_LR0.01
Epoch 3/3, Train Acc: 11.13%, Test Acc: 11.35%
Config: L5_K7_LR0.01
Epoch 1/3, Train Acc: 91.67%, Test Acc: 98.83%
Config: L5_K7_LR0.001
Epoch 2/3, Train Acc: 97.95%, Test Acc: 99.15%
Config: L5_K7_LR0.001
Epoch 3/3, Train Acc: 98.38%, Test Acc: 99.20%
Config: L5_K7_LR0.001
Epoch 1/3, Train Acc: 78.48%, Test Acc: 95.17%
Config: L5_K7_LR0.0001
Epoch 2/3, Train Acc: 94.58%, Test Acc: 97.25%
Config: L5_K7_LR0.0001