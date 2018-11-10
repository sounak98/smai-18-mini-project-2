# Mini Project 2

## Data

### Train Data

- CIFAR-10 `data_batch_1` (10000 samples)

### Test Data

- CIFAR-10 `test_batch` (10000 samples)

## Data Representations

- Raw CIFAR-10 Data
- PCA
- LDA

## Models

### Logistic Regression

#### Parameters

- `solver`: `'lbfgs'`
- `max_iter`: 1000
- `multi_class`: `'auto'`

### SVM

#### Parameters (Set 1)

- `gamma`: `'scale'`
- `kernel`: `'rbf'`

#### Parameters (Set 2)

- `gamma`: `'scale'`
- `kernel`: `'poly'`

### Multi Layer Perceptron

#### Parameters

- `max_iter`: 1000

### CART

## Comparision

| Classifier | Features | Accuracy | F1-score |
|----|----|----|----|
| Poly-SVM | 100 Principal Components | 0.1508 | 0.14934665327562524 |
| Poly-SVM | LDA | 0.3805 | 0.37718876874450924 |
| Poly-SVM | Raw | 0.4166 | 0.41787978841459705 |
| RBF-SVM | 30 Principal Components | 0.0961 | 0.09267867665949577 |
| RBF-SVM | 100 Principal Components | 0.1 | 0.01818181818181818 |
| RBF-SVM | LDA | 0.4153 | 0.4139908084640867 |
| RBF-SVM | Raw | 0.2787 | 0.2780054554707241 |
| CART | 30 Principal Components | 0.1075 | 0.10701763059782572 |
| CART | 100 Principal Components | 0.1232 | 0.12310722750591085 |
| CART | LDA | 0.3157 | 0.3128037755685489 |
| CART | Raw | 0.2359 | 0.23577901598834647 |
| Logistic Regression | 30 Principal Components | 0.0961 | 0.09267867665949577 |
| Logistic Regression | 100 Principal Components | 0.1359 | 0.13413626349406624 |
| Logistic Regression | LDA | 0.4153 | 0.4139908084640867 |
| Logistic Regression | Raw | 0.2787 | 0.2780054554707241 |
| MLP | 30 Principal Components | 0.107 | 0.09898344564383543 |
| MLP | 100 Principal Components | 0.1378 | 0.13517955576492136 |
| MLP | LDA | 0.3986 | 0.3954523858932417 |
| MLP | Raw | 0.1 | 0.018560050274619155 |