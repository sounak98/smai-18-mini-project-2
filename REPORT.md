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

## Summary of Code

We got to experiment with multiple hyperparameters and
also learnt about various scikit-learn libraries and
how to use them. The above code has assumed F1 score
averaging to be done using `weighted`.

## Observations

There were a lot of interesting observations, which
are mentioned as follows.

- _LDA_ was giving the best results.
- _MLP_ gave more accurate results when hyperparameters
  like number of hidden layers, number of iterations,
  and activation functions. In the default format it
  gave accuracy of about `0.1`.
- Solver `lbfgs` gives maximum accuracy in _Logistic Regression_
  and _MLP_.
- Accuracy increased by `0.05` when used `100` principal components
  instead of `30`.
- Activation functions `tanh` and `relu` in _MLP_ give
  almost the same accuracy and f1 score.
- MLP was the fastest to calculate as compared to
  Soft Margin Linear SVM, Logistic Regression and
  Kernel SVM with RBF as kernel.
- `poly` Kernel gives better results than `rbf`.
- Using shrinking heuristic in Kernel SVM with `rbf` as
  kernel increases accuracy and f1 score.
- Limiting maximum iterations in both Kernel SVM with
  `rbf` and `poly` as kernel reduces accuracy and f1 score 
  by as much as `0.06`.


## Problems Faced

The primary issue was with crude pixel information due to the
measure of time spent on preparing and anticipating was
large. This required high processing power. Also
changing and playing with the hyperparameters of the 
dimensionality reduction systems and classifiers 
required repeated execution of code.

### Problem of Overfitting

One way to quantify overfitting is as the difference
between the training accuracy and the test accuracy. So
this notion of overfitting already occurs on the
existing dataset. Another notion of overfitting is the
gap between the test accuracy and the accuracy on the
underlying data distribution. By adapting model design
choices to the test set, the concern is that we
implicitly fit the model to the test set. The test
accuracy then loses its validity as an accurate measure
of performance on truly unseen data.

To remove overfitting, we create another dataset and 
we measure the accuracy of CIFAR-10 classifiers by
creating a new test set of truly unseen images. The
data collection for new dataset will be designed to
minimize the distribution shift relative to the original
dataset.