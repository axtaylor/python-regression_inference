# regression-inference

[![PyPI version](https://img.shields.io/pypi/v/regression-inference)](https://pypi.org/project/regression-inference/)
[![License](https://img.shields.io/github/license/axtaylor/python-ordinary_least_squares?color)](https://github.com/axtaylor/python-ordinary_least_squares/blob/main/LICENSE)


```
pip install regression-inference
```

Python packaged designed for statistical inference in machine learning, econometrics, and research, with support for hardware accelerated training.


---
### Features


- Linear Regression, Logistic Regression, Multinomial Logistic Regression, and Ordinal Logistic Regression model fitting.

- Full inference statistics for model predictions including standard errors, t/z,
significance, and confidence ranges.

- Support for accelerated model training.

- Modular regression tables.


---
### Dependencies

`numpy` and `scipy` are required as dependencies in `regression-inference`

```
numpy>=2.0.0
scipy>=1.15.0
```


Using `pandas` is optional but recommended.

---
### Hardware Acceleration with CUDA

CUDA acceleration can be used for model training on supported hardware.


[CuPy](https://cupy.dev/) is a required dependency for hardware acceleration, it is not installed with `regression-inference` by default. Additionally, the [CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit) is a 
required dependency unless `cupy` is installed from `conda-forge`.






Fit models with the parameter `cuda = True` to enable hardware GPU acceleration.

```py
model = OrdinalLogisticRegression().fit(X, y, cuda = True)
```

```py
model = MultinomialLogisticRegression().fit(X, y, cuda = True)
```


---
### Documentation / How To

See the provided [notebooks](https://github.com/axtaylor/python-ordinary_least_squares/tree/main/tests/notebooks) on GitHub for example workflows.

```
/tests/notebooks/linear_regression_example.ipynb

/tests/notebooks/logit_regression_example.ipynb

/tests/notebooks/multinomial_regression_example.ipynb

/tests/notebooks/ordinal_regression_example.ipynb
```

---
### Classes and Functions


```python
from regression_inference import LinearRegression, LogisticRegression, MultinomialLogisticRegression, OrdinalLogisticRegression, summary
```


---
### Ordinary Least Squares Regression Output



```
==================================================
OLS Regression Results
--------------------------------------------------
Dependent:                     educ     robust_edu
--------------------------------------------------
 
const                     7.3256***      7.3256***
                           (0.3684)       (0.4345)
 
paeduc                    0.2144***      0.2144***
                           (0.0241)       (0.0236)
 
maeduc                    0.2569***      0.2569***
                           (0.0271)       (0.0294)
 
age                       0.0241***      0.0241***
                           (0.0043)       (0.0042)

--------------------------------------------------
R-squared                     0.276          0.276
Adjusted R-squared            0.274          0.274
F Statistic                 177.548        177.548
Observations               1402.000       1402.000
Log Likelihood            -3359.107      -3359.107
AIC                        6726.213       6726.213
BIC                        6747.196       6747.196
TSS                       13663.270      13663.270
RSS                        9893.727       9893.727
ESS                        3769.543       3769.543
MSE                           7.077          7.077
==================================================
*p<0.1; **p<0.05; ***p<0.01
```

### Logistic Regression Summary

```
===================================
Logistic Regression Results
-----------------------------------
Dependent:                    GRADE
-----------------------------------
 
const                    -13.0213**
                           (5.1976)
 
GPA                        2.8261**
                           (1.2675)
 
TUCE                         0.0952
                           (0.1179)
 
PSI                        2.3787**
                           (0.9644)

-----------------------------------
Pseudo R-squared              0.374
LR Statistic                 15.404
Observations                 32.000
Log Likelihood              -12.890
Deviance                     25.779
Null Deviance                41.183
AIC                          33.779
BIC                          39.642
===================================
*p<0.1; **p<0.05; ***p<0.01
```

### Multinomial Logit Summary

```
=============================================
Multinomial Regression Results
---------------------------------------------
Dependent:                                PID
---------------------------------------------
Class:                                      1

const                                 -0.3734
                                     (0.5943)
 
logpopul                              -0.0115
                                     (0.0341)
 
selfLR                              0.2977***
                                     (0.0993)
 
age                                -0.0249***
                                     (0.0061)
 
educ                                   0.0825
                                     (0.0740)
 
income                                 0.0052
                                     (0.0168)
 
---------------------------------------------
Class:                                      2

const                              -2.2509***
                                     (0.7579)
 
logpopul                            -0.0888**
                                     (0.0377)
 
selfLR                              0.3917***
                                     (0.1089)
 
age                                -0.0229***
                                     (0.0084)
 
educ                                 0.1810**
                                     (0.0862)
 
income                               0.0479**
                                     (0.0234)
 
---------------------------------------------
Class:                                      3

const                              -3.6656***
                                     (1.3816)
 
logpopul                              -0.1060
                                     (0.0659)
 
selfLR                              0.5735***
                                     (0.1648)
 
age                                   -0.0149
                                     (0.0107)
 
educ                                  -0.0072
                                     (0.1234)
 
income                                 0.0576
                                     (0.0390)
 
---------------------------------------------
Class:                                      4

const                              -7.6138***
                                     (1.0433)
 
logpopul                            -0.0916**
                                     (0.0452)
 
selfLR                              1.2788***
                                     (0.1382)
 
age                                   -0.0087
                                     (0.0086)
 
educ                                 0.1998**
                                     (0.0966)
 
income                              0.0845***
                                     (0.0262)
 
---------------------------------------------
Class:                                      5

const                              -7.0605***
                                     (0.8462)
 
logpopul                            -0.0933**
                                     (0.0399)
 
selfLR                              1.3470***
                                     (0.1252)
 
age                                 -0.0179**
                                     (0.0078)
 
educ                                0.2169***
                                     (0.0816)
 
income                              0.0810***
                                     (0.0219)
 
---------------------------------------------
Class:                                      6

const                             -12.1058***
                                     (1.2198)
 
logpopul                           -0.1409***
                                     (0.0427)
 
selfLR                              2.0701***
                                     (0.1747)
 
age                                   -0.0094
                                     (0.0084)
 
educ                                0.3219***
                                     (0.0879)
 
income                              0.1089***
                                     (0.0260)
 
---------------------------------------------
Accuracy                                0.394
Pseudo R-squared                        0.165
LR Statistic                          576.848
Observations                          944.000
Log Likelihood                      -1461.923
Null Log Likelihood                 -1750.347
Deviance                             2923.845
Null Deviance                        3500.693
AIC                                  2995.845
BIC                                  3170.450
=============================================
*p<0.1; **p<0.05; ***p<0.01
```

### Ordinal Regression Summary

```
=============================================
Ordinal Regression Results
---------------------------------------------
Dependent:                                PID
---------------------------------------------
 
logpopul                           -0.0707***
                                     (0.0191)
 
selfLR                              1.0192***
                                     (0.0533)
 
age                                   -0.0042
                                     (0.0037)
 
educ                                0.1777***
                                     (0.0408)
 
income                              0.0472***
                                     (0.0108)
 
0:1                                 3.6891***
                                     (0.3729)
 
1:2                                 0.2243***
                                     (0.0686)
 
2:3                                -0.3445***
                                     (0.0910)
 
3:4                                -1.3565***
                                     (0.1607)
 
4:5                                -0.4247***
                                     (0.0979)
 
5:6                                  0.1653**
                                     (0.0752)

---------------------------------------------
Accuracy                                0.388
Pseudo R-squared                        0.146
LR Statistic                          511.454
Observations                          944.000
Log Likelihood                      -1494.620
Null Log Likelihood                 -1750.347
Deviance                             2989.239
Null Deviance                        3500.693
AIC                                  3011.239
BIC                                  3064.590
=============================================
*p<0.1; **p<0.05; ***p<0.01
```

---
### Coefficient Inference Table

Inference tables can be generated for the model features.


```py
pd.DataFrame(model.inference_table())
```



![](./static/3.png)


---
### Predictions

Extract the order of feature names using `feature_names[:1]`

```
model.feature_names[1:]
```
```
[Out]: Index(['paeduc', 'maeduc', 'age'], dtype='object')
```

Predict in the order of the features without a constant.

```
model.predict(np.array([0, 0, 0], ))
```
```
[Out]: 7.32564767
```

---
### Inference


Use `return_table = True` to include inference statistics.



```py
# Range over values of a feature

prediction_set = [
    (np.array([i, X['maeduc'].mean(), X['age'].mean()],))
    for i in range(int(X['paeduc'].min()), int(X['paeduc'].max())+1)
    ] 
    
predictions = pd.concat(pd.DataFrame(model.predict(i, return_table=True)) for i in prediction_set)
```

![](./static/1.png)



```py
# Predict discrete values

prediction_set = [
    np.array([2.66, 20.0, 0.0]),
    np.array([2.89, 22.0, 0.0]),
    np.array([3.28, 24.0, 0.0]),
    np.array([2.92, 12.0, 0.0]),
]

predictions = pd.concat(pd.DataFrame(model.predict(test_set, return_table=True)) for test_set in prediction_set)
```

![](./static/2.png)


---
### Variance Inflation Factor

Variance Inflation Factor table can be generated for the model features.

```py
model.variance_inflation_factor()
```


```
{'feature': Index(['paeduc', 'maeduc', 'age'], dtype='object'),
 'VIF': array([2.0233, 2.0285, 1.0971])}
```

---
### Heteroskedastic-Robust Standard Errors

Set the covariance matrix on fit using `cov_type`:

```py
model = MultinomialLogisticRegression().fit(X, y, cov_type="HC0")

model = LogisticRegression().fit(X, y, cov_type="HC1")

model = LinearRegression().fit(X, y, cov_type="HC3")
```

Preview robust covariance without setting:

```py
model.robust_se(type="HC3")
```