# diabetes_ml

Predicted diabetes progression after 1 year using seven different regression ML techniques and the predictors (age, sex, BMI, etc) found in the LARS diabetes dataset. Selected best model of the seven (LASSO) using RMSE and R^2 on test data. 

## Required Packages/Modules
* pandas
* numpy
* sklearn
* matplotlib
* math

## Diabetes Predictors
* age in years
* sex
* BMI
* average blood pressure
* s1 tc, T-Cells (a type of white blood cells)
* s2 ldl, low-density lipoproteins
* s3 hdl, high-density lipoproteins
* s4 tch, thyroid stimulating hormone
* s5 ltg, lamotrigine
* s6 glu, blood sugar level

## Regression Techniques Used
* ElasticNet
* LASSO
* Ridge
* Linear Regression
* Polynomial Regression
* Support Vector Regression (SVR)
* Random Forest Regression
