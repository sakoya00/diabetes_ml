import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.datasets import load_diabetes

diabetes_data= load_diabetes()
#print (diabetes_data.keys())
data1 = pd.DataFrame(data= np.c_[diabetes_data["data"], diabetes_data["target"]],
                     columns= diabetes_data["feature_names"] + ["target"])
predictors= data1.drop("target", axis=1).values
target_df= data1['target'].values

ridge_try_cv = RidgeCV(alphas=[0.01, 0.05, 0.1, 0.5, 1.0])
ridge_reg0 = ridge_try_cv.fit(predictors, target_df)
ridge_reg0.alpha_

X_train, X_test, y_train, y_test= train_test_split(predictors, target_df,test_size=0.30, random_state=42)

ridge_reg1 = Ridge(alpha=0.01)
ridge_reg1.fit(X_train, y_train)
ridge_pred_train= ridge_reg1.predict(X_train)
ridge_pred_test= ridge_reg1.predict(X_test)

print("Ridge Train RMSE: %.2f"
      % np.sqrt(mean_squared_error(y_train, ridge_pred_train)))
print("Ridge Train R^2 Score: %.2f"
      % r2_score(y_train, ridge_pred_train))

print("Ridge Test RMSE: %.2f"
      % np.sqrt(mean_squared_error(y_test, ridge_pred_test)))
print("Ridge Test R^2 Score: %.2f"
      % r2_score(y_test, ridge_pred_test))
