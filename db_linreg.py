import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.datasets import load_diabetes

diabetes_data= load_diabetes()
#print (diabetes_data.keys())
data1 = pd.DataFrame(data= np.c_[diabetes_data['data'], diabetes_data['target']],
                     columns= diabetes_data['feature_names'] + ['target'])
predictors= data1.drop('target', axis=1).values
target_df= data1['target'].values
X_train, X_test, y_train, y_test= train_test_split(predictors, target_df,test_size=0.30, random_state=42)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

linreg_pred_train= linreg.predict(X_train)
print("Linear Regression Train RMSE: %.2f"
      % np.sqrt(mean_squared_error(y_train, linreg_pred_train)))
print("Linear Regression R^2 Score: %.2f"
      % r2_score(y_train, linreg_pred_train))

linreg_pred_test= linreg.predict(X_test)
print("Linear Regression Test RMSE: %.2f"
      % np.sqrt(mean_squared_error(y_test, linreg_pred_test)))
print("Linear Regression Test R^2 Score: %.2f"
      % r2_score(y_test, linreg_pred_test))

plt.plot(y_test, linreg_pred,'.')
plt.plot(y_test, y_test, color="red")
plt.xlabel("Test Set Diabetes Progression")
plt.ylabel("Predicted Diabetes Progression ")
plt.show()
