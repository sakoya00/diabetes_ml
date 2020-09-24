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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.datasets import load_diabetes

diabetes_data= load_diabetes()
#print (diabetes_data.keys())
data1 = pd.DataFrame(data= np.c_[diabetes_data['data'], diabetes_data['target']],
                     columns= diabetes_data['feature_names'] + ['target'])
predictors= data1.drop('target', axis=1).values
target_df= data1['target'].values

lasso_0= Lasso()
lasso_alphas= {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0]}
lasso_0_reg= GridSearchCV(lasso_0, lasso_alphas, scoring= "neg_root_mean_squared_error")
lasso_0_reg.fit(predictors, target_df)
print(lasso_0_reg.best_params_)

#best alpha = 0.05
lasso_1_reg = Lasso(alpha=0.05)
lasso_1_reg.fit(X_train, y_train) 

lasso_pred_train= lasso_1_reg.predict(X_train)
print("Lasso Train RMSE: %.2f"
      % np.sqrt(mean_squared_error(y_train, lasso_pred_train)))
print("Lasso Train R^2 Score: %.2f"
      % r2_score(y_train, lasso_pred_train))

lasso_pred_test= lasso_1_reg.predict(X_test)
print("Lasso Test RMSE: %.2f"
      % np.sqrt(mean_squared_error(y_test, lasso_pred_test)))
print("Lasso Test R^2 Score: %.2f"
      % r2_score(y_test, lasso_pred_test))
