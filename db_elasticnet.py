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
X_train, X_test, y_train, y_test= train_test_split(predictors, target_df,test_size=0.30, random_state=42)

elastic_reg_0 = ElasticNetCV(cv=5, random_state=0)
elastic_reg_0.fit(predictors, target_df)
print(elastic_reg_0.alpha_) 
#best alpha is .004

elastic_reg_1 = ElasticNet(alpha = 0.004)
elastic_reg_1.fit(X_train, y_train) 
e_pred_train= elastic_reg_1.predict(X_train)
print("ElasticNet Train RMSE: %.2f" 
      % np.sqrt(mean_squared_error(y_train,e_pred_train)))
print("ElasticNet Train R^2 Score: %.2f" 
      % r2_score(y_train, e_pred_train))

e_pred_test= elastic_reg_1.predict(X_test)
print("ElasticNet Test RMSE: %.2f"
      % np.sqrt(mean_squared_error(y_test,e_pred_test)))
print("ElasticNet Test R^2 Score: %.2f"
      % r2_score(y_test, e_pred_test))
