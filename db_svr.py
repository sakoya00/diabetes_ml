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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.datasets import load_diabetes

diabetes_data= load_diabetes()
print (diabetes_data.keys())
data1 = pd.DataFrame(data= np.c_[diabetes_data['data'], diabetes_data['target']],
                     columns= diabetes_data['feature_names'] + ['target'])
predictors= data1.drop('target', axis=1).values
target_df= data1['target'].values
X_train, X_test, y_train, y_test= train_test_split(predictors, target_df,test_size=0.30, random_state=42)

svr_reg = SVR(kernel = "rbf")
svr_reg.fit(X_train, y_train)

svr_reg_train= svr_reg.predict(X_train)
rmse_svr_0= np.sqrt(mean_squared_error(y_train, svr_reg_train))
print(rmse_svr_0)

svr_train_r2= r2_score(y_train, svr_reg_train)
print(svr_train_r2)

svr_reg_test= svr_reg.predict(X_test)
rmse_svr= np.sqrt(mean_squared_error(y_test, svr_reg_test))
print(rmse_linreg)

svr_test_r2= r2_score(y_test, svr_reg_test)
print(svr_test_r2)

