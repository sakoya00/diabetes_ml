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
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.datasets import load_diabetes

diabetes_data= load_diabetes()
print (diabetes_data.keys())
data1 = pd.DataFrame(data= np.c_[diabetes_data['data'], diabetes_data['target']],
                     columns= diabetes_data['feature_names'] + ['target'])
predictors= data1.drop('target', axis=1).values
target_df= data1['target'].values
