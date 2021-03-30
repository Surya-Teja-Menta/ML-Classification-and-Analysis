
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
import math
from __future__ import division
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


# Read the data as DataFrame
data = pd.read_csv('kc_house_data.csv')

 
# Check the info about data set
print(len(data))
print(len(data.columns))
print(data.dtypes.unique())

 
# defining the type('O')
data.select_dtypes(include=['O']).columns.tolist()

 


 
# Check number of columns and data points with NaN
print(data.isnull().any().sum(), len(data.columns))
print(data.isnull().any(axis=1).sum(),len(data))

 
"""
- The data set is pretty and doesn't have any NaN values. So we can jump into finding correlations between the features and the target variable
"""

 


 
features = data.iloc[:,3:].columns.tolist()
target = data.iloc[:,2].name

 
correlations = {}
for f in features:
    data_temp = data[[f,target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' <=> ' + target
    correlations[key] = pearsonr(x1,x2)[0]

 
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]

 
"""
- We can see that the top 5 features are the most correlated features with the target "price"
- Let's plot the best 2 regressors jointly
"""

 
y = data.loc[:,['sqft_living','grade',target]].sort_values(target, ascending=True).values
x = np.arange(y.shape[0])

 
%matplotlib inline
plt.subplot(3,1,1)
plt.plot(x,y[:,0])
plt.title('Sqft and Grade vs Price')
plt.ylabel('Sqft')

plt.subplot(3,1,2)
plt.plot(x,y[:,1])
plt.ylabel('Grade')

plt.subplot(3,1,3)
plt.plot(x,y[:,2],'r')
plt.ylabel("Price")

plt.show()

 
# Train using linear regression model
model = LinearRegression()
new_data = data[['sqft_living','grade', 'sqft_above', 'sqft_living15','bathrooms','view','sqft_basement','lat','waterfront','yr_built','bedrooms']]

 
X = new_data.values
y = data.price.values

 
X_train, X_test, y_train, y_test =train_test_split(X, y ,test_size=0.2)

 
model.fit(X_train, y_train)
print(model.predict(X_test))

 
model.score(X_test,y_test)

 
"""
- Prediction score is about 69.7~70 which is not really optimal
"""

 
# Calculate the Root Mean Squared Error
print("RMSE: %.2f" % math.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))

 
# Let's try XGboost algorithm to see if we can get better results
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

 
traindf, testdf = train_test_split(X_train, test_size = 0.3)
xgb.fit(X_train,y_train)

 
predictions = xgb.predict(X_test)
print(explained_variance_score(predictions,y_test))

 
"""
- Our accuracy is changing between 70%-80%. I think it is close to an optimal solution.
"""

 
